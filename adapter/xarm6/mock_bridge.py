"""A standalone fake xArm6 bridge, so the whole loop runs with no robot and no tunnel."""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# This is a stand-in for the robot PC's bridge, so it speaks the xArm's conventions, not
# the controller's neutral ones.
from adapter.xarm6.conventions import (  # noqa: E402
    _wrap as wrap_to_pi,
    flange_pose_from_bridge,
    flange_to_tool,
)

try:
    import cv2
except ImportError:  # frames still carry K and cam2base, just no pixels
    cv2 = None

CONFIRMATION = "I_UNDERSTAND_ROBOT_WILL_MOVE"
HEARTBEAT_TIMEOUT_S = 2.0
TABLE_Z_M = 0.05
# The rebuilt rig bridge reports the BARE FLANGE (no tool offset on the arm), and the host adds
# the 172 mm tool point itself.
TCP_OFFSET_MM = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
HOST_ADDS_M = 0.172
CAMERAS = ("scene", "wrist", "side")
GRIPPER_OPEN_RAW = 850
GRIPPER_CLOSE_RAW = 0
# The jaw gap, in metres, when fully open.
OPEN_GAP_M = 0.0889
REACH_MAX_M = 0.70
REACH_MIN_M = 0.20

LIMITS = {
    "max_translation_mm": 50.0,
    "max_rotation_rad": 0.1,
    "max_speed_mm_s": 20.0,
    "min_duration_s": 0.2,
    "max_duration_s": 120.0,
}

HOME_POSE = [400.0, 0.0, 400.0, math.pi, 0.0, 0.0]   # flange, mm + rad


def look_at_cam2base(eye, target):
    """A camera at ``eye`` looking at ``target``, in the OpenCV optical convention."""
    eye = np.asarray(eye, dtype=float)
    forward = np.asarray(target, dtype=float) - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    if down[2] > 0:  # keep the image upright: world +z must appear image-up
        right, down = -right, -down
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = right, down, forward
    T[:3, 3] = eye
    return T


class FakeArm:
    """The flange pose, the jaws, and one job at a time."""

    def __init__(self, latency_ms: float = 0.0, time_scale: float = 1.0):
        self.lock = threading.RLock()
        self.time_scale = max(1e-3, float(time_scale))
        self.flange = list(HOME_POSE)
        self.gap_m = OPEN_GAP_M
        self.gripper_command = "open"
        self.gripper_moving_until = 0.0
        self.state = 2
        self.error_code = 0
        self.collision = False
        self.parked = False
        self.jobs = {}
        self.current = None
        self.latency_s = latency_ms / 1000.0
        self.base_currents = [0.1, 0.9, 0.4, 0.1, 0.1, 0.0]
        self.currents = list(self.base_currents)
        threading.Thread(target=self._run, name="fake-arm", daemon=True).start()

    # ---------------------------------------------------------------- helpers

    def grip_position(self, flange=None):
        """The control point: the reported pose, plus whatever the host adds to it."""
        return flange_to_tool(flange_pose_from_bridge(flange or self.flange),
                              HOST_ADDS_M).position_m

    def validate_cartesian(self, body):
        try:
            target = [float(v) for v in body["target_pose"]]
            duration = float(body["duration"])
        except (KeyError, TypeError, ValueError):
            return False, "body needs target_pose (6 numbers) and duration"
        if len(target) != 6:
            return False, "target_pose needs 6 numbers, got {}".format(len(target))
        if not (LIMITS["min_duration_s"] <= duration <= LIMITS["max_duration_s"]):
            return False, "duration {:.2f} s outside {:.1f}..{:.0f} s".format(
                duration, LIMITS["min_duration_s"], LIMITS["max_duration_s"])
        with self.lock:
            start = list(self.flange)
        translation = math.dist(target[:3], start[:3])
        if translation > LIMITS["max_translation_mm"]:
            return False, "translation {:.1f} mm over the {:.0f} mm limit".format(
                translation, LIMITS["max_translation_mm"])
        for name, a, b in zip("roll pitch yaw".split(), start[3:], target[3:]):
            delta = abs(wrap_to_pi(b - a))
            if delta > LIMITS["max_rotation_rad"]:
                return False, "{} changes by {:.3f} rad, over the {:.2f} rad limit".format(
                    name, delta, LIMITS["max_rotation_rad"])
        speed = translation / duration
        if speed > LIMITS["max_speed_mm_s"] + 1e-6:
            return False, "{:.1f} mm in {:.2f} s is {:.1f} mm/s, over {:.0f}".format(
                translation, duration, speed, LIMITS["max_speed_mm_s"])
        reach = np.linalg.norm(np.asarray(target[:3]) / 1000.0)
        if reach > REACH_MAX_M:
            return False, "target {:.0f} mm from the base, outside the {:.0f} mm reach".format(
                reach * 1000, REACH_MAX_M * 1000)
        if reach < REACH_MIN_M:
            return False, "target {:.0f} mm from the base, inside the {:.0f} mm dead zone".format(
                reach * 1000, REACH_MIN_M * 1000)
        return True, {"valid": True, "duration_s": duration,
                      "translation_mm": round(translation, 2),
                      "rotation_rad": round(max(abs(wrap_to_pi(b - a))
                                                for a, b in zip(start[3:], target[3:])), 4)}

    def start(self, kind, body):
        with self.lock:
            if self.parked:
                return 503, {"ok": False, "reason": "the arm is parked after a stopped job; "
                                                    "POST /stop to clear it"}
            if self.current is not None and self.jobs[self.current]["status"] in ("queued", "running"):
                return 409, {"ok": False, "reason": "a job is already running"}
            if kind == "cartesian":
                ok, detail = self.validate_cartesian(body)
                if not ok:
                    return 400, {"ok": False, "reason": detail}
                target = [float(v) for v in body["target_pose"]]
                duration = float(body["duration"])
            else:  # home
                target = list(HOME_POSE)
                duration = 4.0
            job_id = uuid.uuid4().hex[:12]
            self.jobs[job_id] = {
                "id": job_id, "kind": kind, "status": "queued",
                "progress": {"fraction": 0.0}, "contact": False, "contact_evidence": None,
                "message": "", "elapsed_s": 0.0,
                "_start_pose": list(self.flange), "_target": target, "_duration": duration,
                "_t0": None, "_heartbeat": time.time(),
                "_stop_on_contact": bool(body.get("stop_on_contact")),
                "_contact": body.get("contact") or {},
            }
            self.current = job_id
            return 200, {"id": job_id, "status": "queued"}

    def heartbeat(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return 404, {"ok": False, "reason": "unknown job"}
            job["_heartbeat"] = time.time()
            return 200, {"ok": True}

    def stop(self):
        with self.lock:
            job = self.jobs.get(self.current)
            if job and job["status"] in ("queued", "running"):
                job["status"] = "aborted"
                job["message"] = "stopped by the host"
            self.parked = False
            self.state = 2
            self.error_code = 0
            self.collision = False
            self.current = None
            return 200, {"ok": True}

    def set_gripper(self, command):
        with self.lock:
            self.gripper_command = command
            self.gripper_moving_until = time.time() + 1.2
            return 200, {"ok": True, "command": command}

    # ---------------------------------------------------------------- the loop

    def _run(self):
        while True:
            time.sleep(0.01)
            now = time.time()
            with self.lock:
                if self.gripper_moving_until and now > self.gripper_moving_until:
                    self.gap_m = OPEN_GAP_M if self.gripper_command == "open" else 0.0
                    self.gripper_moving_until = 0.0
                job = self.jobs.get(self.current)
                if job is None or job["status"] in ("complete", "stopped", "failed", "aborted"):
                    self.currents = list(self.base_currents)
                    continue
                if job["status"] == "queued":
                    job["status"] = "running"
                    job["_t0"] = now
                if now - job["_heartbeat"] > HEARTBEAT_TIMEOUT_S:
                    job["status"] = "stopped"
                    job["message"] = "heartbeat lost"
                    self.parked, self.state = True, 4
                    continue
                elapsed = now - job["_t0"]
                fraction = min(1.0, elapsed * self.time_scale / job["_duration"])
                smooth = 0.5 * (1.0 - math.cos(math.pi * fraction))  # the bridge's cosine profile
                start, target = job["_start_pose"], job["_target"]
                pose = [a + (b - a) * smooth for a, b in zip(start[:3], target[:3])]
                pose += [a + wrap_to_pi(b - a) * smooth for a, b in zip(start[3:], target[3:])]
                self.flange = pose
                job["progress"] = {"fraction": round(fraction, 3)}
                job["elapsed_s"] = round(elapsed, 3)
                self.state = 0

                descending = target[2] < start[2] - 1e-6
                grip_z = float(self.grip_position(pose)[2])
                if descending:
                    # the table takes the arm's weight: J2/J3 gravity current FALLS
                    load = max(0.0, 1.0 - max(0.0, grip_z - TABLE_Z_M) / 0.05)
                    self.currents = list(self.base_currents)
                    self.currents[1] -= 0.8 * load
                    self.currents[2] -= 0.4 * load
                if job["_stop_on_contact"] and grip_z <= TABLE_Z_M:
                    job["status"] = "stopped"
                    job["contact"] = True
                    job["contact_evidence"] = {
                        "source": "joint_current", "joint": 1,
                        "delta_a": round(self.currents[1] - self.base_currents[1], 3),
                        "fraction": round(fraction, 3),
                    }
                    job["message"] = "contact"
                    self.parked, self.state, self.collision = True, 4, True
                    continue
                if fraction >= 1.0:
                    job["status"] = "complete"
                    self.state = 2
                    self.current = None

    # ---------------------------------------------------------------- reads

    def state_document(self):
        with self.lock:
            job = self.jobs.get(self.current)
            return {
                "joint_rad": [0.0, -0.042, -1.165, 0.0, 1.264, 0.0],
                "joint_vel_rad_s": [0.0] * 6,
                "tcp_pose": [round(v, 4) for v in self.flange],
                # the real bridge reports a raw count and, in /health, its two endpoints --
                # enough to tell an empty close from a full one without a width calibration
                "gripper": {"gap_m": self.gap_m,
                            "raw": round(GRIPPER_CLOSE_RAW + (GRIPPER_OPEN_RAW - GRIPPER_CLOSE_RAW)
                                         * (self.gap_m / OPEN_GAP_M)),
                            "moving": bool(self.gripper_moving_until),
                            "command": self.gripper_command},
                "joint_current_a": [round(c, 3) for c in self.currents],
                "collision": self.collision,
                "error_code": self.error_code,
                "state": self.state,
                "timestamp": time.time(),
                "axis_count": 6,
                "job": job["id"] if job and job["status"] == "running" else None,
            }


class Cameras:
    """Three pinhole cameras (scene, side, wrist) that actually see the gripper, so extrinsics can be checked."""

    def __init__(self, arm: FakeArm):
        self.arm = arm
        # In line with the base +x axis, elevated, tilted down about 40 degrees -- the placement
        # adapter/xarm6/README.md section 5 asks the rig for, and the one where robot -forward and
        # robot-left land on different image axes.
        self.scene_cam2base = look_at_cam2base((1.05, 0.0, 0.70), (0.40, 0.0, 0.18))
        # A third view from the +y side, as the rig bridge advertises.
        self.side_cam2base = look_at_cam2base((0.40, 0.85, 0.45), (0.40, 0.0, 0.15))
        # The wrist camera sits BEHIND the tool point looking along it (+z), so the thing it is
        # there to watch is in front of it.
        self.flange_T_cam = np.array([[0.0, -1.0, 0.0, 0.0],
                                      [1.0, 0.0, 0.0, -0.02],
                                      [0.0, 0.0, 1.0, -0.15],
                                      [0.0, 0.0, 0.0, 1.0]])
        self.K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
        self.native = (640, 480)

    def cam2base(self, name):
        if name == "scene":
            return self.scene_cam2base
        if name == "side":
            return self.side_cam2base
        with self.arm.lock:
            pose = flange_pose_from_bridge(self.arm.flange)
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = pose.rotation, pose.position_m
            return T @ self.flange_T_cam

    def render(self, name, size):
        cam2base = self.cam2base(name)
        width, height = self.native
        K = self.K.copy()
        if size:
            x0, y0 = (width - height) // 2, 0
            K[0, 2] -= x0
            K[1, 2] -= y0
            K[:2, :] *= size / float(height)
            width = height = size
        if cv2 is None:
            return None, K, cam2base, width, height

        # A legible synthetic scene.
        image = np.full((height, width, 3), 26, np.uint8)   # RGB; flipped to BGR at encode
        base2cam = np.linalg.inv(cam2base)

        def project(point):
            q = base2cam @ np.array([point[0], point[1], point[2], 1.0])
            if q[2] <= 1e-3:
                return None
            uv = K @ (q[:3] / q[2])
            if not (-4 * width < uv[0] < 4 * width and -4 * height < uv[1] < 4 * height):
                return None
            return int(round(uv[0])), int(round(uv[1]))

        def line(a, b, colour, thickness=1):
            pa, pb = project(a), project(b)
            if pa and pb:
                cv2.line(image, pa, pb, colour, thickness, cv2.LINE_AA)

        x_range, y_range = (0.15, 0.80), (-0.35, 0.35)
        quad = [project((x, y, TABLE_Z_M)) for x, y in
                ((x_range[0], y_range[0]), (x_range[1], y_range[0]),
                 (x_range[1], y_range[1]), (x_range[0], y_range[1]))]
        if all(point is not None for point in quad):
            cv2.fillPoly(image, [np.array(quad, np.int32)], (44, 52, 62))

        for x in np.arange(x_range[0], x_range[1] + 1e-6, 0.05):
            line((x, y_range[0], TABLE_Z_M), (x, y_range[1], TABLE_Z_M), (66, 78, 92))
        for y in np.arange(y_range[0], y_range[1] + 1e-6, 0.05):
            line((x_range[0], y, TABLE_Z_M), (x_range[1], y, TABLE_Z_M), (66, 78, 92))

        # the base frame's own axes, so the view can be read without a legend
        for axis, colour, label in (((0.18, 0, 0), (232, 96, 88), "+x"),
                                    ((0, 0.18, 0), (120, 208, 132), "+y"),
                                    ((0, 0, 0.18), (110, 168, 255), "+z")):
            line((0, 0, 0), axis, colour, 2)
            tip = project(axis)
            if tip:
                cv2.putText(image, label, (tip[0] + 3, tip[1] - 3), cv2.FONT_HERSHEY_SIMPLEX,
                            0.3 * width / 256.0, colour, 1, cv2.LINE_AA)

        # the gripper: the control point, with a stalk back to the reported pose
        with self.arm.lock:
            pose = list(self.arm.flange)
        control = self.arm.grip_position(pose)
        line((pose[0] / 1000.0, pose[1] / 1000.0, pose[2] / 1000.0), control, (150, 160, 175), 2)
        marker = project(control)
        if marker:
            radius = max(3, int(7 * width / 256.0))
            cv2.circle(image, marker, radius, (250, 196, 90), -1, cv2.LINE_AA)
            cv2.circle(image, marker, radius + 3, (250, 196, 90), 1, cv2.LINE_AA)
        shadow = project((control[0], control[1], TABLE_Z_M))
        if shadow:
            cv2.circle(image, shadow, max(2, int(4 * width / 256.0)), (90, 100, 116), 1,
                       cv2.LINE_AA)

        scale = width / 256.0
        cv2.putText(image, "MOCK {}".format(name), (int(6 * scale), int(14 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36 * scale, (150, 160, 175), 1, cv2.LINE_AA)
        cv2.putText(image, "z {:+.0f}mm".format(control[2] * 1000),
                    (int(6 * scale), height - int(7 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34 * scale, (150, 160, 175), 1, cv2.LINE_AA)
        return image, K, cam2base, width, height

    def document(self, name, size):
        image, K, cam2base, width, height = self.render(name, size)
        out = {
            "depth_scale": 0.001,
            "intrinsic": K.tolist(),
            "cam2base": cam2base.tolist(),
            "width": width, "height": height,
            "timestamp": time.time(),
        }
        if image is not None:
            ok, buf = cv2.imencode(".jpg", image[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ok:
                out["rgb_jpeg"] = base64.b64encode(buf.tobytes()).decode("ascii")
            depth = np.full((height, width), 800, np.uint16)
            ok, buf = cv2.imencode(".png", depth, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            if ok:
                out["depth_png16"] = base64.b64encode(buf.tobytes()).decode("ascii")
        return out


class Handler(BaseHTTPRequestHandler):
    arm: FakeArm = None
    cameras: Cameras = None
    token = ""

    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        if os.environ.get("MOCK_BRIDGE_VERBOSE"):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorised(self, motion):
        if self.token and self.headers.get("X-Bridge-Token") != self.token:
            self._send(401, {"ok": False, "reason": "bad or missing X-Bridge-Token"})
            return False
        if motion and self.headers.get("X-Motion-Confirmation") != CONFIRMATION:
            self._send(403, {"ok": False, "reason": "X-Motion-Confirmation header required"})
            return False
        return True

    def _body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except ValueError:
            return {}

    def do_GET(self):
        parsed = urlparse(self.path)
        path, query = parsed.path, parse_qs(parsed.query)
        if path == "/health":
            with self.arm.lock:
                robot = {"connected": True, "axis_count": 6, "mode": 1,
                         "state": self.arm.state, "error_code": self.arm.error_code,
                         "tcp_offset_mm": list(TCP_OFFSET_MM)}
            return self._send(200, {"ok": True, "backend": "mock", "robot": robot,
                                    "cameras": CAMERAS,
                                    "heartbeat_timeout_s": HEARTBEAT_TIMEOUT_S,
                                    "motion_ready": not self.arm.parked,
                                    "limits": LIMITS, "calibration_ready": True,
                                    "tcp_frame": "flange",
                                    "gripper_config": {"open_raw": GRIPPER_OPEN_RAW,
                                                       "close_raw": GRIPPER_CLOSE_RAW,
                                                       "width_measurement_method": None}})
        if not self._authorised(False):
            return
        if path == "/state":
            if self.arm.latency_s:
                time.sleep(self.arm.latency_s)
            return self._send(200, self.arm.state_document())
        if path == "/frames":
            if self.arm.latency_s:
                time.sleep(self.arm.latency_s)
            names = (query.get("cameras", [",".join(CAMERAS)])[0]).split(",")
            size = int(query.get("size", [0])[0] or 0)
            out = {}
            for name in names:
                if name not in CAMERAS:
                    return self._send(404, {"ok": False, "reason": "unknown camera " + name})
                out[name] = self.cameras.document(name, size)
            return self._send(200, {"cameras": out})
        for kind in ("cartesian", "trajectory", "home"):
            prefix = "/{}/job/".format(kind)
            if path.startswith(prefix):
                job = self.arm.jobs.get(path[len(prefix):])
                if job is None:
                    return self._send(404, {"ok": False, "reason": "unknown job"})
                return self._send(200, {k: v for k, v in job.items() if not k.startswith("_")})
        return self._send(404, {"ok": False, "reason": "no route " + path})

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/stop":
            if not self._authorised(False):
                return
            code, payload = self.arm.stop()
            return self._send(code, payload)
        motion = path in ("/cartesian/start", "/home", "/gripper", "/trajectory/start")
        if not self._authorised(motion):
            return
        body = self._body()
        if path == "/cartesian/validate":
            ok, detail = self.arm.validate_cartesian(body)
            return self._send(200, detail if ok else {"valid": False, "reason": detail})
        if path == "/cartesian/start":
            code, payload = self.arm.start("cartesian", body)
            return self._send(code, payload)
        if path == "/home":
            code, payload = self.arm.start("home", body)
            return self._send(code, payload)
        if path == "/gripper":
            command = str(body.get("command", "")).lower()
            if command not in ("open", "close"):
                return self._send(400, {"ok": False, "reason": "command must be open or close"})
            code, payload = self.arm.set_gripper(command)
            return self._send(code, payload)
        for kind in ("cartesian", "trajectory", "home"):
            prefix, suffix = "/{}/job/".format(kind), "/heartbeat"
            if path.startswith(prefix) and path.endswith(suffix):
                code, payload = self.arm.heartbeat(path[len(prefix):-len(suffix)])
                return self._send(code, payload)
        return self._send(404, {"ok": False, "reason": "no route " + path})


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=18766)
    parser.add_argument("--token", default=os.environ.get("XARM_BRIDGE_TOKEN", ""))
    parser.add_argument("--latency-ms", type=float, default=0.0,
                        help="added to /state and /frames, to imitate the tunnel")
    parser.add_argument("--time-scale", type=float, default=1.0,
                        help="play jobs this many times faster than their duration (tests)")
    args = parser.parse_args()

    Handler.arm = FakeArm(args.latency_ms, args.time_scale)
    Handler.cameras = Cameras(Handler.arm)
    Handler.token = args.token
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print("mock xArm6 bridge on http://127.0.0.1:{}{}".format(
        args.port, " (token required)" if args.token else " (no token)"))
    print("table at z = {:.0f} mm, reach {:.0f}..{:.0f} mm, {}".format(
        TABLE_Z_M * 1000, REACH_MIN_M * 1000, REACH_MAX_M * 1000,
        "frames with pixels" if cv2 is not None else "frames without pixels (no cv2)"))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
