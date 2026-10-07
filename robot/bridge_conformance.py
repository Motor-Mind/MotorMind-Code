#!/usr/bin/env python3
"""Check an xArm bridge against the contract in adapter/xarm6/README.md.

    python3 robot/bridge_conformance.py --url http://127.0.0.1:18765 --token "$XARM_BRIDGE_TOKEN"
    python3 robot/bridge_conformance.py --url ... --token ... --gripper          # the jaws move
    python3 robot/bridge_conformance.py --url ... --token ... --move             # THE ARM MOVES
    python3 robot/bridge_conformance.py --url ... --token ... --extrinsic-check  # THE ARM MOVES
"""

import argparse
import base64
import json
import math
import struct
import sys
import time
import urllib.error
import urllib.request

CONFIRMATION = "I_UNDERSTAND_ROBOT_WILL_MOVE"
MOTION_ROUTES = ("/cartesian/start", "/trajectory/start", "/home", "/gripper")

EXPECTED_TCP_OFFSET_MM = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# Filled in from /health: the rig grew a third camera ("side") between two bridge builds, so
# nothing here may assume which cameras exist.
CAMERAS = ["scene", "wrist"]

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
results = []


def record(level, what, detail=""):
    results.append((level, what, detail))
    colour = {PASS: "\033[32m", WARN: "\033[33m", FAIL: "\033[31m"}.get(level, "")
    print("{}{:4}\033[0m {}{}".format(colour, level, what, "  -- " + detail if detail else ""))
    return level == PASS


def check(condition, what, detail_ok="", detail_bad="", soft=False):
    if condition:
        return record(PASS, what, detail_ok)
    return record(WARN if soft else FAIL, what, detail_bad or detail_ok)


class Bridge:
    def __init__(self, url, token):
        self.url = url.rstrip("/")
        self.token = token

    def call(self, method, path, body=None, token=True, confirm=None, timeout=15.0):
        headers = {"Content-Type": "application/json"}
        if token and self.token:
            headers["X-Bridge-Token"] = self.token
        want_confirm = path in MOTION_ROUTES if confirm is None else confirm
        if want_confirm:
            headers["X-Motion-Confirmation"] = CONFIRMATION
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(self.url + path, data, headers, method=method)
        started = time.time()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as reply:
                raw = reply.read().decode("utf-8")
                return reply.status, (json.loads(raw) if raw.strip() else {}), time.time() - started
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", "replace")
            try:
                payload = json.loads(raw)
            except ValueError:
                payload = {"raw": raw}
            return exc.code, payload, time.time() - started
        except urllib.error.URLError as exc:
            return 0, {"error": str(exc.reason)}, time.time() - started


# --------------------------------------------------------------------------- helpers


def png_size(blob):
    """Width, height and bit depth straight out of a PNG header -- no decoder needed."""
    if len(blob) < 26 or blob[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width, height = struct.unpack(">II", blob[16:24])
    return width, height, blob[24], blob[25]


def is_jpeg(blob):
    return len(blob) > 3 and blob[:2] == b"\xff\xd8" and blob[-2:] == b"\xff\xd9"


def orthonormal(R):
    """Is this 3x3 a rotation?  Columns unit length, mutually perpendicular, det +1."""
    for column in range(3):
        norm = math.sqrt(sum(R[row][column] ** 2 for row in range(3)))
        if abs(norm - 1.0) > 1e-3:
            return False, "column {} has length {:.4f}".format(column, norm)
    for a in range(3):
        for b in range(a + 1, 3):
            dot = sum(R[row][a] * R[row][b] for row in range(3))
            if abs(dot) > 1e-3:
                return False, "columns {} and {} are not perpendicular ({:.4f})".format(a, b, dot)
    det = (R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1])
           - R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0])
           + R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]))
    if abs(det - 1.0) > 1e-3:
        return False, "determinant {:.4f}, not +1 (a transposed or mirrored frame)".format(det)
    return True, "orthonormal, det +1"


# What the HOST adds to tcp_pose to reach the control point.
GRIP_SITE_OFFSET_M = 0.172


def rpy_to_matrix(roll, pitch, yaw):
    """The xArm's convention: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]]


def grip_site(tcp_pose):
    """The control point in metres: tcp_pose plus whatever the host adds along its own +z.
    The extrinsic check projects THIS."""
    R = rpy_to_matrix(*tcp_pose[3:])
    return [tcp_pose[i] / 1000.0 + R[i][2] * GRIP_SITE_OFFSET_M for i in range(3)]


def project(point_base, cam2base, K):
    """Where a base-frame point lands in the image, given the served extrinsic and K."""
    R = [row[:3] for row in cam2base[:3]]
    t = [row[3] for row in cam2base[:3]]
    d = [point_base[i] - t[i] for i in range(3)]
    cam = [sum(R[row][col] * d[row] for row in range(3)) for col in range(3)]  # R^T d
    if cam[2] <= 1e-6:
        return None
    u = K[0][0] * cam[0] / cam[2] + K[0][2]
    v = K[1][1] * cam[1] / cam[2] + K[1][2]
    return u, v


# --------------------------------------------------------------------------- sections


def check_health(bridge):
    status, health, elapsed = bridge.call("GET", "/health")
    if not check(status == 200, "GET /health answers", "{:.0f} ms".format(elapsed * 1000),
                 "status {} {}".format(status, health)):
        return None
    robot = health.get("robot") or {}
    check(robot.get("connected") is True, "the arm is connected")
    check(robot.get("axis_count") == 6, "axis_count is 6",
          detail_bad="got {!r}".format(robot.get("axis_count")))
    offset = robot.get("tcp_offset_mm")
    expected = EXPECTED_TCP_OFFSET_MM
    ok = (isinstance(offset, list) and len(offset) == len(expected)
          and all(abs(float(a) - float(b)) < 1.0 for a, b in zip(offset, expected)))
    check(ok, "tcp_offset_mm is the tool point the host plans for",
          "{} mm".format([round(float(v), 1) for v in offset or []]),
          "got {!r}, host expects {} -- every action is relative to this point, so the two "
          "sides must agree on it (--expect-tcp-offset-mm)".format(offset, expected))
    global CAMERAS
    CAMERAS = [str(c) for c in (health.get("cameras") or [])]
    missing = [name for name in ("scene", "wrist") if name not in CAMERAS]
    check(not missing, "the cameras are advertised", ", ".join(CAMERAS) or "none",
          "missing {} (got {!r})".format(", ".join(missing), CAMERAS))
    timeout = health.get("heartbeat_timeout_s")
    check(isinstance(timeout, (int, float)) and 0.5 <= timeout <= 10,
          "heartbeat_timeout_s is advertised", "{} s".format(timeout),
          "got {!r}".format(timeout))
    check(isinstance(health.get("limits"), dict), "limits are advertised",
          json.dumps(health.get("limits")), "no limits block", soft=True)
    return health


def check_auth(bridge):
    if not bridge.token:
        record(WARN, "token checks skipped", "no --token given; the bridge must require one")
        return
    saved = bridge.token
    bridge.token = "definitely-not-the-token"
    status, _, _ = bridge.call("GET", "/state")
    check(status == 401, "a wrong token is refused on /state (401)",
          detail_bad="got {}".format(status))
    bridge.token = saved

    status, _, _ = bridge.call("POST", "/cartesian/start", {"target_pose": [0] * 6, "duration": 1},
                               confirm=False)
    check(status == 403, "a motion route without the confirmation header is refused (403)",
          detail_bad="got {} -- the header is the last thing between a stray request and "
                     "the arm moving".format(status))

    status, payload, _ = bridge.call("POST", "/stop")
    check(status == 200 and payload.get("ok") is True,
          "POST /stop works with no confirmation header",
          detail_bad="got {} {} -- /stop must never be blocked".format(status, payload))

    status, _, _ = bridge.call("GET", "/cartesian/job/definitely-not-a-job")
    check(status == 404, "an unknown job id is 404", detail_bad="got {}".format(status))


def check_state(bridge, samples=20):
    latencies = []
    state = None
    for _ in range(samples):
        status, state, elapsed = bridge.call("GET", "/state")
        if status != 200:
            return check(False, "GET /state answers", detail_bad="status {}".format(status)) and None
        latencies.append(elapsed * 1000)
    latencies.sort()
    median = latencies[len(latencies) // 2]
    check(median < 60, "/state latency", "median {:.0f} ms, worst {:.0f} ms".format(median, latencies[-1]),
          "median {:.0f} ms -- the host polls at 10 Hz".format(median), soft=median < 150)

    for field, kind, length in [("joint_rad", list, 6), ("joint_vel_rad_s", list, 6),
                                ("tcp_pose", list, 6), ("joint_current_a", list, 6)]:
        value = state.get(field)
        check(isinstance(value, kind) and len(value) == length,
              "/state.{} is {} numbers".format(field, length),
              detail_bad="got {!r}".format(value))
    check(all(abs(float(v)) <= 7.0 for v in state.get("joint_rad", [])),
          "joint_rad is in RADIANS", detail_bad="values look like degrees: {}".format(
              state.get("joint_rad")))
    pose = state.get("tcp_pose") or [0] * 6
    check(max(abs(float(v)) for v in pose[:3]) > 5.0,
          "tcp_pose translation is in MILLIMETRES",
          "{:.0f}, {:.0f}, {:.0f} mm".format(*pose[:3]),
          "got {} -- metres would be a factor of 1000 out".format(pose[:3]))
    check(all(abs(float(v)) <= math.pi + 1e-3 for v in pose[3:]),
          "tcp_pose rpy is in radians within (-pi, pi]",
          detail_bad="got {}".format(pose[3:]))
    gripper = state.get("gripper") or {}
    gap = gripper.get("gap_m")
    check(isinstance(gap, (int, float)) and 0.0 <= gap <= 0.15,
          "gripper.gap_m is a gap in METRES", "{:.4f} m".format(gap or 0.0),
          "got {!r}".format(gap))
    check(isinstance(state.get("timestamp"), (int, float))
          and abs(state["timestamp"] - time.time()) < 60,
          "timestamp is time.time() seconds",
          detail_bad="got {!r}; the host maps the PC clock onto its own with it"
                     .format(state.get("timestamp")))
    for field in ("collision", "error_code", "state", "axis_count"):
        check(field in state, "/state carries {}".format(field))
    return state


def check_frames(bridge):
    wanted = ",".join(CAMERAS)
    status, native, elapsed = bridge.call("GET", "/frames?cameras=" + wanted)
    if not check(status == 200, "GET /frames (native) answers",
                 "{:.0f} ms".format(elapsed * 1000), "status {} {}".format(status, native)):
        return
    status, small, elapsed = bridge.call("GET", "/frames?cameras=" + wanted + "&size=256")
    check(status == 200, "GET /frames?size=256 answers", "{:.0f} ms".format(elapsed * 1000),
          "status {}".format(status))
    check(elapsed < 0.4, "/frames?size=256 latency",
          "{:.0f} ms for {} cameras".format(elapsed * 1000, len(CAMERAS)),
          "{:.0f} ms -- the host polls at 2-3 Hz".format(elapsed * 1000), soft=True)

    status, _, _ = bridge.call("GET", "/frames?cameras=nosuchcamera")
    check(status == 404, "an unknown camera name is 404", detail_bad="got {}".format(status))

    for name in CAMERAS:
        for label, payload, expect in (("native", native, None), ("256", small, 256)):
            camera = (payload.get("cameras") or {}).get(name)
            if camera is None:
                check(False, "{} {}: served".format(name, label))
                continue
            jpeg = base64.b64decode(camera.get("rgb_jpeg") or "")
            check(is_jpeg(jpeg), "{} {}: rgb_jpeg is a JPEG".format(name, label),
                  "{:.0f} kB".format(len(jpeg) / 1024.0))
            depth = base64.b64decode(camera.get("depth_png16") or "")
            info = png_size(depth)
            check(info is not None and info[2] == 16,
                  "{} {}: depth_png16 is a 16-bit PNG".format(name, label),
                  "" if info is None else "{}x{}, {} bit".format(*info[:3]),
                  "not a 16-bit PNG: {}".format(info))
            if info and expect:
                check(info[0] == expect and info[1] == expect,
                      "{} {}: depth is {}x{}".format(name, label, expect, expect),
                      detail_bad="got {}x{}".format(info[0], info[1]))
                check(camera.get("width") == expect and camera.get("height") == expect,
                      "{} {}: width/height match the request".format(name, label),
                      detail_bad="got {}x{}".format(camera.get("width"), camera.get("height")))

            K = camera.get("intrinsic")
            check(isinstance(K, list) and len(K) == 3 and abs(K[2][2] - 1.0) < 1e-6,
                  "{} {}: intrinsic is a 3x3 K".format(name, label),
                  detail_bad="got {!r}".format(K))
            cam2base = camera.get("cam2base")
            if not check(isinstance(cam2base, list) and len(cam2base) == 4,
                         "{} {}: cam2base is a 4x4".format(name, label),
                         detail_bad="got {!r} -- a camera frame cannot be used without it"
                                    .format(cam2base)):
                continue
            ok, detail = orthonormal([row[:3] for row in cam2base[:3]])
            check(ok, "{} {}: cam2base rotation is a rotation".format(name, label), detail, detail)
            translation = [cam2base[i][3] for i in range(3)]
            check(max(abs(v) for v in translation) < 5.0,
                  "{} {}: cam2base translation is in METRES".format(name, label),
                  "[{:.3f}, {:.3f}, {:.3f}] m".format(*translation),
                  "got {} -- millimetres would be 1000x out".format(translation))

        if expect_scale_check(native, small, name):
            record(PASS, "{}: the intrinsic follows the centre-crop-and-resize".format(name))


def expect_scale_check(native, small, name):
    """cx, cy, fx, fy must be adjusted exactly for the centre square crop and the resize."""
    a = (native.get("cameras") or {}).get(name)
    b = (small.get("cameras") or {}).get(name)
    if not a or not b:
        return False
    width, height = a.get("width"), a.get("height")
    size = b.get("width")
    if not (width and height and size):
        return False
    scale = size / float(min(width, height))
    x0 = (width - min(width, height)) / 2.0
    y0 = (height - min(width, height)) / 2.0
    for index, (row, col, offset) in enumerate([(0, 0, None), (1, 1, None), (0, 2, x0), (1, 2, y0)]):
        want = (a["intrinsic"][row][col] - (offset or 0.0)) * scale
        got = b["intrinsic"][row][col]
        if abs(want - got) > 1.0:
            record(FAIL, "{}: intrinsic entry [{}][{}] after size=256".format(name, row, col),
                   "expected {:.2f}, got {:.2f} -- crop then scale fx, fy, cx, cy".format(want, got))
            return False
    return True


# --------------------------------------------------------------------------- motion


def run_job(bridge, kind, body, expected_s, beat=True):
    status, reply, _ = bridge.call("POST", "/{}/start".format(kind) if kind != "home" else "/home",
                                   body)
    if status != 200 or "id" not in reply:
        return None, "start refused: {} {}".format(status, reply)
    job_id = reply["id"]
    deadline = time.time() + expected_s + 10.0
    while time.time() < deadline:
        if beat:
            bridge.call("POST", "/{}/job/{}/heartbeat".format(kind, job_id))
        status, job, _ = bridge.call("GET", "/{}/job/{}".format(kind, job_id))
        if status != 200:
            return None, "job poll {}".format(status)
        if job.get("status") not in ("queued", "running"):
            return job, ""
        time.sleep(0.2)
    return None, "the job never finished"


def check_gripper(bridge):
    print("\n-- gripper (the jaws move) --")
    for command in ("close", "open"):
        status, reply, _ = bridge.call("POST", "/gripper", {"command": command})
        check(status == 200, "POST /gripper {}".format(command),
              detail_bad="{} {}".format(status, reply))
        time.sleep(0.4)
        _, state, _ = bridge.call("GET", "/state")
        moving = (state.get("gripper") or {}).get("moving")
        check(moving is True, "/state.gripper.moving is true while the jaws move",
              detail_bad="got {!r}".format(moving), soft=True)
        time.sleep(2.0)
        _, state, _ = bridge.call("GET", "/state")
        gap = (state.get("gripper") or {}).get("gap_m", 0.0)
        if command == "open":
            check(gap > 0.06, "open leaves a gap", "{:.1f} mm".format(gap * 1000),
                  "{:.1f} mm -- measure the real full-open gap with calipers".format(gap * 1000))
        else:
            check(gap < 0.02, "close closes", "{:.1f} mm".format(gap * 1000),
                  "{:.1f} mm".format(gap * 1000))


def check_move(bridge):
    print("\n-- motion (THE ARM MOVES -- stand at the E-stop) --")
    job, error = run_job(bridge, "home", {}, 12.0)
    if not check(job is not None and job.get("status") == "complete", "POST /home completes",
                 detail_bad=error or str(job)):
        return
    _, state, _ = bridge.call("GET", "/state")
    start = list(state["tcp_pose"])

    status, reply, _ = bridge.call("POST", "/cartesian/validate",
                                   {"target_pose": [start[0] + 80.0] + start[1:], "duration": 8.0})
    check(reply.get("valid") is False, "/cartesian/validate refuses an 80 mm jog",
          str(reply.get("reason")), "it accepted {} -- the 50 mm cap is the backstop".format(reply))

    status, reply, _ = bridge.call("POST", "/cartesian/validate",
                                   {"target_pose": [start[0] + 20.0] + start[1:], "duration": 0.01})
    check(reply.get("valid") is False, "/cartesian/validate refuses a 0.01 s duration",
          str(reply.get("reason")), "it accepted {}".format(reply))

    for axis, name in ((0, "base +x (forward)"), (1, "base +y (left)"), (2, "base +z (up)")):
        for sign in (1, -1):
            _, state, _ = bridge.call("GET", "/state")
            before = list(state["tcp_pose"])
            target = list(before)
            target[axis] += 20.0 * sign
            job, error = run_job(bridge, "cartesian",
                                 {"target_pose": target, "duration": 2.0,
                                  "stop_on_contact": False, "contact": {}}, 2.0)
            if job is None or job.get("status") != "complete":
                check(False, "20 mm along {} ({})".format(name, "+" if sign > 0 else "-"),
                      detail_bad=error or str(job))
                continue
            time.sleep(0.4)
            _, state, _ = bridge.call("GET", "/state")
            after = list(state["tcp_pose"])
            moved = [after[i] - before[i] for i in range(3)]
            along = moved[axis]
            lateral = math.sqrt(sum(moved[i] ** 2 for i in range(3) if i != axis))
            check(abs(along - 20.0 * sign) < 2.0 and lateral < 2.0,
                  "20 mm along {} ({})".format(name, "+" if sign > 0 else "-"),
                  "moved {:.1f} mm, {:.1f} mm sideways".format(along, lateral),
                  "moved {:.1f} mm along and {:.1f} mm sideways -- commanded {:.0f}"
                  .format(along, lateral, 20.0 * sign))

    print("   (the failsafe: starting a job and not heartbeating it)")
    _, state, _ = bridge.call("GET", "/state")
    before = list(state["tcp_pose"])
    target = list(before)
    target[2] += 20.0
    status, reply, _ = bridge.call("POST", "/cartesian/start",
                                   {"target_pose": target, "duration": 20.0,
                                    "stop_on_contact": False, "contact": {}})
    if status == 200 and "id" in reply:
        time.sleep(4.0)
        _, job, _ = bridge.call("GET", "/cartesian/job/{}".format(reply["id"]))
        check(job.get("status") == "stopped" and "heartbeat" in (job.get("message") or ""),
              "a job with no heartbeat is stopped",
              str(job.get("message")),
              "status {!r} message {!r} -- this failsafe is what makes a 250 ms tunnel safe"
              .format(job.get("status"), job.get("message")))
        _, state, _ = bridge.call("GET", "/state")
        travelled = state["tcp_pose"][2] - before[2]
        check(abs(travelled) < 19.0, "the arm stopped short", "{:.1f} mm of 20".format(travelled),
              "it travelled {:.1f} mm -- the job ran to the end".format(travelled))
        bridge.call("POST", "/stop")
        _, state, _ = bridge.call("GET", "/state")
        check(state.get("error_code") in (0, None) and state.get("state") != 4,
              "/stop clears the parked state after a stopped job",
              detail_bad="state {} error {} -- the next job cannot start like this"
                         .format(state.get("state"), state.get("error_code")))
    else:
        check(False, "a job with no heartbeat is stopped", detail_bad="could not start one")

    run_job(bridge, "home", {}, 12.0)


def check_extrinsics(bridge, distance=60.0):
    """The check 'calibrated' does not cover: does cam2base predict what the cameras see?"""
    print("\n-- extrinsics (THE ARM MOVES) --")
    print("   Point the gripper somewhere both cameras can see it, then read the rows below:")
    print("   for each move, the projected pixel shift is what cam2base PREDICTS. Compare it")
    print("   with the gripper's real shift in the saved images. A mismatch means cam2base")
    print("   refers to the wrong optical frame, is transposed, or maps base->cam.")
    print("   The WRIST camera moves with the arm, so a pure translation should predict ~0 px")
    print("   there: what it checks is that the gripper lands in the image where you see it.\n")

    for axis, name in ((0, "+x forward"), (1, "+y left"), (2, "-z down")):
        sign = -1.0 if axis == 2 else 1.0
        _, state, _ = bridge.call("GET", "/state")
        before_pose = list(state["tcp_pose"])
        _, before_frames, _ = bridge.call("GET", "/frames?cameras=" + ",".join(CAMERAS))

        target = list(before_pose)
        target[axis] += distance * sign
        steps = int(math.ceil(distance / 49.0))
        ok = True
        for step in range(1, steps + 1):
            leg = list(before_pose)
            leg[axis] += distance * sign * step / steps
            job, error = run_job(bridge, "cartesian",
                                 {"target_pose": leg, "duration": 3.0,
                                  "stop_on_contact": False, "contact": {}}, 3.0)
            if job is None or job.get("status") != "complete":
                check(False, "move {:.0f} mm along {}".format(distance, name),
                      detail_bad=error or str(job))
                ok = False
                break
        if not ok:
            continue

        time.sleep(0.5)
        _, state, _ = bridge.call("GET", "/state")
        after_pose = list(state["tcp_pose"])
        _, after_frames, _ = bridge.call("GET", "/frames?cameras=" + ",".join(CAMERAS))
        measured = [after_pose[i] - before_pose[i] for i in range(3)]
        check(abs(measured[axis] - distance * sign) < 3.0,
              "the arm moved {:.0f} mm along {}".format(distance, name),
              "measured [{:.1f}, {:.1f}, {:.1f}] mm".format(*measured),
              "measured [{:.1f}, {:.1f}, {:.1f}] mm".format(*measured))

        for camera in CAMERAS:
            a = (before_frames.get("cameras") or {}).get(camera)
            b = (after_frames.get("cameras") or {}).get(camera)
            if not a or not b or a.get("cam2base") is None:
                record(WARN, "{}: no cam2base to check".format(camera))
                continue
            p0 = grip_site(before_pose)
            p1 = grip_site(after_pose)
            # the scene camera is fixed, the wrist camera moved with the arm: use each
            # frame's own extrinsic, which is exactly what the harness does
            uv0 = project(p0, a["cam2base"], a["intrinsic"])
            uv1 = project(p1, b["cam2base"], b["intrinsic"])
            if uv0 is None or uv1 is None:
                record(WARN, "{}: the gripper projects behind the camera".format(camera),
                       "check the sign of cam2base's z axis")
                continue
            record(PASS, "{} {}: cam2base predicts the gripper moves".format(camera, name),
                   "({:.0f}, {:.0f}) -> ({:.0f}, {:.0f}) px, i.e. {:+.0f} right {:+.0f} down"
                   .format(uv0[0], uv0[1], uv1[0], uv1[1], uv1[0] - uv0[0], uv1[1] - uv0[1]))

        for frames, when in ((before_frames, "before"), (after_frames, "after")):
            for camera in CAMERAS:
                blob = ((frames.get("cameras") or {}).get(camera) or {}).get("rgb_jpeg")
                if blob:
                    path = "extrinsic_{}_{}_{}.jpg".format(name.split()[0].strip("+-"), camera, when)
                    with open(path, "wb") as handle:
                        handle.write(base64.b64decode(blob))
        print("   images saved as extrinsic_*_{}_*.jpg\n".format(name.split()[0].strip("+-")))

        # put it back
        for step in range(1, steps + 1):
            leg = list(after_pose)
            leg[axis] -= distance * sign * step / steps
            run_job(bridge, "cartesian", {"target_pose": leg, "duration": 3.0,
                                          "stop_on_contact": False, "contact": {}}, 3.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:18765")
    parser.add_argument("--token", default="")
    parser.add_argument("--gripper", action="store_true", help="the jaws move")
    parser.add_argument("--move", action="store_true", help="THE ARM MOVES")
    parser.add_argument("--extrinsic-check", action="store_true",
                        help="THE ARM MOVES: 60 mm along each base axis, with frames")
    parser.add_argument("--expect-tcp-offset-mm", default="0,0,0,0,0,0",
                        help="the tool offset the host plans for; /health must match it")
    parser.add_argument("--host-grip-offset-mm", type=float, default=172.0,
                        help="what the host adds to tcp_pose, for the extrinsic projection")
    args = parser.parse_args()

    global EXPECTED_TCP_OFFSET_MM, GRIP_SITE_OFFSET_M
    EXPECTED_TCP_OFFSET_MM = [float(v) for v in args.expect_tcp_offset_mm.split(",")]
    GRIP_SITE_OFFSET_M = args.host_grip_offset_mm / 1000.0

    bridge = Bridge(args.url, args.token)
    print("checking {}\n".format(bridge.url))
    if check_health(bridge) is None:
        print("\nthe bridge did not answer /health; nothing else can be checked")
        return 2
    check_auth(bridge)
    check_state(bridge)
    check_frames(bridge)
    if args.gripper:
        check_gripper(bridge)
    if args.move:
        check_move(bridge)
    if args.extrinsic_check:
        check_extrinsics(bridge)

    failures = [r for r in results if r[0] == FAIL]
    warnings = [r for r in results if r[0] == WARN]
    print("\n{} pass, {} warn, {} FAIL".format(
        len(results) - len(failures) - len(warnings), len(warnings), len(failures)))
    if failures:
        print("\nfix these before the GPU host connects:")
        for _, what, detail in failures:
            print("  - {}{}".format(what, "  -- " + detail if detail else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
