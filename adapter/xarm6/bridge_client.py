"""Host-side client for the xArm bridge -- the only place this project talks HTTP."""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

CONFIRMATION = "I_UNDERSTAND_ROBOT_WILL_MOVE"
MOTION_ROUTES = ("/cartesian/start", "/trajectory/start", "/home", "/gripper")
TERMINAL = ("complete", "stopped", "failed", "aborted")



def fault_of(health: Dict[str, Any], state: Dict[str, Any], contact_expected: bool = False) -> str:
    """What stops the arm that only the operator may clear, in words -- a controller error code,
    or a collision it was not asked to stop on -- or "" for a park that is only a park: idle
    (state 5, motion_ready false), or the stop a job or the host itself left (state 4, no
    error), which clearing is ordinary operation. FAILS CLOSED: a state it could not read
    (``None``) is a fault -- a collision nobody could see is not an absent one."""
    if health is None or state is None:
        return "could not read the arm's state"
    robot = health.get("robot") or {}
    if robot.get("connected") is False:
        return "the controller is not connected"
    if health.get("motion_enabled") is False or robot.get("motion_enabled") is False:
        return "the motors are not enabled"
    code = robot.get("error_code") or state.get("error_code")
    if code:
        return "controller error code {}".format(code)
    if (state or {}).get("collision") and not contact_expected:
        return "a collision it was not asked to stop on"
    return ""

class BridgeError(RuntimeError):
    """The bridge refused something, or could not be reached."""

    def __init__(self, message: str, status: Optional[int] = None, payload: Any = None,
                 hardware: bool = False, lost_job: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.status = status
        self.payload = payload
        #: a start whose answer never came: the job may have run anyway, until the bridge's
        #: heartbeat failsafe stopped it -- for the recording to say so
        self.lost_job = lost_job
        #: the link or the arm, not the request: nothing the plan can change -- the mission
        #: pauses for the operator rather than replanning (190658: a timeout, then a job
        #: still running refused the next one, and the mission planned on)
        self.hardware = hardware


#: what the bridge answers a job started while another is running (190658, 80 s)
JOB_ACTIVE = "another job or stop is active"


class BridgeClient:
    def __init__(self, url: str = "http://127.0.0.1:18765", token: str = "",
                 timeout: float = 10.0, heartbeat_period_s: float = 0.3):
        self.url = url.rstrip("/")
        self.token = token or ""
        self.timeout = timeout
        self.heartbeat_period_s = heartbeat_period_s
        self._needs_recovery = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ plumbing

    def request(self, method: str, path: str, body: Optional[Dict[str, Any]] = None,
                timeout: Optional[float] = None) -> Any:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["X-Bridge-Token"] = self.token
        if path in MOTION_ROUTES:
            headers["X-Motion-Confirmation"] = CONFIRMATION
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(self.url + path, data, headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as reply:
                raw = reply.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            try:
                payload = json.loads(detail)
                message = payload.get("reason") or payload.get("message") or detail
            except ValueError:
                payload, message = None, detail
            if exc.code == 401:
                message = "the bridge rejected the token (401). Set XARM_BRIDGE_TOKEN."
            elif exc.code == 403:
                message = "the bridge wants the motion confirmation header (403): " + message
            raise BridgeError("{} {}: {}".format(method, path, message), exc.code, payload)
        except urllib.error.URLError as exc:
            raise BridgeError(
                "cannot reach the bridge at {} ({}). Is the ssh tunnel up? "
                "`curl -m 5 {}/health`".format(self.url, exc.reason, self.url), hardware=True)
        except (socket.timeout, TimeoutError, ConnectionError) as exc:
            # A raw socket timeout is not a URLError (190658 at 32 s: "timed out" escaped as
            # itself and the loop went on to verdicts and replans).
            raise BridgeError("{} {}: the bridge did not answer ({})".format(
                method, path, exc or type(exc).__name__), hardware=True)
        return json.loads(raw) if raw.strip() else {}

    # ------------------------------------------------------------------ read

    def health(self) -> Dict[str, Any]:
        return self.request("GET", "/health")

    def state(self) -> Dict[str, Any]:
        return self.request("GET", "/state")

    def frames(self, cameras: str = "scene,wrist", size: int = 256) -> Dict[str, Any]:
        return self.request(
            "GET", "/frames?cameras={}&size={}".format(cameras, size), timeout=max(self.timeout, 15.0)
        )

    def check_health(self, expected_tcp_offset_mm=None, tolerance_mm: float = 1.0
                     ) -> Dict[str, Any]:
        """Health, plus the refusals that must happen before anything moves."""
        health = self.health()
        offset = (health.get("robot") or {}).get("tcp_offset_mm")
        if expected_tcp_offset_mm is not None:
            if offset is None or len(offset) != len(expected_tcp_offset_mm):
                raise BridgeError(
                    "the bridge reports tcp_offset_mm as {!r}; this side expects {!r}"
                    .format(offset, list(expected_tcp_offset_mm))
                )
            drift = [abs(float(a) - float(b)) for a, b in zip(offset, expected_tcp_offset_mm)]
            if max(drift) > tolerance_mm:
                raise BridgeError(
                    "the arm's tool offset is {} but this side plans for {}. Every action is "
                    "relative to that point, so a {:.1f} mm change moves the reference of "
                    "every motion. Either restore it on the PC, or update "
                    "expected_bridge_tcp_offset_mm and grip_site_offset_m in the rig profile (robot/rigs/)."
                    .format([round(float(v), 1) for v in offset],
                            [round(float(v), 1) for v in expected_tcp_offset_mm], max(drift))
                )
        axes = (health.get("robot") or {}).get("axis_count")
        if axes is not None and int(axes) != 6:
            raise BridgeError("the bridge reports {} axes; this project is xArm6 only".format(axes))
        return health

    # ------------------------------------------------------------------ motion

    def stop(self) -> Dict[str, Any]:
        out = self.request("POST", "/stop")
        self._needs_recovery = self._contact_expected = False
        return out

    #: the last job stopped ON a contact it was asked to stop on: a collision flag after it is
    #: that contact, not a fault
    _contact_expected = False

    def fault(self) -> str:
        """Why the arm is stopped on something only the operator may clear, or "" -- see
        :func:`fault_of`. /stop clears a controller error and a collision flag, so it is never
        sent over one: a C31 collision cleared by the host is a collision nobody looked at."""
        try:
            health, state = self.health(), self.state()
        except BridgeError:
            health = state = None
        return fault_of(health, state, self._contact_expected)

    def recover_if_needed(self) -> None:
        if self._needs_recovery:
            fault = self.fault()
            if fault:
                raise BridgeError("the arm stopped on {} and is left for the operator: it was "
                                  "not cleared".format(fault))
            self.stop()

    #: the last job this client started, (kind, id): the one a "job active" refusal is about
    _last_job = None
    #: how long a job still running is waited for before one retry
    SETTLE_S = 15.0

    def start_job(self, kind: str, body: Dict[str, Any]) -> str:
        self.recover_if_needed()
        path = "/home" if kind == "home" else "/{}/start".format(kind)
        try:
            reply = self.request("POST", path, body)
        except BridgeError as exc:
            if exc.hardware and exc.status is None:
                raise BridgeError("{} -- the job may have started anyway and run until the "
                                  "bridge's heartbeat failsafe stopped it".format(exc),
                                  hardware=True, lost_job={"kind": kind, "path": path})
            if JOB_ACTIVE not in str(exc).lower():
                raise
            # The job this client lost track of (a timed-out poll) is still running: wait
            # for it to end, then try once more -- else the operator is asked.
            if not self.settle():
                raise BridgeError("{}: a job is still running on the arm and did not end "
                                  "within {:.0f} s".format(exc, self.SETTLE_S), hardware=True)
            try:
                reply = self.request("POST", path, body)
            except BridgeError as again:
                raise BridgeError(str(again), again.status, again.payload, hardware=True)
        job_id = reply.get("id")
        if not job_id:
            raise BridgeError("{} did not return a job id: {!r}".format(path, reply))
        self._last_job = (kind, job_id)
        return job_id

    def settle(self, poll_s: float = 0.3) -> bool:
        """Wait for the last job this client started to end (GET only): True once it has --
        or when there is none to wait for."""
        if self._last_job is None:
            return True
        deadline = time.monotonic() + self.SETTLE_S
        while time.monotonic() < deadline:
            try:
                status = self.job(*self._last_job).get("status")
            except BridgeError:
                status = None
            if status in TERMINAL:
                self._needs_recovery = status in ("stopped", "failed")
                return True
            time.sleep(poll_s)
        return False

    def job(self, kind: str, job_id: str) -> Dict[str, Any]:
        return self.request("GET", "/{}/job/{}".format(kind, job_id))

    def validate(self, kind: str, body: Dict[str, Any]) -> Dict[str, Any]:
        return self.request("POST", "/{}/validate".format(kind), body)

    def gripper(self, command: str) -> Dict[str, Any]:
        self.recover_if_needed()
        return self.request("POST", "/gripper", {"command": command})

    def run_job(self, kind: str, body: Dict[str, Any], expected_s: float,
                margin_s: float = 1.5, poll_s: float = 0.15) -> Dict[str, Any]:
        """Start a job, heartbeat it, and wait for it to finish: the bridge's final job
        document, plus how long the host waited (``elapsed_host_s``)."""
        started = time.monotonic()
        job_id = self.start_job(kind, body)
        deadline = started + expected_s + margin_s
        stop_beating = threading.Event()

        def beat():
            beater = BridgeClient(self.url, self.token, timeout=2.0)
            while not stop_beating.wait(self.heartbeat_period_s):
                try:
                    beater.request("POST", "/{}/job/{}/heartbeat".format(kind, job_id))
                except BridgeError:
                    return  # the job is over, or the bridge is gone; the failsafe covers us

        thread = threading.Thread(target=beat, name="heartbeat", daemon=True)
        thread.start()
        try:
            while True:
                try:
                    document = self.job(kind, job_id)
                except BridgeError as exc:
                    # One poll the link dropped is not the job's end: poll again to the
                    # deadline, and only then give up on it (hardware, not the plan).
                    if not exc.hardware or time.monotonic() > deadline:
                        raise
                    time.sleep(poll_s)
                    continue
                status = document.get("status")
                if status in TERMINAL:
                    document["elapsed_host_s"] = round(time.monotonic() - started, 3)
                    self._needs_recovery = status in ("stopped", "failed")
                    self._contact_expected = bool(document.get("contact"))
                    return document
                if time.monotonic() > deadline:
                    self.stop()
                    document = self.job(kind, job_id)
                    document["status"] = "aborted"
                    document["message"] = (
                        "the host stopped it: {:.1f} s over the {:.1f} s this job should have "
                        "taken".format(time.monotonic() - deadline, expected_s)
                    )
                    document["elapsed_host_s"] = round(time.monotonic() - started, 3)
                    self._needs_recovery = True
                    return document
                time.sleep(poll_s)
        finally:
            stop_beating.set()
