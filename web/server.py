"""The operator page: type a semantic command, see what it resolves to, then run it.

    python web/server.py --port 3008                          # the real xArm6
    python web/server.py --port 3008 --bridge http://127.0.0.1:18766 --token dev
    python web/server.py --port 3008 --backend libero --task 0 # the simulated Panda
"""

from __future__ import annotations

import argparse
import base64
import math
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import overlay as overlay_draw  # noqa: E402
import yaml  # noqa: E402
from src.controller import runner  # noqa: E402
from src.controller.clearance import from_wrist, grasp_state, held_to_the_table  # noqa: E402
from src.controller.convert import (ConversionError, TOTAL_ROTATION_LIMIT_RAD,  # noqa: E402
                                    TOTAL_TRANSLATION_LIMIT_M, load_directions, to_deltas)
from src.executor.context import (ClearanceTracker, TablePlane, aim_depth,  # noqa: E402
                                  describe_aim_point,
                                  describe_capabilities, describe_robot,
                                  paint_aim_marker)
from src.executor.loop import ExecutorLoop, Observation  # noqa: E402
from src.executor.proposal import Proposer  # noqa: E402
from src.executor.supervision import MotionSupervisor  # noqa: E402
from src.planner.evidence import EvidenceLog  # noqa: E402
from src.planner.mission import MISSION_BUDGET_S, Mission  # noqa: E402
from src.recording import start_recording  # noqa: E402
from src.planner.monitor import SceneMonitor  # noqa: E402
from src.planner.planner import Planner  # noqa: E402
from src.planner.verify import Verifier  # noqa: E402
from src.schema.actions import ActionError, parse_command  # noqa: E402
from vlms import client_for  # noqa: E402

PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "src", "prompts")
PROMPTS_PATH = os.path.join(PROMPTS_DIR, "executor.yaml")
PLANNER_PROMPTS_PATH = os.path.join(PROMPTS_DIR, "planner.yaml")

try:
    import cv2
except ImportError:
    cv2 = None

HERE = os.path.dirname(os.path.abspath(__file__))

EXAMPLES = [
    {"name": "reach out and down",
     "json": {"speed_mm_s": 10, "actions": [
         {"type": "move", "direction": "forward", "distance_mm": 80},
         {"type": "move", "direction": "down", "distance_mm": 60,
          "note": "check the live clearance and leave yourself room"}]}},
    {"name": "pick straight down",
     "json": {"actions": [
         {"type": "gripper", "state": "open"},
         {"type": "move", "direction": "down", "distance_mm": 40, "speed_mm_s": 8},
         {"type": "gripper", "state": "close"},
         {"type": "wait", "seconds": 1},
         {"type": "move", "direction": "up", "distance_mm": 40}]}},
    {"name": "turn the wrist",
     "json": {"actions": [{"type": "rotate", "direction": "yaw_left", "angle_deg": 15}]}},
    {"name": "an axis with no word for it",
     "json": {"actions": [{"type": "move", "axis": [1, 1, 0], "distance_mm": 40}]}},
    {"name": "a slow descent",
     "json": {"speed_mm_s": 5, "actions": [
         {"type": "move", "direction": "down", "distance_mm": 25,
          "note": "slow enough to watch; speed is clamped to what the backend allows"}]}},
]


class BackendRequest(BaseModel):
    name: str


class TaskRequest(BaseModel):
    suite: str
    task: int


class SubgoalRequest(BaseModel):
    subgoal: str
    max_cycles: int = 6
    # What "done" looks like, in the model's own terms.
    criterion: str = ""


class MissionRequest(BaseModel):
    """A whole task, planned and driven by the three models."""

    task: str
    budget: int = 10
    monitor: bool = True


class PlanRequest(BaseModel):
    command: Any
    # The pose the previewed plan was built from.
    expect_position_mm: Optional[List[float]] = None


DRIFT_MM = 3.0


class ExecutorSession:
    """One subgoal being driven, on its own thread, with everything it decided kept."""

    def __init__(self):
        self.lock = threading.Lock()
        self.subgoal = ""
        self.events: List[Dict[str, Any]] = []
        self.cycles: List[Dict[str, Any]] = []
        self.running = False
        self.error = ""
        self.loop: Optional[ExecutorLoop] = None

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            events = list(self.events)
            return {"subgoal": self.subgoal, "running": self.running, "error": self.error,
                    "events": events, "cycles": list(self.cycles),
                    # The same three parts the mission tab averages, over this subgoal only.
                    "timings": timings_from(events, EXECUTOR_PARTS)}

    def append(self, event: Dict[str, Any]) -> None:
        with self.lock:
            event["t"] = round(time.time(), 3)
            self.events.append(event)


#: How many evidence rows the page is shown.
LOG_TAIL = 40


#: Every part of a run that costs time, and the events its seconds are read from.
TIMED_PARTS = (("plan", ("planned", "replanned")),
               ("verify", ("verdict",)),
               ("monitor", ("alert",)),
               ("note", ("note",)),
               ("propose", ("proposed",)),
               ("supervise", ("supervision",)),
               ("motion", ("step_end",)))

#: The three of them the executor owns, so one subgoal on the console can be timed the same
#: way a whole mission is.
EXECUTOR_PARTS = tuple(part for part in TIMED_PARTS
                       if part[0] in ("propose", "supervise", "motion"))


def _stats(seconds: List[float]) -> Dict[str, Any]:
    if not seconds:
        return {"count": 0, "mean_s": 0.0, "total_s": 0.0, "last_s": 0.0}
    total = float(sum(seconds))
    return {"count": len(seconds), "mean_s": round(total / len(seconds), 2),
            "total_s": round(total, 2), "last_s": round(float(seconds[-1]), 2)}


def _number(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) \
        else None


def timings_from(events: List[Dict[str, Any]],
                 parts=TIMED_PARTS) -> Dict[str, Dict[str, Any]]:
    """How long each part of a run has been taking, averaged over the run so far."""
    buckets: Dict[str, List[float]] = {name: [] for name, _ in parts}
    for event in events:
        for name, kinds in parts:
            if event.get("event") in kinds:
                seconds = _number(event.get("elapsed_s"))
                if seconds is not None:
                    buckets[name].append(seconds)
    return {name: _stats(buckets[name]) for name, _ in parts}


def wall_spans(events: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Wall time per motion cycle and per subgoal, from the ``t`` the session stamps."""
    cycles: List[float] = []
    subgoals: List[float] = []
    cycle_open: Optional[float] = None
    subgoal_open: Optional[float] = None
    for event in events:
        name = event.get("event")
        stamp = _number(event.get("t"))
        if name in ("proposed", "subgoal_start", "verdict"):
            if cycle_open is not None and stamp is not None:
                cycles.append(max(0.0, stamp - cycle_open))
            cycle_open = stamp if name == "proposed" else None
        if name == "subgoal_start":
            subgoal_open = stamp
        elif name == "verdict" and subgoal_open is not None:
            if stamp is not None:
                subgoals.append(max(0.0, stamp - subgoal_open))
            subgoal_open = None
    return {"cycle": _stats(cycles), "subgoal": _stats(subgoals)}


def _row(step: Dict[str, Any], version: int, number: int) -> Dict[str, Any]:
    return {"name": step.get("name", ""), "subgoal": step.get("subgoal", ""),
            "criterion": step.get("criterion", ""), "target": step.get("target", ""),
            "max_cycles": step.get("max_cycles"), "gripper_only": bool(step.get("gripper_only")),
            "version": version, "n": number, "state": "pending", "away": False,
            "attempt": 0, "verdict": {}, "alerts": []}


def mission_snapshot(events: List[Dict[str, Any]], live: Dict[str, Any],
                     started_at: Optional[float] = None,
                     now: Optional[float] = None) -> Dict[str, Any]:
    """Everything the page draws, derived from the event stream the mission emitted."""
    rows: List[Dict[str, Any]] = []
    current: List[int] = []                 # indices into rows: the plan as it stands now
    index = -1                              # where in `current` the running subgoal sits
    plans: List[Dict[str, Any]] = []
    notes: List[Dict[str, Any]] = []
    task_verdict: Dict[str, Any] = {}
    finished: Dict[str, Any] = {}
    executor: Dict[str, Any] = {"subgoal": "", "events": [], "cycles": []}

    for event in events:
        name = event.get("event")
        if name in ("planned", "replanned"):
            plan = event.get("plan") or {}
            steps = list(plan.get("subgoals") or [])
            version = int(event.get("version") or (len(plans) + 1))
            plans.append({"version": version, "trigger": event.get("trigger", ""),
                          "clamped": list(event.get("clamped") or []),
                          "rationale": plan.get("rationale", ""),
                          "subgoals": len(steps)})
            keep = 0 if name == "planned" else (index if index >= 0 else len(current))
            for slot in current[keep:]:
                rows[slot]["away"] = True
                if rows[slot]["state"] in ("pending", "running"):
                    rows[slot]["state"] = "replanned_away"
            current = current[:keep]
            # The event carries the whole plan (the kept prefix plus the new tail), so the
            # rows to add start where the kept prefix ends.
            for step in steps[keep:]:
                rows.append(_row(step, version, len(current) + 1))
                current.append(len(rows) - 1)
            index = -1
        elif name == "subgoal_start":
            slot = None
            for position in range(max(0, index), len(current)):
                row = rows[current[position]]
                if row["name"] == event.get("name") and row["state"] != "done":
                    slot = position
                    break
            if slot is None:                # a start with no plan behind it: show it anyway
                rows.append(_row({"name": event.get("name", "")}, plans[-1]["version"]
                                 if plans else 1, len(current) + 1))
                current.append(len(rows) - 1)
                slot = len(current) - 1
            row = rows[current[slot]]
            row.update({"state": "running", "attempt": int(event.get("attempt") or 1),
                        "subgoal": event.get("subgoal") or row["subgoal"],
                        "criterion": event.get("criterion") or row["criterion"],
                        "max_cycles": event.get("max_cycles", row["max_cycles"])})
            index = slot
            executor = {"subgoal": row["subgoal"] or row["name"], "events": [], "cycles": []}
        elif name == "alert":
            if index >= 0:
                rows[current[index]]["alerts"].append(
                    {"level": event.get("level", ""), "finding": event.get("finding", ""),
                     "because": event.get("because", ""), "attempt":
                     rows[current[index]]["attempt"]})
        elif name == "verdict":
            said = {"status": event.get("status", ""), "because": event.get("because", ""),
                    "evidence": event.get("evidence", ""), "replan": bool(event.get("replan"))}
            slot = None
            for position in range(len(current)):
                if rows[current[position]]["name"] == event.get("name") \
                        and rows[current[position]]["state"] == "running":
                    slot = position
                    break
            if slot is None:
                # The task-level verify: it judges the whole task, not a subgoal, and
                # mission.py names it "task".
                task_verdict = {**said, "name": event.get("name", "task")}
                continue
            row = rows[current[slot]]
            row["verdict"] = said
            row["state"] = "done" if said["status"] in ("done", "task_done") else "not_done"
            if said["status"] == "task_done":
                task_verdict = {**said, "name": event.get("name", "")}
        elif name == "note":
            notes.append({"version": event.get("version"), "note": event.get("note"),
                          "error": event.get("error", ""),
                          "fallback": not event.get("note")})
        elif name == "finished":
            finished = {"status": event.get("status", ""), "error": event.get("error", ""),
                        "total_cycles": event.get("total_cycles", 0),
                        "elapsed_s": event.get("elapsed_s", 0.0)}
            for slot in current:
                if rows[slot]["state"] == "running":
                    rows[slot]["state"] = "not_done"
                    rows[slot]["verdict"] = rows[slot]["verdict"] or {
                        "status": "", "because": "the mission ended ({}) before this subgoal "
                                                 "was judged".format(finished["status"]),
                        "evidence": "", "replan": False}
        elif "subgoal" in event:
            # A forwarded executor event: it belongs to the subgoal now running.
            executor["events"].append(event)

    # How long each part has been taking, and how long the mission has been going.
    stamps = [t for t in (_number(e.get("t")) for e in events) if t is not None]
    if finished:
        wall_s = float(finished.get("elapsed_s") or 0.0)
    elif stamps:
        first = stamps[0] if started_at is None else started_at
        wall_s = max(0.0, (stamps[-1] if now is None else now) - first)
    else:
        wall_s = 0.0

    out = dict(live)
    out.update({"plans": plans, "subgoals": rows, "task_verdict": task_verdict,
                "executor": executor, "notes": list(reversed(notes)), "finished": finished,
                "timings": {**timings_from(events), **wall_spans(events)},
                "wall_s": round(wall_s, 1)})
    return out


class MissionSession:
    """One mission being driven, on its own thread, with everything it decided kept."""

    def __init__(self):
        self.lock = threading.Lock()
        self.task = ""
        self.budget = 10
        self.monitor = True
        self.events: List[Dict[str, Any]] = []
        self.running = False
        self.error = ""
        self.status = "idle"
        self.started_at: Optional[float] = None     # for the wall time, before any event lands
        self.mission: Optional[Mission] = None
        self.log: Optional[EvidenceLog] = None
        self.result: Optional[Dict[str, Any]] = None

    def append(self, event: Dict[str, Any]) -> None:
        with self.lock:
            event["t"] = round(time.time(), 3)
            self.events.append(event)

    def snapshot(self, models: Dict[str, Any], busy: bool) -> Dict[str, Any]:
        with self.lock:
            events = list(self.events)
            rows = [] if self.log is None else list(self.log.rows)[-LOG_TAIL:]
            live = {"task": self.task, "budget": self.budget, "running": self.running,
                    "status": self.status, "error": self.error, "busy": busy,
                    "monitor": self.monitor, "models": models,
                    # NOT the result: it carries every cycle, which carries every prompt the
                    # models were sent -- hundreds of KB on a poll the page runs every 700 ms,
                    # and nothing on the page reads it. The scalars are in `finished`.
                    "log": (EvidenceLog(rows).render().split("\n") if rows else [])}
            started_at, running = self.started_at, self.running
        return mission_snapshot(events, live, started_at=started_at,
                                now=time.time() if running else None)

    def cancel(self) -> None:
        mission = self.mission
        if mission is not None:
            mission.cancel()


class Backends:
    """Builds and keeps the adapters, so switching does not restart the server."""

    def __init__(self, options: Dict[str, Any]):
        self.options = options
        self.built: Dict[str, Any] = {}
        self.current: Optional[str] = None
        self.errors: Dict[str, str] = {}

    @property
    def names(self) -> List[str]:
        return ["xarm6", "libero"]

    def adapter(self):
        if self.current is None:
            raise RuntimeError("no backend selected")
        return self.built[self.current]

    def select(self, name: str):
        if name not in self.names:
            raise ValueError("unknown backend {!r}; try {}".format(name, ", ".join(self.names)))
        if name not in self.built:
            if name == "xarm6":
                from adapter.xarm6 import XArm6Adapter
                built = XArm6Adapter(url=self.options.get("bridge"),
                                     token=self.options.get("token", ""))
            else:
                from adapter.libero import LiberoAdapter
                built = LiberoAdapter(suite=self.options.get("suite", "libero_10"),
                                      task_id=int(self.options.get("task", 0)),
                                      seed=int(self.options.get("seed", 0)))
            built.connect()
            self.built[name] = built
        self.errors.pop(name, None)
        self.current = name
        return self.built[name]


class Run:
    def __init__(self):
        self.lock = threading.Lock()
        self.id: Optional[str] = None
        self.events: List[Dict[str, Any]] = []
        self.report: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.running = False

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {"id": self.id, "running": self.running, "events": list(self.events),
                    "report": self.report, "error": self.error}

    def append(self, event: Dict[str, Any]) -> None:
        with self.lock:
            event["t"] = round(time.time(), 3)
            self.events.append(event)


def action_schema(directions: Dict[str, Any]) -> Dict[str, Any]:
    """Generated from the same tables the parser reads, so it cannot drift from them."""
    return {
        "shape": '{"actions": [{"type": <type>, ...}, ...]}  -- a bare list or a single '
                 'action object also works; "frame": "base" is optional and is the only frame',
        "types": [
            {"type": "move", "required": ["direction | axis", "distance_mm"], "optional": [],
             "note": "every motion, vertical or horizontal. distance_mm is a magnitude, so "
                     "say the opposite direction word rather than a negative one. Several "
                     "moves in one command are fine: the offsets you are given are per axis"},
            {"type": "rotate", "required": ["direction | axis", "angle_deg"], "optional": [],
             "note": "pivots about the control point, so the tool turns in place"},
            {"type": "gripper", "required": ["state"], "optional": ["width_mm"],
             "note": 'state is "open" or "close"'},
            {"type": "home", "required": [], "optional": [],
             "note": "nothing may move after a home in the same command"},
            {"type": "wait", "required": ["seconds"], "optional": [], "note": "hold still"},
        ],
        "options": [
            {"key": "speed_mm_s", "note": "clamped to what the backend says it can do"},
            {"key": "rot_speed_deg_s", "note": ""},
            {"key": "stop_on_contact", "note": "on a move: drive until something stops it "
                                               "rather than stopping at distance_mm -- this "
                                               "is how a drawer, a door or a lever is pushed. "
                                               "Only if the backend supports it"},
            {"key": "note", "note": "free text, carried into the plan"},
        ],
        "values": {"type": ["move", "rotate", "gripper", "home", "wait"],
                   "frame": [directions["frame"]], "state": ["open", "close"]},
        "command_limits": {
            "max_total_translation_mm": TOTAL_TRANSLATION_LIMIT_M * 1000,
            "max_total_rotation_deg": round(math.degrees(TOTAL_ROTATION_LIMIT_RAD)),
        },
    }


EXECUTOR_CAMERAS: List[str] = []

# How big a frame the browser is served, on its longest edge.
PAGE_SIDE = int(os.environ.get("STORM_PAGE_IMAGE_SIDE", "512"))


def build_app(backends: Backends, directions: Dict[str, Any]) -> FastAPI:
    app = FastAPI(title="storm_0918")
    run = Run()
    executor = ExecutorSession()
    mission = MissionSession()
    prompts = yaml.safe_load(open(PROMPTS_PATH))
    planner_prompts = yaml.safe_load(open(PLANNER_PROMPTS_PATH))
    # One client per role -- the executor's supervision call has to land while the motion it
    # watches is still running, and a planner call carrying four images queued in front of it
    # would make that impossible.
    clients = {role: client_for(role)
               for role in ("executor", "planner", "monitor")}
    vlm = clients["executor"]
    models: Dict[str, Any] = {}
    # The recording of the run in progress (src/recording.py): every event, call and frame.
    recording: Dict[str, Any] = {"now": None}

    def preflight() -> str:
        """Can the arm move -- once the adapter has cleared an idle arm, as a motion would? Asked
        before anything is planned: run 5 spent a cycle, a verdict, a note and a replan on an arm
        that would not move. "" when it can, or when the backend cannot say."""
        ready = getattr(adapter(), "readiness", None)
        if not callable(ready):
            return ""
        ok, why = ready()
        return "" if ok else "the arm will not accept motion, so nothing was started: " + why

    def recorded(append, kind: str):
        """An event sink that also files every event into a fresh recording of this run."""
        recording["now"] = start_recording(kind)
        now = recording["now"]
        if now is None:
            return append

        def both(event):
            append(event)
            try:
                now.on_event(event)
            except Exception as exc:          # a lost record, never a lost mission
                print("recording: event failed: {}".format(exc), file=sys.stderr)
        return both

    def stop_recording(log=None) -> None:
        now, recording["now"] = recording["now"], None
        if now is not None:
            from vlms import qwen
            qwen.RECORDER = None
            try:
                if log is not None:
                    now.write_evidence(list(log.rows))
                now.finish()
            except Exception as exc:
                print("recording: closing failed: {}".format(exc), file=sys.stderr)

    def check_models(roles=("executor", "planner", "monitor")) -> Dict[str, Any]:
        """Ask each role's server whether it is there."""
        for role in roles:
            client = clients[role]
            reachable, detail = client.health()
            models[role] = {"reachable": reachable, "detail": detail, "url": client.url,
                            "model": client.model, "checked": round(time.time(), 1)}
        return models

    check_models()
    tracker = ClearanceTracker()     # clearance by dead reckoning once the depth is blind

    def adapter():
        return backends.adapter()

    def caps():
        return adapter().capabilities()

    def has(name: str) -> bool:
        return callable(getattr(adapter(), name, None))

    @app.get("/")
    def index():
        return FileResponse(os.path.join(HERE, "index.html"))

    @app.post("/api/backend")
    def api_backend(request: "BackendRequest"):
        if run.running or executor.running or mission.running:
            return JSONResponse({"ok": False, "error": "a run is going; stop it first"},
                                status_code=200)
        try:
            backends.select(request.name)
        except Exception as exc:
            backends.errors[request.name] = str(exc)
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=200)
        tracker.reset()
        return {"ok": True, "current": backends.current}

    @app.get("/api/tasks")
    def api_tasks():
        """Every task the current backend can be switched to; empty on a backend with none."""
        if backends.current is None or not has("available_tasks"):
            return {"suites": {}, "current": None}
        try:
            suites = adapter().available_tasks()
        except Exception as exc:
            return JSONResponse({"suites": {}, "current": None, "error": str(exc)},
                                status_code=200)
        a = adapter()
        return {"suites": suites,
                "current": {"suite": a.suite_name, "task_id": a.task_id,
                            "language": a.task_language}}

    @app.post("/api/task")
    def api_task(request: "TaskRequest"):
        if run.running or executor.running or mission.running:
            return JSONResponse({"ok": False, "error": "something is running; stop it first"},
                                status_code=200)
        if backends.current is None or not has("set_task"):
            return JSONResponse({"ok": False, "error": "this backend has no tasks to switch"},
                                status_code=200)
        try:
            adapter().set_task(request.suite, request.task)
        except Exception as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=200)
        # the next build of this backend, and the page's own record, follow the switch
        backends.options.update({"suite": request.suite, "task": int(request.task)})
        # both were measured over the old scene: the clearance odometry, and the table
        # height the aim point projects to -- TablePlane only ever goes lower, so a switch
        # from a low table would otherwise keep projecting to a surface that is not there
        tracker.reset()
        table.z_mm = None
        a = adapter()
        return {"ok": True, "suite": a.suite_name, "task_id": a.task_id,
                "language": a.task_language}

    @app.get("/api/config")
    def api_config():
        if backends.current is None:
            return JSONResponse({"backend": None, "error": "no backend connected"},
                                status_code=200)
        c = caps()
        return {
            "backend": c.name,
            "backends": {"current": backends.current, "available": backends.names,
                         "connected": sorted(backends.built), "errors": dict(backends.errors)},
            "capabilities": c.as_dict(),
            "frames": {directions["frame"]: {
                "directions": sorted(directions.get("directions") or {}),
                "rotations": sorted(directions.get("rotations") or {}),
                "aliases": dict(directions.get("aliases") or {})}},
            "schema": action_schema(directions),
            "examples": EXAMPLES,
            "has_cameras": has("frames"),
            "cameras": adapter().camera_names() if has("camera_names") else [],
        }

    @app.get("/api/status")
    def api_status():
        if backends.current is None:
            return JSONResponse({"connected": False, "error": "no backend connected"},
                                status_code=200)
        try:
            pose = adapter().tcp_pose()
        except Exception as exc:
            return JSONResponse({"connected": False, "error": str(exc)}, status_code=200)
        out = {"connected": True, "backend": caps().name,
               "tool_mm": [round(v * 1000.0, 1) for v in pose.position_m],
               "readiness_issues": list(caps().notes),
               # so the page keeps polling the mission, and keeps the backend and task
               # pickers disabled, from whichever tab is on screen
               "mission": {"running": mission.running, "status": mission.status}}
        if has("extra_status"):
            try:
                out.update(adapter().extra_status())
            except Exception:
                pass
        if has("gripper_reading"):
            try:
                reading = adapter().gripper_reading()
                # the neutral reading as well as the verdict, so the page's indicator draws
                # the same way on either backend instead of sniffing backend-specific fields
                out["gripper_reading"] = reading.as_dict()
                out["grasp"] = grasp_state(reading).to_dict()
            except Exception:
                pass
        return out

    def _for_the_page(bgr):
        """Shrink a frame to :data:`PAGE_SIDE` on its longest edge, if it is bigger."""
        side = max(bgr.shape[:2])
        if not PAGE_SIDE or side <= PAGE_SIDE:
            return bgr
        scale = PAGE_SIDE / float(side)
        return cv2.resize(bgr, (max(1, int(bgr.shape[1] * scale)),
                                max(1, int(bgr.shape[0] * scale))), interpolation=cv2.INTER_AREA)

    @app.get("/api/frames")
    def api_frames(overlay: bool = True, cameras: str = ""):
        if backends.current is None or not has("frames"):
            return {"cameras": {}, "clearance": None}
        wanted = [c for c in cameras.split(",") if c] or None
        try:
            payload = adapter().frames(wanted)
            pose = adapter().tcp_pose()
            flange = adapter().flange_pose() if has("flange_pose") else pose
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=200)

        clearance = None
        wrist = (payload.get("cameras") or {}).get("wrist")
        if wrist is not None:
            try:
                clearance = held_to_the_table(tracker.update(
                    from_wrist(wrist, flange, caps().control_point_offset_m).to_dict(), flange),
                    pose, caps().table_z_m)
            except Exception:
                clearance = None

        out = {}
        for name, camera in (payload.get("cameras") or {}).items():
            blob = camera.get("rgb_jpeg")
            if cv2 is not None and blob:
                try:
                    raw = np.frombuffer(base64.b64decode(blob), np.uint8)
                    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                    if bgr is not None:
                        if overlay and camera.get("cam2base") is not None:
                            bgr = overlay_draw.draw(
                                bgr[:, :, ::-1], camera["intrinsic"], camera["cam2base"],
                                pose=flange,
                                control_offset_m=caps().control_point_offset_m)[:, :, ::-1]
                        # for the page only: a no-op when the render is already PAGE_SIDE
                        bgr = _for_the_page(bgr)
                        ok, buffer = cv2.imencode(".jpg", bgr,
                                                  [cv2.IMWRITE_JPEG_QUALITY, 88])
                        if ok:
                            blob = base64.b64encode(buffer.tobytes()).decode("ascii")
                except Exception:
                    pass
            out[name] = {"rgb_jpeg": blob,
                         "has_extrinsic": camera.get("cam2base") is not None,
                         "timestamp": camera.get("timestamp")}
        return {"cameras": out, "clearance": clearance}

    def _plan(command):
        deltas = to_deltas(parse_command(command), caps(), directions)
        return deltas, runner.plan(adapter(), deltas)

    @app.post("/api/plan")
    def api_plan(request: PlanRequest):
        try:
            deltas, plan = _plan(request.command)
        except ActionError as exc:
            return JSONResponse({"ok": False, "stage": "parse", "error": str(exc)}, status_code=200)
        except ConversionError as exc:
            return JSONResponse({"ok": False, "stage": "convert", "error": str(exc)}, status_code=200)
        except Exception as exc:
            return JSONResponse({"ok": False, "stage": "robot", "error": str(exc)}, status_code=200)
        return {"ok": True, "deltas": [d.as_dict() for d in deltas], "plan": plan.as_dict()}

    @app.post("/api/execute")
    def api_execute(request: PlanRequest):
        # The executor and the mission drive the same adapter on their own threads: a manual
        # command admitted beside one of them would be a second hand on the robot.
        if run.running or executor.running or mission.running:
            return JSONResponse({"ok": False, "error": "something is already running"},
                                status_code=200)
        try:
            if request.expect_position_mm is not None:
                now = adapter().tcp_pose().position_m * 1000.0
                drift = float(np.max(np.abs(now - np.asarray(request.expect_position_mm))))
                if drift > DRIFT_MM:
                    return JSONResponse(
                        {"ok": False, "error":
                         "the robot has moved {:.1f} mm since that plan was previewed; it has "
                         "been refreshed, look again before running".format(drift)},
                        status_code=200)
            deltas, plan = _plan(request.command)
        except Exception as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=200)

        return _start_run(deltas, plan)

    def _start_run(deltas, plan):
        run.id = "run-{}".format(int(time.time()))
        run.events, run.report, run.error, run.running = [], None, None, True

        def work():
            try:
                report = runner.run(adapter(), deltas, on_event=run.append)
                with run.lock:
                    run.report = report.as_dict()
            except Exception as exc:
                with run.lock:
                    run.error = str(exc)
            finally:
                with run.lock:
                    run.running = False

        threading.Thread(target=work, name="run", daemon=True).start()
        return {"ok": True, "id": run.id, "steps": len(plan.steps)}

    @app.post("/api/reset")
    def api_reset():
        """The page's reset-scene button: one ``home`` action through the ordinary runner."""
        if run.running or executor.running or mission.running:
            return JSONResponse({"ok": False, "error": "something is running; stop it first"},
                                status_code=200)
        if backends.current is None:
            return JSONResponse({"ok": False, "error": "no backend connected"}, status_code=200)
        not_ready = preflight()
        if not_ready:
            return JSONResponse({"ok": False, "error": not_ready}, status_code=200)
        try:
            deltas, plan = _plan({"actions": [{"type": "home"}]})
        except Exception as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=200)
        # the clearance odometry was measured over the old scene
        tracker.reset()
        return _start_run(deltas, plan)

    # ------------------------------------------------------------------ executor

    table = TablePlane()

    def _surface(clearance) -> Optional[float]:
        """Where the table is, for the aim marker: the commissioned height if the robot has
        one, else the lowest surface the wrist has seen this session."""
        commissioned = caps().table_z_m
        seen = table.update(clearance)
        return seen if commissioned is None else commissioned * 1000.0


    def build_observation() -> Observation:
        """What the model looks at, and what the robot says about itself."""
        pose = adapter().tcp_pose()
        images, names, capture = [], [], time.time()
        clearance = grasp = None
        cameras_payload: Dict[str, Any] = {}
        if has("frames"):
            try:
                payload = adapter().frames()
                cameras_payload = payload.get("cameras") or {}
                if EXECUTOR_CAMERAS:
                    cameras_payload = {n: c for n, c in cameras_payload.items()
                                       if n in EXECUTOR_CAMERAS}
                flange = adapter().flange_pose() if has("flange_pose") else pose
                wrist = cameras_payload.get("wrist")
                if wrist is not None:
                    clearance = held_to_the_table(tracker.update(
                        from_wrist(wrist, flange, caps().control_point_offset_m).to_dict(),
                        flange), pose, caps().table_z_m)
                surface = _surface(clearance)
                depth = aim_depth(pose, clearance, surface)
                for name, camera in cameras_payload.items():
                    blob = camera.get("rgb_jpeg")
                    if not blob or cv2 is None:
                        continue
                    bgr = cv2.imdecode(np.frombuffer(base64.b64decode(blob), np.uint8),
                                       cv2.IMREAD_COLOR)
                    if bgr is None:
                        continue
                    # the aim spot drawn where the text says it is -- the one thing on the
                    # model's copy of the picture that is not the scene
                    rgb = bgr[:, :, ::-1]
                    try:
                        if recording["now"] is not None:
                            recording["now"].on_camera(name, camera, pose, flange)
                    except Exception as exc:  # never drops a picture the models are shown
                        print("recording: frame failed: {}".format(exc), file=sys.stderr)
                    camera["image"] = rgb          # the picture locate() measures, unmarked
                    images.append(paint_aim_marker(rgb, camera, pose, depth))
                    names.append(name)
                    capture = min(capture, float(camera.get("timestamp") or capture))
            except Exception:
                pass
        if has("gripper_reading"):
            try:
                grasp = grasp_state(adapter().gripper_reading()).to_dict()
            except Exception:
                grasp = None
        surface = _surface(clearance)
        seen = (clearance or {}).get("mm")
        if seen is None:
            seen = (clearance or {}).get("tracked_mm")

        return Observation(images=images, names=names, capture_time=capture,
                           robot_text=describe_robot(pose, clearance, grasp), pose=pose,
                           grasp=grasp or {},
                           legend=describe_aim_point(cameras_payload, pose, clearance,
                                                     surface_z_mm=surface),
                           clearance_mm=None if seen is None else float(seen),
                           clearance_seen=(clearance or {}).get("mm") is not None,
                           cameras=cameras_payload)

    @app.get("/api/executor")
    def api_executor():
        snapshot = executor.snapshot()
        snapshot["model"] = models["executor"]
        return snapshot

    @app.post("/api/executor/start")
    def api_executor_start(request: SubgoalRequest):
        if executor.running or run.running or mission.running:
            return JSONResponse({"ok": False, "error": "something is already running"
                                 + (" (a mission)" if mission.running else "")},
                                status_code=200)
        if backends.current is None:
            return JSONResponse({"ok": False, "error": "no backend connected"}, status_code=200)
        not_ready = preflight()
        if not_ready:
            return JSONResponse({"ok": False, "error": not_ready}, status_code=200)
        subgoal = (request.subgoal or "").strip()
        if not subgoal:
            return JSONResponse({"ok": False, "error": "say what the subgoal is"},
                                status_code=200)

        loop = ExecutorLoop(
            adapter=adapter(),
            proposer=Proposer(vlm, prompts, action_schema(directions), directions),
            supervisor=MotionSupervisor(vlm, prompts),
            observe=build_observation,
            capability_text=describe_capabilities(caps()),
            max_cycles=max(1, min(20, int(request.max_cycles))))
        executor.subgoal, executor.events, executor.cycles = subgoal, [], []
        executor.error, executor.running, executor.loop = "", True, loop

        def work():
            try:
                cycles = loop.run(subgoal, on_event=recorded(executor.append, "subgoal"),
                                  criterion=(request.criterion or "").strip())
                with executor.lock:
                    executor.cycles = [c.as_dict() for c in cycles]
            except Exception as exc:
                with executor.lock:
                    executor.error = "{}: {}".format(type(exc).__name__, exc)
            finally:
                stop_recording()
                with executor.lock:
                    executor.running = False

        threading.Thread(target=work, name="executor", daemon=True).start()
        return {"ok": True, "subgoal": subgoal}

    # ------------------------------------------------------------------ mission

    @app.get("/api/mission")
    def api_mission():
        return mission.snapshot(models, busy=bool(run.running or executor.running))

    @app.post("/api/mission/start")
    def api_mission_start(request: MissionRequest):
        """The whole task: the planner writes the subgoals, the executor drives each one, the
        monitor watches it and the verifier judges it."""
        if mission.running or executor.running or run.running:
            return JSONResponse({"ok": False, "error": "something is already running"},
                                status_code=200)
        if backends.current is None:
            return JSONResponse({"ok": False, "error": "no backend connected"}, status_code=200)
        not_ready = preflight()
        if not_ready:
            return JSONResponse({"ok": False, "error": not_ready}, status_code=200)
        task = (request.task or "").strip()
        if not task:
            return JSONResponse({"ok": False, "error": "say what the task is"}, status_code=200)
        budget = max(1, min(20, int(request.budget)))
        roles = ["planner", "executor"]
        check_models(roles)
        down = [r for r in roles if not models[r]["reachable"]]
        if down:
            return JSONResponse({"ok": False, "error": "; ".join(models[r]["detail"]
                                                                 for r in down)},
                                status_code=200)

        def loop_factory(subgoal):
            """One fresh :class:`ExecutorLoop` per subgoal, built as the console builds it."""
            return ExecutorLoop(
                adapter=adapter(),
                proposer=Proposer(vlm, prompts, action_schema(directions), directions),
                supervisor=MotionSupervisor(vlm, prompts),
                observe=build_observation,
                capability_text=describe_capabilities(caps()),
                gripper_only=bool(subgoal.gripper_only),
                max_cycles=max(1, min(budget, int(subgoal.max_cycles))))

        log = EvidenceLog()
        driver = Mission(
            planner=Planner(clients["planner"], planner_prompts, budget=budget),
            verifier=Verifier(clients["planner"], planner_prompts),
            monitor=None,
            log=log,
            loop_factory=loop_factory,
            observe=build_observation,
            capability_text=describe_capabilities(caps()),
            budget=budget, budget_s=caps().mission_budget_s or MISSION_BUDGET_S,
            subgoal_s=(caps().subgoal_median_s, caps().subgoal_verdict_s))
        if request.monitor:
            # After the mission, because both of its callables are the mission's: the tee it
            # reads and the filter that decides what an alert does.
            driver.monitor = SceneMonitor(clients["monitor"], planner_prompts,
                                          latest=driver.latest, on_alert=driver.on_alert)
        with mission.lock:
            mission.task, mission.budget = task, budget
            mission.monitor = bool(request.monitor)
            mission.events, mission.error, mission.result = [], "", None
            mission.status, mission.running = "running", True
            mission.started_at = time.time()
            mission.mission, mission.log = driver, log

        def work():
            try:
                result = driver.run(task, on_event=recorded(mission.append, "mission"))
                with mission.lock:
                    mission.result = result.as_dict()
                    mission.status = result.status
                    mission.error = result.error
            except Exception as exc:
                with mission.lock:
                    mission.status = "error"
                    mission.error = "{}: {}".format(type(exc).__name__, exc)
            finally:
                stop_recording(log)
                with mission.lock:
                    mission.running = False

        threading.Thread(target=work, name="mission", daemon=True).start()
        return {"ok": True, "task": task, "budget": budget}

    @app.post("/api/mission/stop")
    def api_mission_stop():
        """Cancel the mission, which cancels whatever executor loop it is holding, and stop
        the robot."""
        mission.cancel()
        try:
            adapter().stop()
        except Exception as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=200)
        mission.append({"event": "cancelled_by_operator"})
        return {"ok": True}

    @app.get("/api/run")
    def api_run():
        return run.snapshot()

    @app.post("/api/clear_fault")
    def api_clear_fault():
        """The operator's word that a fault the arm stopped on has been dealt with at the robot:
        nothing clears one but this, or a restart."""
        clear = getattr(adapter(), "clear_fault", None) if backends.current is not None else None
        if not callable(clear):
            return JSONResponse({"ok": False, "error": "this backend latches no faults"},
                                status_code=200)
        clear()
        return {"ok": True}

    @app.post("/api/stop")
    def api_stop():
        mission.cancel()                    # cancels the mission's own executor loop too
        if executor.loop is not None:
            executor.loop.cancel()
        try:
            adapter().stop()
        except Exception as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=200)
        run.append({"event": "stopped_by_operator"})
        return {"ok": True}

    app.state.mission = mission             # what a test flips to prove the refusals
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=3008)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--backend", choices=("xarm6", "libero"), default="xarm6",
                        help="which backend to connect at startup; the page can switch")
    parser.add_argument("--bridge", default=os.environ.get("MOTION_BRIDGE_URL"))
    parser.add_argument("--token", default=os.environ.get("XARM_BRIDGE_TOKEN", ""))
    parser.add_argument("--cameras", default="",
                        help="comma separated cameras the EXECUTOR may use; default all. The "
                             "page still shows every camera -- this only limits what the model "
                             "is given, for a view that is miscalibrated or unhelpful.")
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0,
                        help="which of LIBERO's 50 saved initial states to start from")
    args = parser.parse_args()

    EXECUTOR_CAMERAS.extend(n for n in args.cameras.split(",") if n.strip())
    backends = Backends({"bridge": args.bridge, "token": args.token,
                         "suite": args.suite, "task": args.task, "seed": args.seed})
    try:
        backends.select(args.backend)
        c = backends.adapter().capabilities()
        print("backend {}: tool offset {:.0f} mm, {:.0f} mm / {:.1f} deg per step"
              .format(c.name, c.control_point_offset_m * 1000,
                      c.max_translation_m * 1000, math.degrees(c.max_rotation_rad)))
        for note in c.notes:
            print("  note: {}".format(note))
    except Exception as exc:
        backends.errors[args.backend] = str(exc)
        print("WARNING: {} did not connect: {}".format(args.backend, exc))
        print("the page will load and offer the other backend.")

    if "libero" in backends.names and not os.environ.get("MUJOCO_GL"):
        print("note: switching to LIBERO from the page needs MUJOCO_GL=egl (and MUJOCO_EGL_DEVICE_ID) "
              "in this process's environment")

    import uvicorn
    print("page on http://{}:{}".format(args.host, args.port))
    uvicorn.run(build_app(backends, load_directions()), host=args.host, port=args.port,
                log_level="warning")


if __name__ == "__main__":
    main()
