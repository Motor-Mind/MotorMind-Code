"""Load one mission recording directory, read-only.

A recording directory (written by ``src/recording.py``) holds::

    calls.jsonl     one model call per line: role, subgoal, cycle, t (when it ended),
                    latency_s, system, prompt, images (downscaled copies), reply_json
    cameras.jsonl   one camera frame per line: camera, timestamp (capture), t (logged),
                    rgb_jpeg, depth_png16 (file names under images/), intrinsic (3x3),
                    cam2base (4x4, metres), width, height, depth_scale, subgoal, cycle
    events.jsonl    the executor's event stream: planned, subgoal_start, reached (every look,
                    with each view's box_px), step_start/step_end (every motion, with its
                    base-frame axis and measured along_axis_mm), held, letting_go, finished...
    evidence.jsonl  the evidence log rows; run_log.md the human-readable log;
    recording.json  counts and problems.

The tool pose is not an event field. It is recovered from the wrist camera: the camera is
rigidly mounted, so the tool point is one fixed point in the wrist camera's frame. That point
is fitted by least squares from the executor prompts that state "tool at [x, y, z]" (or "tool
point is at [...]") against the wrist frame captured just before each call
(:meth:`Recording.tool_in_wrist`). Once fitted, every wrist frame gives the tool point.

Usage::

    rec = Recording(path)
    cams = rec.cameras_at(t)            # {name: camera dict}, the payload shape geometry takes
    tool = rec.tool_at(t)               # xyz metres, base frame, or None
    for close in rec.closes(): ...      # every gripper close, where, and whether it held
"""

from __future__ import annotations

import base64
import bisect
import json
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np

_TOOL_SAID = re.compile(r"tool (?:point is )?at \[(-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?), "
                        r"(-?\d+(?:\.\d+)?)\]")
_TABLE_SAID = re.compile(r"table is at z = (-?\d+(?:\.\d+)?) mm")
_GRIPPER_SAID = re.compile(r"^gripper: (.*)$", re.MULTILINE)
WRIST = "wrist"


def _jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def call_start(call: Dict[str, Any]) -> float:
    """When a call was sent: its logged time is when it ended."""
    return float(call["t"]) - float(call.get("latency_s") or 0.0)


class Recording:
    """One recording directory. Nothing is written; images are read on demand."""

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self.name = os.path.basename(self.path.rstrip("/"))
        self.calls = _jsonl(os.path.join(path, "calls.jsonl"))
        self.events = _jsonl(os.path.join(path, "events.jsonl"))
        self.evidence = _jsonl(os.path.join(path, "evidence.jsonl"))
        self.frames = sorted(_jsonl(os.path.join(path, "cameras.jsonl")),
                             key=lambda f: float(f.get("timestamp", f["t"])))
        log = os.path.join(path, "run_log.md")
        self.run_log = open(log).read() if os.path.exists(log) else ""
        info = os.path.join(path, "recording.json")
        self.info = json.load(open(info)) if os.path.exists(info) else {}
        self._by_camera: Dict[str, List[Dict[str, Any]]] = {}
        for frame in self.frames:
            self._by_camera.setdefault(frame["camera"], []).append(frame)
        self._stamps = {name: [float(f.get("timestamp", f["t"])) for f in frames]
                        for name, frames in self._by_camera.items()}
        self._tool_in_wrist: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ what the run was

    @property
    def status(self) -> str:
        done = [e for e in self.events if e.get("event") == "finished"]
        return done[-1].get("status", "") if done else ""

    @property
    def task(self) -> str:
        for call in self.calls:
            if call.get("role") == "planner" and "TASK" in (call.get("prompt") or ""):
                match = re.search(r"TASK\s*\n(.+)", call["prompt"])
                if match:
                    return match.group(1).strip()
        return ""

    def table_z_m(self) -> Optional[float]:
        """The table height the prompts stated, in metres (the rig's commissioned value)."""
        for call in self.calls:
            match = _TABLE_SAID.search(call.get("prompt") or "")
            if match:
                return float(match.group(1)) / 1000.0
        return None

    # ------------------------------------------------------------------ frames

    def camera_names(self) -> List[str]:
        return sorted(self._by_camera)

    def frame_at(self, name: str, t: float) -> Optional[Dict[str, Any]]:
        """The last ``name`` frame captured at or before ``t``."""
        stamps = self._stamps.get(name) or []
        index = bisect.bisect_right(stamps, float(t)) - 1
        return self._by_camera[name][index] if index >= 0 else None

    def frame_after(self, name: str, t: float) -> Optional[Dict[str, Any]]:
        """The first ``name`` frame captured after ``t``."""
        stamps = self._stamps.get(name) or []
        index = bisect.bisect_right(stamps, float(t))
        return self._by_camera[name][index] if index < len(stamps) else None

    def camera(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        """A frame as the live payload carries it: rgb JPEG bytes as ``image`` and the 16-bit
        depth PNG base64-encoded as ``depth_png16`` (what geometry.depth_metres and
        controller.clearance.from_wrist decode)."""
        out = {key: frame[key] for key in ("camera", "intrinsic", "cam2base", "width", "height",
                                           "depth_scale", "depth_min_range_m", "timestamp")
               if key in frame}
        out["image"] = self._read(frame.get("rgb_jpeg"))
        depth = self._read(frame.get("depth_png16"))
        out["depth_png16"] = None if depth is None else base64.b64encode(depth).decode()
        out["rgb_path"] = None if not frame.get("rgb_jpeg") else \
            os.path.join(self.path, "images", frame["rgb_jpeg"])
        return out

    def cameras_at(self, t: float, names=None) -> Dict[str, Dict[str, Any]]:
        """Every camera's last frame at or before ``t``, as :meth:`camera` gives it."""
        out = {}
        for name in names or self.camera_names():
            frame = self.frame_at(name, t)
            if frame is not None:
                out[name] = self.camera(frame)
        return out

    def _read(self, name: Optional[str]) -> Optional[bytes]:
        if not name:
            return None
        path = os.path.join(self.path, "images", name)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as handle:
            return handle.read()

    # ------------------------------------------------------------------ the tool

    def tool_in_wrist(self) -> Optional[np.ndarray]:
        """The tool point in the wrist camera's frame (metres), fitted from the prompts."""
        if self._tool_in_wrist is not None:
            return self._tool_in_wrist
        rotations, rest = [], []
        for call in self.calls:
            match = _TOOL_SAID.search(call.get("prompt") or "")
            frame = None if match is None else self.frame_at(WRIST, call_start(call))
            if frame is None:
                continue
            pose = np.asarray(frame["cam2base"], dtype=float)
            rotations.append(pose[:3, :3])
            rest.append(np.array([float(v) for v in match.groups()]) / 1000.0 - pose[:3, 3])
        if len(rotations) < 3:
            return None
        point, *_ = np.linalg.lstsq(np.vstack(rotations), np.concatenate(rest), rcond=None)
        self._tool_in_wrist = point
        return point

    def tool_of(self, frame: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
        """The tool point (xyz, metres, base frame) when this wrist frame was captured."""
        point = self.tool_in_wrist()
        if frame is None or point is None:
            return None
        pose = np.asarray(frame["cam2base"], dtype=float)
        return pose[:3, :3] @ point + pose[:3, 3]

    def tool_at(self, t: float) -> Optional[np.ndarray]:
        """The tool point at ``t``: the last wrist frame's, plus every translation the robot
        reported finishing since (step_end axis x along_axis_mm) -- the arm moves between
        frames."""
        frame = self.frame_at(WRIST, t)
        tool = self.tool_of(frame)
        if tool is None:
            return None
        since = float(frame.get("timestamp", frame["t"]))
        for event in self.events:
            if event.get("event") == "step_end" and event.get("kind") == "translate" \
                    and since < float(event["t"]) <= float(t) and event.get("axis"):
                tool = tool + np.asarray(event["axis"], dtype=float) \
                    * float(event.get("along_axis_mm") or 0.0) / 1000.0
        return tool

    # ------------------------------------------------------------------ the gripper

    def grasp_said(self, start: float, end: float) -> List[str]:
        """Every "gripper: ..." status a prompt sent between ``start`` and ``end`` stated."""
        out = []
        for call in self.calls:
            if float(start) < call_start(call) < float(end):
                match = _GRIPPER_SAID.search(call.get("prompt") or "")
                if match:
                    out.append(match.group(1))
        return out

    def _gripper_steps(self) -> List[Dict[str, Any]]:
        return [e for e in self.events if e.get("event") == "step_end"
                and e.get("kind") == "gripper" and e.get("outcome") == "done"]

    def closes(self) -> List[Dict[str, Any]]:
        """Every gripper close: {t, subgoal, tool (xyz m at the close), holding}. ``holding``
        is whether the executor logged a "held" event, or any prompt sent, before the next
        gripper command reported the jaws "holding something" (the first prompt after a close
        can carry the state from before it, so the first alone is not enough)."""
        steps, out = self._gripper_steps(), []
        for index, event in enumerate(steps):
            if "close" not in str(event.get("label")):
                continue
            end = float(steps[index + 1]["t"]) if index + 1 < len(steps) else float("inf")
            said = self.grasp_said(float(event["t"]), end)
            out.append({"t": float(event["t"]), "subgoal": event.get("subgoal", ""),
                        "tool": self.tool_at(float(event["t"])), "until": end,
                        "holding": any("holding something" in s for s in said) or any(
                            float(event["t"]) <= float(e["t"]) < end
                            for e in self.events_named("held"))})
        return out

    def opens(self) -> List[Dict[str, Any]]:
        """Every gripper open: {t, subgoal, tool}."""
        return [{"t": float(e["t"]), "subgoal": e.get("subgoal", ""),
                 "tool": self.tool_at(float(e["t"]))}
                for e in self.events if e.get("event") == "step_end"
                and e.get("kind") == "gripper" and "open" in str(e.get("label"))
                and e.get("outcome") == "done"]

    # ------------------------------------------------------------------ looks

    def looks(self) -> List[Dict[str, Any]]:
        """Every recorded look ("reached" event): {t, capture_t, subgoal, words, views: {name:
        {label, box_px}}, located (as recorded), asked_a_model}. ``capture_t`` is when the
        frames it looked at were taken: before the first locate call it made, or, with no
        call, before the event."""
        locate_calls = [call_start(c) for c in self.calls
                        if c.get("role") == "executor"
                        and (c.get("system") or "").startswith("You look at pictures")]
        out, previous = [], float("-inf")
        for event in self.events:
            if event.get("event") != "reached":
                continue
            t = float(event["t"])
            mine = [s for s in locate_calls if previous < s <= t]
            located = event.get("located") or {}
            out.append({"t": t, "capture_t": min(mine) if mine else t,
                        "subgoal": event.get("subgoal", ""), "cycle": event.get("cycle"),
                        "words": event.get("target", ""), "views": located.get("views") or {},
                        "located": located, "asked_a_model": bool(event.get("asked_a_model"))})
            previous = t
        return out

    def events_named(self, name: str) -> List[Dict[str, Any]]:
        return [e for e in self.events if e.get("event") == name]


def recordings(root: str) -> List[Recording]:
    """Every recording directory directly under ``root`` (one with an events.jsonl)."""
    return [Recording(os.path.join(root, name)) for name in sorted(os.listdir(root))
            if os.path.exists(os.path.join(root, name, "events.jsonl"))]
