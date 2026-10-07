"""The semantic mid-level movement JSON: what it looks like and how it is read.

    {
      "speed_mm_s": 10,
      "actions": [
        {"type": "move",    "direction": "forward",  "distance_mm": 60},
        {"type": "rotate",  "direction": "yaw_left", "angle_deg": 30},
        {"type": "gripper", "state": "close"},
        {"type": "wait",    "seconds": 0.5},
        {"type": "home"}
      ]
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

BASE_FRAME = "base"

TRANSLATE = "translate"
ROTATE = "rotate"
GRIPPER = "gripper"
HOME = "home"
WAIT = "wait"

_COMMON_KEYS = {"frame", "speed_mm_s", "rot_speed_deg_s", "stop_on_contact", "note"}


class ActionError(ValueError):
    """A command that cannot be read, or that asks for something incoherent."""


@dataclass
class Action:
    """One semantic action, already validated but not yet resolved against the robot."""

    kind: str
    frame: str
    index: int = 0
    direction: Optional[str] = None          # the named word, when one was used
    axis: Optional[Sequence[float]] = None   # explicit axis in the frame, when one was given
    distance_mm: float = 0.0
    angle_deg: float = 0.0
    speed_mm_s: Optional[float] = None
    rot_speed_deg_s: Optional[float] = None
    stop_on_contact: Optional[bool] = None
    command: Optional[str] = None            # gripper: the requested state
    width_mm: Optional[float] = None         # gripper: target gap, when given
    seconds: float = 0.0                     # wait
    note: str = ""

    def describe(self) -> str:
        if self.kind == TRANSLATE:
            what = self.direction or "axis {}".format(list(self.axis or ()))
            return "move {} {:.1f} mm ({})".format(what, self.distance_mm, self.frame)
        if self.kind == ROTATE:
            what = self.direction or "axis {}".format(list(self.axis or ()))
            return "rotate {} {:.1f} deg ({})".format(what, self.angle_deg, self.frame)
        if self.kind == GRIPPER:
            if self.width_mm is not None:
                return "gripper {} to {:.1f} mm".format(self.command, self.width_mm)
            return "gripper {}".format(self.command)
        if self.kind == HOME:
            return "go home"
        if self.kind == WAIT:
            return "wait {:.2f} s".format(self.seconds)
        return "{} ({})".format(self.kind, self.frame)


@dataclass
class Command:
    """A whole parsed command: the actions in order."""

    actions: List[Action] = field(default_factory=list)


def _as_float(value: Any, what: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ActionError("{} must be a number, got {!r}".format(what, value))
    if out != out or out in (float("inf"), float("-inf")):
        raise ActionError("{} must be finite, got {!r}".format(what, value))
    return out


def _as_axis(value: Sequence[Any], what: str) -> List[float]:
    if len(value) != 3:
        raise ActionError("{} must have three components, got {!r}".format(what, value))
    axis = [_as_float(v, what) for v in value]
    if max(abs(v) for v in axis) < 1e-9:
        raise ActionError("{} is a zero vector, which names no direction".format(what))
    return axis


def _as_bool(value: Any, what: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ActionError("{} must be true or false, got {!r}".format(what, value))


_FIELDS = {
    "move": {"direction", "axis", "distance_mm"},
    "rotate": {"direction", "axis", "angle_deg"},
    "gripper": {"state", "width_mm"},
    "home": set(),
    "wait": {"seconds"},
}
_TYPES = tuple(_FIELDS)


def _magnitude(entry: Dict[str, Any], key: str, index: int, kind: str, word: str,
               verb: str) -> float:
    """A distance or an angle: present, a number, and greater than zero."""
    if key not in entry:
        raise ActionError("action {} ({}) needs {}".format(index, kind, key))
    value = _as_float(entry[key], key)
    if value < 0:
        raise ActionError(
            "action {}: {} is a magnitude and cannot be negative -- use the "
            "opposite {} word, or negate the axis".format(index, key, word)
        )
    if value == 0:
        raise ActionError("action {}: {} is zero, which {} nothing".format(index, key, verb))
    return value


def _parse_action(entry: Any, index: int, inherited: Dict[str, Any]) -> Action:
    if not isinstance(entry, dict):
        raise ActionError("action {} must be an object, got {!r}".format(index, entry))

    if "type" not in entry:
        raise ActionError(
            "action {} has no \"type\". Every action names one of {}, and its other fields "
            "say how. Got keys {}".format(
                index, ", ".join(_TYPES), ", ".join(sorted(entry)) or "none")
        )

    kind = entry["type"]
    if not isinstance(kind, str) or kind.strip().lower() not in _TYPES:
        raise ActionError(
            "action {}: type must be one of {}, got {!r}".format(
                index, ", ".join(_TYPES), kind)
        )
    kind = kind.strip().lower()

    unknown = set(entry) - _COMMON_KEYS - _FIELDS[kind] - {"type"}
    if unknown:
        raise ActionError(
            "action {} ({}) does not take {}. It takes {}".format(
                index, kind, ", ".join(sorted(unknown)),
                ", ".join(sorted(_FIELDS[kind] | _COMMON_KEYS)) or "no other fields")
        )

    frame = entry.get("frame", inherited.get("frame"))
    speed = entry.get("speed_mm_s", inherited.get("speed_mm_s"))
    rot_speed = entry.get("rot_speed_deg_s", inherited.get("rot_speed_deg_s"))
    contact = entry.get("stop_on_contact", inherited.get("stop_on_contact"))
    common = dict(
        index=index,
        speed_mm_s=None if speed is None else _as_float(speed, "speed_mm_s"),
        rot_speed_deg_s=None if rot_speed is None else _as_float(rot_speed, "rot_speed_deg_s"),
        stop_on_contact=None if contact is None else _as_bool(contact, "stop_on_contact"),
        note=str(entry.get("note", "")),
    )

    if frame is not None and str(frame).strip().lower() != BASE_FRAME:
        raise ActionError(
            "action {}: the only frame is {!r}, got {!r}. Directions are resolved in the "
            "arm's base frame".format(index, BASE_FRAME, frame)
        )
    frame = BASE_FRAME

    if kind in ("move", "rotate"):
        has_direction, has_axis = "direction" in entry, "axis" in entry
        if has_direction == has_axis:
            raise ActionError(
                "action {} ({}) needs exactly one of \"direction\" (a word from this "
                "frame's table) or \"axis\" ([x, y, z] in the frame); {}".format(
                    index, kind, "it has both" if has_direction else "it has neither")
            )
        direction = axis = None
        if has_direction:
            if not isinstance(entry["direction"], str):
                raise ActionError(
                    "action {}: direction must be a word, got {!r}. For a raw vector use "
                    "\"axis\"".format(index, entry["direction"])
                )
            direction = entry["direction"].strip().lower()
        elif has_axis:
            if not isinstance(entry["axis"], (list, tuple)):
                raise ActionError(
                    "action {}: axis must be [x, y, z], got {!r}".format(index, entry["axis"])
                )
            axis = _as_axis(entry["axis"], "action {} axis".format(index))

    if kind == "move":
        distance = _magnitude(entry, "distance_mm", index, kind, "direction", "moves")
        return Action(kind=TRANSLATE, frame=frame, direction=direction, axis=axis,
                      distance_mm=distance, **common)

    if kind == "rotate":
        angle = _magnitude(entry, "angle_deg", index, kind, "rotation", "rotates")
        return Action(kind=ROTATE, frame=frame, direction=direction, axis=axis,
                      angle_deg=angle, **common)

    if kind == "gripper":
        if "state" not in entry:
            raise ActionError(
                "action {} (gripper) needs \"state\": \"open\" or \"close\"".format(index))
        state = str(entry["state"]).strip().lower()
        if state not in ("open", "close"):
            raise ActionError(
                "action {}: gripper state is \"open\" or \"close\", got {!r}".format(
                    index, entry["state"])
            )
        width = entry.get("width_mm")
        return Action(kind=GRIPPER, frame=frame, command=state,
                      width_mm=None if width is None else _as_float(width, "width_mm"),
                      **common)

    if kind == "home":
        return Action(kind=HOME, frame=frame, **common)

    if "seconds" not in entry:
        raise ActionError("action {} (wait) needs seconds".format(index))
    seconds = _as_float(entry["seconds"], "seconds")
    if seconds <= 0:
        raise ActionError("action {}: seconds must be positive".format(index))
    return Action(kind=WAIT, frame=frame, seconds=seconds, **common)


def parse_command(payload: Any) -> Command:
    """Read a semantic command from a JSON string, a dict, or a list of actions."""
    if isinstance(payload, (str, bytes)):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ActionError("not valid JSON: {}".format(exc))

    inherited: Dict[str, Any] = {}
    if isinstance(payload, dict) and "actions" in payload:
        unknown = set(payload) - _COMMON_KEYS - {"actions"}
        if unknown:
            raise ActionError("unknown top-level key(s) {}".format(", ".join(sorted(unknown))))
        for key in _COMMON_KEYS:
            if key in payload:
                inherited[key] = payload[key]
        entries = payload["actions"]
        if not isinstance(entries, list):
            raise ActionError("\"actions\" must be a list")
    elif isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        entries = [payload]
    else:
        raise ActionError("a command is an object or a list, got {!r}".format(type(payload).__name__))

    if not entries:
        raise ActionError("the command contains no actions")

    actions = [_parse_action(entry, i, inherited) for i, entry in enumerate(entries)]
    return Command(actions=actions)
