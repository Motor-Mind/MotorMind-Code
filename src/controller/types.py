"""What the controller hands an adapter, and what it expects back."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

import numpy as np

from ..schema import actions as _schema

# The action kinds: the schema's one copy, named here for the adapters that read them.
TRANSLATE, ROTATE, GRIPPER, WAIT, HOME = (_schema.TRANSLATE, _schema.ROTATE, _schema.GRIPPER,
                                          _schema.WAIT, _schema.HOME)


def _mm(metres: Optional[float]) -> Optional[float]:
    """Metres as millimetres to 0.1 mm, with None kept None."""
    return None if metres is None else round(metres * 1000.0, 1)


@dataclass(frozen=True)
class Pose:
    """A frame in the robot's base frame: metres and a rotation matrix."""

    position_m: np.ndarray
    rotation: np.ndarray

    @staticmethod
    def of(position_m: Sequence[float], rotation: np.ndarray) -> "Pose":
        return Pose(np.asarray(position_m, dtype=float).reshape(3),
                    np.asarray(rotation, dtype=float).reshape(3, 3))

    def moved(self, offset_m: Sequence[float]) -> "Pose":
        return Pose(self.position_m + np.asarray(offset_m, dtype=float), self.rotation)

    def turned(self, rotation_delta: np.ndarray) -> "Pose":
        """Rotate in place, about this pose's own origin."""
        return Pose(self.position_m, np.asarray(rotation_delta, dtype=float) @ self.rotation)

    def as_dict(self) -> Dict[str, Any]:
        return {"position_mm": [round(v * 1000.0, 2) for v in self.position_m],
                "rotation": [[round(v, 6) for v in row] for row in self.rotation]}


#: Within DOWN_WITHIN_DEG of vertical -- either way up: an identity frame says nothing about a
#: tilt -- the jaws point exactly DOWN, and every rule counting along them counts as it did.
DOWN, DOWN_WITHIN_DEG = np.array([0.0, 0.0, -1.0]), 45.0


def approach_axis(pose: Optional[Pose]) -> np.ndarray:
    """The unit vector the fingers point along -- the tool's +z -- in the base frame."""
    axis = DOWN if pose is None else np.asarray(pose.rotation, dtype=float)[:, 2]
    axis = axis / float(np.linalg.norm(axis))
    return DOWN.copy() if abs(axis[2]) >= np.cos(np.radians(DOWN_WITHIN_DEG)) else axis


@dataclass
class TcpDelta:
    """One semantic action, resolved into a change at the tool."""

    kind: str
    index: int = 0
    label: str = ""
    axis: Optional[np.ndarray] = None
    magnitude: float = 0.0
    speed: float = 0.0                      # m/s, or rad/s for a rotate
    stop_on_contact: bool = False
    # A translate that EXPECTS to meet something: a drawer, a door, a lever.
    push: bool = False
    gripper_state: Optional[str] = None
    # How far to open, in millimetres of pad gap, for a backend that can measure its own jaws.
    open_to_mm: Optional[float] = None
    seconds: float = 0.0
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"kind": self.kind, "label": self.label, "index": self.index}
        if self.axis is not None:
            out["axis"] = [round(float(v), 4) for v in self.axis]
        if self.kind == TRANSLATE:
            out["distance_mm"] = round(self.magnitude * 1000.0, 2)
            out["speed_mm_s"] = round(self.speed * 1000.0, 2)
        elif self.kind == ROTATE:
            out["angle_deg"] = round(np.degrees(self.magnitude), 2)
            out["speed_deg_s"] = round(np.degrees(self.speed), 2)
        if self.kind == GRIPPER:
            out["state"] = self.gripper_state
            if self.open_to_mm is not None:
                out["open_to_mm"] = round(float(self.open_to_mm), 1)
        if self.kind == WAIT:
            out["seconds"] = self.seconds
        if self.stop_on_contact:
            out["stop_on_contact"] = True
        if self.push:
            out["push"] = True
        if self.note:
            out["note"] = self.note
        return out


@dataclass
class GripperReading:
    """What a backend can say about its jaws, in neutral terms."""

    opening_m: Optional[float] = None
    raw: Optional[float] = None
    raw_open: Optional[float] = None
    raw_closed: Optional[float] = None
    open_span_m: Optional[float] = None
    commanded: Optional[str] = None
    moving: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {"opening_mm": None if self.opening_m is None else round(self.opening_m * 1000, 2),
                "raw": self.raw, "raw_open": self.raw_open, "raw_closed": self.raw_closed,
                "commanded": self.commanded, "moving": self.moving}


@dataclass
class Capabilities:
    """What one robot will accept, so the controller can say why a plan looks as it does."""

    name: str
    max_translation_m: float
    max_rotation_rad: float
    max_speed_m_s: float
    max_rotation_speed_rad_s: float
    # What a command that names no speed should use.
    default_speed_m_s: Optional[float] = None
    default_rotation_speed_rad_s: Optional[float] = None
    control_point_offset_m: float = 0.0
    # The table's surface in base z, measured ONCE when the robot was commissioned -- not per
    # frame.
    table_z_m: Optional[float] = None
    # How long a whole mission may take on this robot: its motion is what costs the time.
    mission_budget_s: Optional[float] = None
    supports_contact_stop: bool = False
    contact_stop_trusted: bool = False
    supports_home: bool = True
    gripper_gap_measured: bool = False
    # How far the jaws open, measured once.
    gripper_open_span_m: Optional[float] = None
    # How long the finger pads are along the approach axis, measured once.
    gripper_pad_length_m: Optional[float] = None
    # How far out, horizontally from the base, this arm has been WORKED so far -- where the
    # objects of the tasks it has run have stood.
    comfortable_reach_m: Optional[float] = None
    # How far out, horizontally from the base, this arm still goes where it is sent.
    reach_m: Optional[float] = None
    # How far each rotation word tilts the jaws before the arm stops, and how low the tool point
    # gets tilted level, measured once; None where nobody has: nothing is taken from the side.
    tilt_limits_deg: Optional[Dict[str, float]] = None
    tilted_floor_m: Optional[float] = None
    # The lowest the tool point may ever be sent, above table_z_m: a hard floor under every
    # move, whatever a camera says is under the tool. None: no floor beyond the other rules.
    floor_above_table_m: Optional[float] = None
    # How close to a commanded point this robot counts itself arrived.
    arrival_tolerance_m: Optional[float] = None
    # How far a tilted take reaches round from straight down, in degrees, before a side take
    # counts as level; how tall the things this robot works on get (the error bar of a height
    # nobody measured); and what a subgoal and the verdict ending it take on this robot, which
    # prices a replan. None: the fallbacks the code names (measured on LIBERO).
    tilt_reach_deg: Optional[float] = None
    max_object_height_m: Optional[float] = None
    subgoal_median_s: Optional[float] = None
    subgoal_verdict_s: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "max_translation_mm": round(self.max_translation_m * 1000.0, 1),
            "max_rotation_deg": round(float(np.degrees(self.max_rotation_rad)), 2),
            "max_speed_mm_s": round(self.max_speed_m_s * 1000.0, 1),
            "default_speed_mm_s": _mm(self.default_speed_m_s),
            "max_rotation_speed_deg_s": round(float(np.degrees(self.max_rotation_speed_rad_s)), 1),
            "control_point_offset_mm": round(self.control_point_offset_m * 1000.0, 1),
            "table_z_mm": _mm(self.table_z_m),
            "supports_contact_stop": self.supports_contact_stop,
            "contact_stop_trusted": self.contact_stop_trusted,
            "supports_home": self.supports_home,
            "gripper_gap_measured": self.gripper_gap_measured,
            "gripper_open_span_mm": _mm(self.gripper_open_span_m),
            "gripper_pad_length_mm": _mm(self.gripper_pad_length_m),
            "comfortable_reach_mm": _mm(self.comfortable_reach_m),
            "reach_mm": _mm(self.reach_m),
            "tilt_limits_deg": self.tilt_limits_deg, "tilted_floor_mm": _mm(self.tilted_floor_m),
            "floor_above_table_mm": _mm(self.floor_above_table_m),
            "arrival_tolerance_mm": _mm(self.arrival_tolerance_m),
            "notes": list(self.notes),
        }


@dataclass
class Step:
    """One thing the adapter will actually do. Previewable: building it moves nothing."""

    delta_index: int
    label: str
    kind: str
    chunk: int = 1
    of: int = 1
    goal: Optional[Pose] = None             # the tool pose this step ends at
    duration_s: float = 0.0
    detail: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "kind": self.kind,
                "chunk": "{}/{}".format(self.chunk, self.of),
                "goal": None if self.goal is None else self.goal.as_dict(),
                "duration_s": round(self.duration_s, 3), "detail": dict(self.detail)}


# How far a VERTICAL motion may wander sideways while the robot reports it is holding something,
# in metres.
PAYLOAD_LATERAL_M = 0.015


DONE = "done"
CONTACT = "contact"
LIMITED = "limited"
FAILED = "failed"
REFUSED = "refused"
SKIPPED = "skipped"


@dataclass
class StepResult:
    """What happened, measured rather than assumed."""

    step: Step
    outcome: str
    message: str = ""
    moved_m: float = 0.0                    # measured displacement of the tool
    along_axis_m: float = 0.0               # ...of which, along the commanded axis
    lateral_m: float = 0.0
    turned_rad: float = 0.0
    pose_after: Optional[Pose] = None
    elapsed_s: float = 0.0
    # Why the adapter stopped commanding before the goal, when it did: "blocked" (no more
    # headway along the commanded axis) or "skidded" (more sideways travel than the motion was
    # allowed).
    abort_reason: str = ""
    # Numbers this KIND of step measures and no other does, so nothing has to read them back out
    # of ``message``.
    detail: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"label": self.step.label, "kind": self.step.kind, "outcome": self.outcome,
                "message": self.message, "abort_reason": self.abort_reason,
                **({"detail": dict(self.detail)} if self.detail else {}),
                "moved_mm": round(self.moved_m * 1000.0, 2),
                "along_axis_mm": round(self.along_axis_m * 1000.0, 2),
                "lateral_mm": round(self.lateral_m * 1000.0, 2),
                "turned_deg": round(float(np.degrees(self.turned_rad)), 2),
                "elapsed_s": round(self.elapsed_s, 2)}


class RobotAdapter(Protocol):
    """What a robot must offer for the controller to drive it."""

    def capabilities(self) -> Capabilities:
        ...

    def connect(self) -> None:
        ...

    def tcp_pose(self) -> Pose:
        """Where the tool is now."""

    def steps_for(self, delta: TcpDelta, start: Pose) -> List[Step]:
        """Split one delta into executable steps. Moves nothing."""

    def run_step(self, step: Step) -> StepResult:
        """Execute one step and report what was measured."""

    def stop(self) -> None:
        ...

    def close(self) -> None:
        ...
