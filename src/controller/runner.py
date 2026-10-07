"""Drive a robot through a list of deltas, and report what it actually did."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from . import types
from .types import DONE, Pose, RobotAdapter, Step, StepResult, TcpDelta

Listener = Optional[Callable[[Dict[str, Any]], None]]


@dataclass
class Plan:
    steps: List[Step] = field(default_factory=list)
    start: Optional[Pose] = None
    end: Optional[Pose] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return sum(step.duration_s for step in self.steps)

    def as_dict(self) -> Dict[str, Any]:
        return {"start": None if self.start is None else self.start.as_dict(),
                "end": None if self.end is None else self.end.as_dict(),
                "total_duration_s": round(self.duration_s, 2),
                "warnings": list(self.warnings),
                "steps": [step.as_dict() for step in self.steps]}


@dataclass
class RunReport:
    results: List[StepResult] = field(default_factory=list)
    stopped_early: bool = False
    reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"stopped_early": self.stopped_early, "reason": self.reason,
                "results": [r.as_dict() for r in self.results]}


def plan(adapter: RobotAdapter, deltas: List[TcpDelta],
         start: Optional[Pose] = None) -> Plan:
    """Turn deltas into the steps this robot will take. Moves nothing."""
    capabilities = adapter.capabilities()
    pose = start if start is not None else adapter.tcp_pose()
    out = Plan(start=pose)
    for delta in deltas:
        steps = adapter.steps_for(delta, pose)
        out.steps.extend(steps)
        for step in steps:
            if step.goal is not None:
                pose = step.goal
        if len(steps) > 1:
            out.warnings.append(
                "{}: split into {} steps -- {} accepts at most {:.0f} mm and {:.1f} deg at "
                "once".format(delta.label, len(steps), capabilities.name,
                              capabilities.max_translation_m * 1000,
                              np.degrees(capabilities.max_rotation_rad)))
        if delta.stop_on_contact and not capabilities.contact_stop_trusted:
            out.warnings.append(
                "{}: asks for stop_on_contact, which {} reports as not trustworthy"
                .format(delta.label, capabilities.name))
    out.end = pose
    if any(d.stop_on_contact for d in deltas) and len(out.steps) > 1:
        out.warnings.append(
            "a contact stop ends the whole command: everything planned after it is dropped, "
            "because it was planned from a pose the robot never reached")
    return out


def run(adapter: RobotAdapter, deltas: List[TcpDelta], on_event: Listener = None,
        tolerance_m: float = 0.005) -> RunReport:
    """Execute the plan, measuring each step against what it asked for."""
    report = RunReport()
    the_plan = plan(adapter, deltas)

    def emit(event: Dict[str, Any]) -> None:
        if on_event is not None:
            on_event(event)

    for position, step in enumerate(the_plan.steps):
        if report.stopped_early:
            report.results.append(StepResult(step=step, outcome=types.SKIPPED,
                                             message="not run: " + report.reason))
            continue
        emit({"event": "step_start", "index": position, "of": len(the_plan.steps),
              "label": step.label})
        result = adapter.run_step(step)
        report.results.append(result)
        emit({"event": "step_end", "index": position, **result.as_dict()})

        if result.outcome == DONE and step.kind == types.TRANSLATE:
            asked = step.detail.get("distance_m")
            if asked and abs(result.along_axis_m - asked) > tolerance_m:
                result.outcome = types.LIMITED
                report.stopped_early = True
                report.reason = (
                    "{}: asked for {:.1f} mm along the commanded axis, measured {:.1f} mm. "
                    "Either the robot was blocked, or the direction table is wrong for this "
                    "mount.".format(step.label, asked * 1000, result.along_axis_m * 1000))
        if result.outcome != DONE and not report.stopped_early:
            report.stopped_early = True
            report.reason = "{} ended {}: {}".format(step.label, result.outcome, result.message)
    return report
