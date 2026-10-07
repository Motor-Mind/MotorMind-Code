"""The real xArm6, behind the HTTP bridge on the robot PC."""

from __future__ import annotations

import math
import re
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from src.controller import types
from src.controller.clearance import grasp_state
from src.controller.transforms import (axis_angle_to_matrix, matrix_to_axis_angle,
                                       rotation_magnitude)
from src.controller.types import (Capabilities, GripperReading, Pose, Step, StepResult,
                                  TcpDelta)

from .bridge_client import BridgeClient, BridgeError, fault_of
from .conventions import (_wrap, flange_pose_from_bridge, flange_pose_to_bridge,
                          flange_to_tool, matrix_to_rpy, tool_to_flange)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROBOT_YAML = os.path.join(ROOT, "robot", "robot.yaml")
#: The rig profile used when STORM_RIG names none: the rig storm was fitted on.
DEFAULT_RIG = "x-arm6"


def _laid_over(base: Dict[str, Any], top: Dict[str, Any]) -> Dict[str, Any]:
    """``top``'s keys over ``base``'s, mapping by mapping."""
    out = dict(base)
    for key, value in top.items():
        out[key] = _laid_over(out[key], value) if isinstance(value, dict) \
            and isinstance(out.get(key), dict) else value
    return out


def load_config(path: str = ROBOT_YAML, rig: Optional[str] = None) -> Dict[str, Any]:
    """The arm's config with its rig's measured profile laid over it: ``rig`` (a name under
    robot/rigs/ or a path), else $STORM_RIG, else DEFAULT_RIG."""
    rig = rig or os.environ.get("STORM_RIG") or DEFAULT_RIG
    profile = rig if rig.endswith(".yaml") else os.path.join(ROOT, "robot", "rigs", rig + ".yaml")
    if not os.path.exists(profile):
        raise FileNotFoundError("no rig profile {} -- copy robot/rigs/TEMPLATE.yaml, measure, "
                                "and set STORM_RIG".format(profile))
    with open(path) as base, open(profile) as top:
        return _laid_over(yaml.safe_load(base), yaml.safe_load(top))

SLOWDOWN_FACTOR = 2.5
SLOWDOWN_TRIES = 4
MAX_DURATION_S = 120.0
HOME_MARGIN_S = 90.0
#: What a failed job's message says when it ended on something only the operator may clear.
FAULT_WORDS = re.compile(r"collision|\bC\d+\b|error[ _]?code", re.IGNORECASE)


class XArm6Adapter:
    """Implements ``src.controller.types.RobotAdapter`` for the real arm."""

    def __init__(self, client: Optional[BridgeClient] = None, config_path: str = ROBOT_YAML,
                 url: Optional[str] = None, token: str = "", rig: Optional[str] = None):
        self.config = load_config(config_path, rig)
        bridge = self.config.get("bridge") or {}
        self.client = client or BridgeClient(
            url or bridge.get("url", "http://127.0.0.1:18765"),
            token or os.environ.get(bridge.get("token_env", "XARM_BRIDGE_TOKEN"), ""),
            heartbeat_period_s=float(bridge.get("heartbeat_period_s", 0.3)))
        limits = self.config["limits"]
        robot = self.config["robot"]
        defaults = self.config["defaults"]
        self.offset_m = float(robot.get("grip_site_offset_m", 0.0))
        self.expected_tcp_offset = robot.get("expected_bridge_tcp_offset_mm")
        self.offset_tolerance_mm = float(robot.get("tcp_offset_tolerance_mm", 1.0))
        self.max_translation_m = float(limits["max_translation_mm"]) / 1000.0
        self.max_rotation_rad = float(limits["max_rotation_rad"])
        self.min_duration_s = float(limits["min_duration_s"])
        self.max_duration_s = float(limits["max_duration_s"])
        self.max_chunks = int(limits.get("max_chunks", 12))
        self.contact = dict(defaults.get("contact") or {})
        self.contact_trusted = bool(self.contact.pop("trusted", False))
        self.margin_s = float(defaults.get("completion_margin_s", 1.5))
        # The table's surface in base z, from COMMISSIONING -- measured once, by fitting a plane
        # to each camera's depth inside the working box (the rig profile's workspace), not per
        # frame.
        workspace = self.config.get("workspace") or {}
        table_z_mm = workspace.get("table_z_mm")
        self.table_z_m = None if table_z_mm is None else float(table_z_mm) / 1000.0
        floor_mm = workspace.get("floor_above_table_mm")
        self._gripper_config: Dict[str, Any] = {}
        self._ready_confirmed = False
        self._caps = Capabilities(
            name="xarm6",
            max_translation_m=self.max_translation_m,
            max_rotation_rad=self.max_rotation_rad,
            max_speed_m_s=float(limits["max_speed_mm_s"]) / 1000.0,
            max_rotation_speed_rad_s=math.radians(float(limits["max_rot_speed_deg_s"])),
            default_speed_m_s=float(defaults["speed_mm_s"]) / 1000.0,
            default_rotation_speed_rad_s=math.radians(float(defaults["rot_speed_deg_s"])),
            control_point_offset_m=self.offset_m,
            table_z_m=self.table_z_m,
            floor_above_table_m=None if floor_mm is None or self.table_z_m is None
            else float(floor_mm) / 1000.0,
            mission_budget_s=float(defaults.get("mission_budget_s") or 0) or None,
            subgoal_median_s=defaults.get("subgoal_median_s"),
            subgoal_verdict_s=defaults.get("subgoal_verdict_s"),
            tilt_reach_deg=robot.get("tilt_reach_deg"),
            max_object_height_m=float(workspace.get("max_object_height_mm") or 0) / 1000.0
            or None,
            supports_contact_stop=True,
            contact_stop_trusted=self.contact_trusted,
            supports_home=True,
            # None where nobody has measured the jaws: the executor then holds a close to a
            # default half-gap and says the span is uncalibrated, rather than inventing one.
            gripper_open_span_m=float(robot.get("gripper", {}).get("full_open_gap_m") or 0)
            or None,
            gripper_pad_length_m=float(robot.get("gripper", {}).get("pad_length_m") or 0)
            or None,
            # The rating less the margin the profile documents: no reach number of this
            # adapter's own, and None if the profile has neither, which turns the rule that
            # reads it off rather than inventing a radius for an arm nobody has measured.
            reach_m=(float(robot["rated_reach_mm"]) - float(robot.get("reach_margin_mm") or 0))
            / 1000.0 if robot.get("rated_reach_mm") else None,
            notes=[],
        )

    # ------------------------------------------------------------------ protocol

    def capabilities(self) -> Capabilities:
        return self._caps

    def connect(self) -> None:
        health = self.client.check_health(self.expected_tcp_offset, self.offset_tolerance_mm)
        notes = list(health.get("readiness_issues") or [])
        self._gripper_config = dict(health.get("gripper_config") or {})
        self._caps.gripper_gap_measured = bool(
            self._gripper_config.get("width_measurement_method"))
        self._caps.notes = notes

    def tcp_pose(self) -> Pose:
        return flange_to_tool(flange_pose_from_bridge(self.client.state()["tcp_pose"]),
                              self.offset_m)

    def stop(self) -> None:
        self.client.stop()
        self._ready_confirmed = False

    # ------------------------------------------------------------------ recovery

    @staticmethod
    def _parked(health: Dict[str, Any]) -> bool:
        """Is the arm refusing motion? The bridge says so directly; state 4 is the fallback."""
        if health.get("motion_ready") is False:
            return True
        robot = health.get("robot") or {}
        return bool(robot.get("state") == 4 or robot.get("error_code"))

    def _fault(self, health) -> str:
        """An error or a collision only the operator may clear (bridge_client.fault_of)."""
        try:
            state = self.client.state()
        except BridgeError:
            state = None                  # unreadable is a fault: fault_of fails closed
        return fault_of(health, state, bool(getattr(self.client, "_contact_expected", False)))

    def readiness(self):
        """(able to move, why not): the same clear-then-check a motion does first -- an idle
        arm reports motion_ready false and state 5 until /stop clears it, which is normal --
        forgetting any earlier 'ready', since the controller can change mode under us. An arm
        stopped on an error code or an unasked-for collision is NOT cleared: not ready, with
        the code, for the operator."""
        self._ready_confirmed = False
        try:
            self._ensure_ready()
        except BridgeError as exc:
            return False, str(exc)
        return True, ""

    #: a job that ended on a fault, latched until the operator says it is dealt with
    _latched_fault = ""

    def clear_fault(self) -> None:
        """The operator's word that the fault a job ended on has been dealt with at the robot."""
        self._latched_fault, self._ready_confirmed = "", False

    def _ensure_ready(self) -> None:
        """Clear a parked arm before moving, whoever parked it -- but never one a job ended on a
        fault for: that is latched (``_latched_fault``) until the operator clears it."""
        if self._latched_fault:
            raise BridgeError("the arm stopped on '{}' and is left for the operator: deal with it "
                              "at the robot, then clear it from the page".format(
                                  self._latched_fault))
        if self._ready_confirmed:
            return
        health = self.client.health()
        if self._parked(health):
            fault = self._fault(health)
            if fault:                     # never cleared from here: /stop would erase it
                raise BridgeError(
                    "the arm will not accept motion: {} -- it needs attention at the robot, "
                    "and was not cleared".format(fault))
            self.client.stop()
            health = self.client.health()
            if self._parked(health):
                robot = health.get("robot") or {}
                raise BridgeError(
                    "the arm will not accept motion: state {}, error {}, motion_ready {}. "
                    "POST /stop did not clear it -- it needs attention at the robot."
                    .format(robot.get("state"), robot.get("error_code"),
                            health.get("motion_ready")))
        self._ready_confirmed = True

    # ------------------------------------------------------------------ optional extras
    #
    # Not part of RobotAdapter: the page feature-detects these, so a backend without cameras
    # simply does not offer them.

    def flange_pose(self) -> Pose:
        """What the bridge reports, before the tool offset -- what the cameras are fixed to."""
        return flange_pose_from_bridge(self.client.state()["tcp_pose"])

    def frames(self, cameras: Optional[List[str]] = None) -> Dict[str, Any]:
        """Only the cameras asked for: each one costs bandwidth over the tunnel. What the bridge
        does not send about a camera -- its depth range floor -- comes from the rig profile's
        cameras: table."""
        bridge = self.config.get("bridge") or {}
        wanted = list(cameras) if cameras else self.camera_names()
        payload = self.client.frames(cameras=",".join(wanted),
                                     size=int(bridge.get("frames_size", 256)))
        for name, camera in ((payload or {}).get("cameras") or {}).items():
            facts = (self.config.get("cameras") or {}).get(name) or {}
            for key in ("depth_min_range_m", "depth_floor_band_share"):
                if facts.get(key) is not None and isinstance(camera, dict):
                    camera.setdefault(key, float(facts[key]))
        return payload

    def camera_names(self) -> List[str]:
        try:
            return list(self.client.health().get("cameras") or []) or ["scene", "wrist"]
        except BridgeError:
            return ["scene", "wrist"]

    def gripper_reading(self) -> GripperReading:
        """The rig's gap is uncalibrated, but `raw` and the two endpoints are enough to tell
        an empty close from a full one -- see controller/clearance.grasp_state."""
        state = self.client.state()
        gripper = state.get("gripper") or {}
        config = self._gripper_config
        gap = gripper.get("gap_m")
        # An empty close is judged from where one really stops (rig profile), else the bridge's end.
        empty = self.config["robot"].get("gripper", {}).get("empty_close_raw")
        return GripperReading(
            opening_m=float(gap) if isinstance(gap, (int, float)) else None,
            raw=gripper.get("raw"),
            raw_open=config.get("open_raw"),
            raw_closed=empty if empty is not None else config.get("close_raw"),
            open_span_m=float(self.config["robot"].get("gripper", {}).get("full_open_gap_m") or 0)
            or None,
            commanded=gripper.get("command"),
            moving=bool(gripper.get("moving")))

    def extra_status(self) -> Dict[str, Any]:
        state = self.client.state()
        health = self.client.health()
        return {"state": state.get("state"), "error_code": state.get("error_code"),
                "collision": state.get("collision"),
                "gripper": state.get("gripper"),
                "motion_ready": health.get("motion_ready"),
                "parked": self._parked(health),
                "flange_mm": [round(v, 1) for v in state["tcp_pose"][:3]]}

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------ planning

    def steps_for(self, delta: TcpDelta, start: Pose) -> List[Step]:
        if delta.kind == types.GRIPPER:
            return [Step(delta_index=delta.index, label=delta.label, kind=types.GRIPPER,
                         duration_s=1.5, detail={"state": delta.gripper_state})]
        if delta.kind == types.WAIT:
            return [Step(delta_index=delta.index, label=delta.label, kind=types.WAIT,
                         duration_s=delta.seconds, detail={"seconds": delta.seconds})]
        if delta.kind == types.HOME:
            return [Step(delta_index=delta.index, label=delta.label, kind=types.HOME,
                         duration_s=8.0)]

        goal = (start.moved(delta.axis * delta.magnitude) if delta.kind == types.TRANSLATE
                else start.turned(axis_angle_to_matrix(delta.axis, delta.magnitude)))
        return self._chunk(delta, start, goal)

    def _chunk(self, delta: TcpDelta, start: Pose, goal: Pose) -> List[Step]:
        axis, total_turn = matrix_to_axis_angle(goal.rotation @ start.rotation.T)
        flange_travel = float(np.linalg.norm(
            tool_to_flange(goal, self.offset_m).position_m
            - tool_to_flange(start, self.offset_m).position_m))
        n = max(1,
                math.ceil(flange_travel / self.max_translation_m - 1e-9),
                math.ceil(total_turn / self.max_rotation_rad - 1e-9))

        for _ in range(6):
            steps, previous, over = [], start, False
            for i in range(1, n + 1):
                fraction = i / float(n)
                here = Pose(start.position_m + (goal.position_m - start.position_m) * fraction,
                            axis_angle_to_matrix(axis, total_turn * fraction) @ start.rotation)
                flange_here = tool_to_flange(here, self.offset_m)
                flange_prev = tool_to_flange(previous, self.offset_m)
                hop = float(np.linalg.norm(flange_here.position_m - flange_prev.position_m))
                turn = rotation_magnitude(flange_here.rotation @ flange_prev.rotation.T)
                component = max(abs(_wrap(b - a)) for a, b in
                                zip(matrix_to_rpy(flange_prev.rotation),
                                    matrix_to_rpy(flange_here.rotation)))
                if (hop > self.max_translation_m + 1e-9
                        or turn > self.max_rotation_rad + 1e-12
                        or component > self.max_rotation_rad + 1e-12):
                    over = True
                    break
                steps.append(Step(
                    delta_index=delta.index, chunk=i, of=n, kind=delta.kind,
                    label="{} [{}/{}]".format(delta.label, i, n), goal=here,
                    duration_s=self._duration(hop, turn, delta),
                    detail={"flange_pose": [round(v, 4) for v in flange_pose_to_bridge(flange_here)],
                            "distance_m": hop if delta.kind == types.TRANSLATE
                            else float(np.linalg.norm(here.position_m - previous.position_m)),
                            "flange_travel_mm": round(hop * 1000.0, 2),
                            "turn_deg": round(float(np.degrees(turn)), 2),
                            "stop_on_contact": delta.stop_on_contact or delta.push,
                            "push": bool(delta.push),
                            "axis": None if delta.axis is None else [round(float(v), 4)
                                                                     for v in delta.axis]}))
                previous = here
            if not over:
                return steps
            n += 1
            if n > self.max_chunks:
                break
        raise ValueError(
            "{} does not fit in {} jobs of {:.0f} mm / {:.1f} deg. Near a wrist singularity "
            "the roll/pitch/yaw the bridge checks can swing much further than the rotation "
            "itself; split the motion, or make it smaller."
            .format(delta.label, self.max_chunks, self.max_translation_m * 1000,
                    math.degrees(self.max_rotation_rad)))

    def _duration(self, translation_m: float, rotation_rad: float, delta: TcpDelta) -> float:
        by_translation = translation_m / delta.speed if delta.kind == types.TRANSLATE else 0.0
        by_rotation = rotation_rad / delta.speed if delta.kind == types.ROTATE else 0.0
        # even a pure rotation about the tool drags the flange, so both bound the time
        if delta.kind == types.ROTATE and self._caps.max_speed_m_s > 0:
            by_translation = translation_m / self._caps.max_speed_m_s
        want = max(by_translation, by_rotation)
        return min(self.max_duration_s, max(self.min_duration_s, want))

    # ------------------------------------------------------------------ execution

    def run_step(self, step: Step) -> StepResult:
        if step.kind == types.WAIT:
            time.sleep(step.detail.get("seconds", 0.0))
            return StepResult(step=step, outcome=types.DONE, elapsed_s=step.duration_s)
        self._ensure_ready()

        before = self.tcp_pose()
        if step.kind == types.GRIPPER:
            self.client.gripper(step.detail["state"])
            time.sleep(1.5)
            grasp = grasp_state(self.gripper_reading())
            message = ("holding something" if grasp.holding
                       else "nothing held" if grasp.holding is False
                       else "grasp unknown") + " -- " + grasp.reason
            return StepResult(step=step, outcome=types.DONE, message=message,
                              pose_after=self.tcp_pose(), elapsed_s=1.5)

        if step.kind == types.HOME:
            result = self.client.run_job("home", {}, expected_s=step.duration_s,
                                         margin_s=HOME_MARGIN_S)
            return self._measure(step, before, result)

        body = {"target_pose": step.detail["flange_pose"], "duration": round(step.duration_s, 3),
                "stop_on_contact": bool(step.detail.get("stop_on_contact")),
                "contact": dict(self.contact) if step.detail.get("stop_on_contact") else {}}
        check = self.client.validate("cartesian", body)
        tries = 0
        while (check.get("valid") is False and tries < SLOWDOWN_TRIES
               and "velocit" in str(check.get("reason", "")).lower()):
            tries += 1
            body = dict(body, duration=round(min(body["duration"] * SLOWDOWN_FACTOR,
                                                 MAX_DURATION_S), 3))
            check = self.client.validate("cartesian", body)
        if check.get("valid") is False:
            return StepResult(step=step, outcome=types.REFUSED,
                              message=str(check.get("reason", "the bridge refused it")))
        result = self.client.run_job("cartesian", body, expected_s=float(body["duration"]),
                                     margin_s=self.margin_s)
        return self._measure(step, before, result)

    def _measure(self, step: Step, before: Pose, result: Dict[str, Any]) -> StepResult:
        after = self.tcp_pose()
        moved = after.position_m - before.position_m
        axis = step.detail.get("axis")
        if axis and step.kind == types.TRANSLATE:
            a = np.asarray(axis, dtype=float)
            along = float(moved @ a)
            lateral = float(np.linalg.norm(moved - along * a))
        else:
            along, lateral = float(np.linalg.norm(moved)), 0.0
        status = result.get("status", "?")
        if status == "complete":
            outcome = types.DONE
        elif status == "stopped" and result.get("contact"):
            outcome = types.CONTACT
        elif status == "stopped":
            outcome = types.LIMITED
        else:
            outcome = types.FAILED
        if outcome != types.DONE:
            self._ready_confirmed = False     # a stopped or failed job parks the arm
        if outcome == types.FAILED and FAULT_WORDS.search(str(result.get("message", ""))):
            # Run 1: 'collision C31', and after it state 2 with error 0 and no collision flag --
            # nothing left for /health to show. Latched here, until the operator clears it.
            self._latched_fault = str(result.get("message", ""))
        # The same two words the simulator's servo loop uses, so one reader serves both.
        abort = ""
        message = str(result.get("message", ""))
        if step.kind == types.TRANSLATE and outcome in (types.LIMITED, types.CONTACT):
            abort = "blocked"
            message = "{}{}moved {:.0f} mm along the axis and {:.0f} mm sideways".format(
                message, " -- " if message else "", along * 1000.0, lateral * 1000.0)
        return StepResult(step=step, outcome=outcome, message=message, abort_reason=abort,
                          moved_m=float(np.linalg.norm(moved)), along_axis_m=along,
                          lateral_m=lateral,
                          turned_rad=rotation_magnitude(after.rotation @ before.rotation.T),
                          pose_after=after, elapsed_s=float(result.get("elapsed_host_s", 0.0)))
