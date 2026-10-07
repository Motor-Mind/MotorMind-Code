"""Turning what the robot knows into the few lines the prompt gets."""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional

import numpy as np

from src.controller import types
from src.controller.types import Capabilities, Pose

from . import geometry


def describe_pose(pose: Pose) -> str:
    """Where the tool is, and which way the jaws point: a model that turned it has no other
    way to know that it did."""
    x, y, z = (v * 1000.0 for v in pose.position_m)
    axis = geometry.approach_axis(pose)
    return "tool at [{:.0f}, {:.0f}, {:.0f}] mm in the base frame; {}".format(x, y, z, (
        "the jaws point straight DOWN" if axis[2] == -1.0 else "the jaws are TILTED {:.0f} deg "
        "from straight down, pointing {}".format(np.degrees(np.arccos(-axis[2])), way(axis))))


def way(vector, directions=None) -> str:
    """The direction word nearest ``vector`` across the table, as the report says it."""
    if directions is None:
        from src.controller.convert import load_directions
        directions = load_directions()
    table = directions["directions"]
    return max((w for w in table if not table[w][2]),
               key=lambda w: float(np.dot(np.asarray(vector)[:2], table[w][:2]))).upper()


def describe_capabilities(caps: Capabilities) -> str:
    lines = [
        "- one step moves at most {:.0f} mm or turns at most {:.1f} deg; longer commands are "
        "split automatically".format(caps.max_translation_m * 1000,
                                     np.degrees(caps.max_rotation_rad)),
        "- speeds are clamped to {:.0f} mm/s and {:.0f} deg/s".format(
            caps.max_speed_m_s * 1000, np.degrees(caps.max_rotation_speed_rad_s)),
    ]
    if caps.supports_contact_stop:
        lines.append("- stop_on_contact exists but is {}".format(
            "trusted" if caps.contact_stop_trusted else
            "NOT trusted on this robot: it fires on the acceleration ramp of every move, so "
            "do not rely on it"))
    else:
        lines.append("- there is no contact stop on this robot")
    if not caps.supports_home:
        lines.append("- there is no home pose")
    if caps.tilt_limits_deg and caps.tilted_floor_m is not None:
        lines.append("- a handle on an UPRIGHT face is taken from the side, jaws tilted level; "
                     "tilted, the tool point gets no lower than {:.0f} mm above the table, so a "
                     "handle well below that cannot be taken".format(caps.tilted_floor_m * 1e3))
    return "\n".join(lines)


def describe_reach(caps: Capabilities, pose: Optional[Pose]) -> str:
    """How far out this arm goes, and how far out the tool is, in one sentence."""
    if caps.reach_m is None or pose is None:
        return ""
    return ("This arm reaches about {:.0f} mm out from its base and the tool point is "
            "{:.0f} mm out now. A move that would end past that is cut back to it."
            .format(caps.reach_m * 1000.0,
                    float(np.hypot(pose.position_m[0], pose.position_m[1])) * 1000.0))


def describe_robot(pose: Pose, clearance: Optional[Dict[str, Any]] = None,
                   grasp: Optional[Dict[str, Any]] = None,
                   extra: Optional[Dict[str, Any]] = None) -> str:
    lines = [describe_pose(pose)]

    if clearance is None:
        lines.append("clearance below the tool: not available on this robot")
    elif clearance.get("mm") is not None and geometry.approach_axis(pose)[2] != -1.0:
        lines.append("clearance ahead of the jaws: {:.0f} mm, measured now".format(clearance["mm"]))
    elif clearance.get("mm") is not None:
        lines.append("clearance below the tool: {:.0f} mm, measured now by the wrist depth "
                     "camera (the surface is at z = {:.0f} mm)".format(
                         clearance["mm"], clearance.get("surface_z_mm") or 0.0))
    elif clearance.get("tracked_mm") is not None:
        surface = clearance.get("tracked_surface_z_mm")
        lines.append("clearance below the tool: about {:.0f} mm above the surface the wrist "
                     "camera LAST SAW{}. NOT measured now -- that surface is inside the "
                     "camera's minimum range, so this is the last direct reading minus how far "
                     "the tool has descended since ({:.0f} mm). If the tool has moved sideways "
                     "since, what is under it now may be a different, higher surface."
                     .format(clearance["tracked_mm"],
                             "" if surface is None else " (at z = {:.0f} mm)".format(surface),
                             clearance.get("descended_mm") or 0.0))
    else:
        lines.append("clearance below the tool: UNKNOWN. The camera cannot see the surface "
                     "under the tool, and there is no earlier reading to carry forward. Do "
                     "not assume there is room.")

    if grasp is None:
        lines.append("gripper: nothing reported")
    else:
        # Two different facts, and a model will run them together if they are not separated.
        holding = grasp.get("holding")
        opening = grasp.get("opening_mm")
        fraction = grasp.get("closed_fraction")
        if fraction is None and opening is None:
            where = "jaw position unknown"
        else:
            if fraction is None:
                word = "closed" if opening < 5 else "open"
            elif fraction < 0.10:
                word = "CLOSED"
            elif fraction > 0.85:
                word = "FULLY OPEN"
            else:
                word = "PARTLY OPEN"
            where = "jaws are {}".format(word)
            if opening is not None:
                where += ", {:.0f} mm apart".format(opening)
            if fraction is not None:
                where += " ({:.0%} of their travel)".format(fraction)
        held = ("holding something" if holding is True
                else "nothing is held between them" if holding is False
                else "whether anything is held is UNKNOWN")
        lines.append("gripper: {}; {}".format(where, held))
        if holding is False:
            lines.append("  (nothing held means no object held the jaws open -- something "
                         "thinner than the pads' closed gap would not register)")
        elif holding is None and grasp.get("reason"):
            lines.append("  ({})".format(grasp["reason"]))

    for key, value in (extra or {}).items():
        lines.append("{}: {}".format(key, value))
    return "\n".join(lines)


def describe_cameras(names: List[str], note: str = "") -> str:
    if not names:
        return "no images are available"
    listed = ", ".join(names)
    return ("{} image(s) follow, in this order: {}{}"
            .format(len(names), listed, ". " + note if note else ""))


def describe_progress(results: List[Dict[str, Any]], commanded_mm: Optional[float]) -> str:
    if not results:
        return "the motion has just started; nothing measured yet"
    last = results[-1]
    lines = ["so far this motion has moved {:.1f} mm ({:.1f} mm along the commanded axis, "
             "{:.1f} mm sideways) and turned {:.1f} deg"
             .format(last.get("moved_mm", 0.0), last.get("along_axis_mm", 0.0),
                     last.get("lateral_mm", 0.0), last.get("turned_deg", 0.0))]
    if commanded_mm:
        lines.append("it was asked for {:.1f} mm in total".format(commanded_mm))
    return "\n".join(lines)


def compact_json(value: Any, limit: int = 1800) -> str:
    text = json.dumps(value, indent=1, sort_keys=False)
    return text if len(text) <= limit else text[:limit] + "\n... (truncated)"


def describe_since_start(origin: Optional[Pose], now: Optional[Pose]) -> str:
    """How far the tool has come since this subgoal began."""
    if origin is None or now is None:
        return ""
    delta = (np.asarray(now.position_m) - np.asarray(origin.position_m)) * 1000.0
    return ("since this subgoal started the tool has moved [{:+.0f}, {:+.0f}, {:+.0f}] mm "
            "(a distance of {:.0f} mm), from [{:.0f}, {:.0f}, {:.0f}]"
            .format(delta[0], delta[1], delta[2], float(np.linalg.norm(delta)),
                    *(np.asarray(origin.position_m) * 1000.0)))


def describe_since_last(previous, now) -> str:
    """What changed between the last observation and this one, in one line."""
    if previous is None or now is None:
        return ""
    before, after = getattr(previous, "pose", None), getattr(now, "pose", None)
    parts = []
    if before is not None and after is not None:
        step = (np.asarray(after.position_m) - np.asarray(before.position_m)) * 1000.0
        if float(np.linalg.norm(step)) >= 1.0:
            parts.append("the tool has moved [{:+.0f}, {:+.0f}, {:+.0f}] mm"
                         .format(step[0], step[1], step[2]))
    was, is_now = getattr(previous, "clearance_mm", None), getattr(now, "clearance_mm", None)
    if was is not None and is_now is not None and abs(float(is_now) - float(was)) >= 1.0:
        parts.append("the clearance under it has gone {:.0f} -> {:.0f} mm"
                     .format(float(was), float(is_now)))
    held_before = (getattr(previous, "grasp", None) or {}).get("holding")
    held_now = (getattr(now, "grasp", None) or {}).get("holding")
    if held_before != held_now:
        parts.append("the jaws went from {} to {}"
                     .format(_held(held_before), _held(held_now)))
    if not parts:
        return ""
    return "since the last look " + ", and ".join(parts) + "."


def _held(value) -> str:
    return "holding something" if value is True else \
        ("holding nothing" if value is False else "not saying what they hold")


def aim_pixel(camera: Dict[str, Any], pose, depth_m: float):
    """Where the spot the jaws point at lands in one camera, as (u, v) pixels."""
    height, width = camera.get("height"), camera.get("width")
    if pose is None or not geometry.usable(camera):
        return None
    target = np.asarray(pose.position_m, dtype=float) \
        + geometry.approach_axis(pose) * float(depth_m)
    spot = geometry.project(camera, target)          # the same projection locate() inverts
    if spot is None or not (0 <= spot[0] < width and 0 <= spot[1] < height):
        return None
    return spot


def aim_depth(pose, clearance: Optional[Dict[str, Any]] = None,
              surface_z_mm: Optional[float] = None, default_depth_m: float = 0.20) -> float:
    """How far below the tool the aim spot is projected: the TABLE when its height is known,
    else the surface the wrist sees now, else the last one it saw, else a guess."""
    level = None if surface_z_mm is None else geometry.to_level(pose, surface_z_mm / 1000.0)
    if level is not None:
        return max(0.02, level)
    if clearance and clearance.get("mm"):
        return float(clearance["mm"]) / 1000.0
    if clearance and clearance.get("tracked_mm"):
        return float(clearance["tracked_mm"]) / 1000.0
    return default_depth_m


def paint_aim_marker(rgb, camera: Dict[str, Any], pose, depth_m: float):
    """Draw the aim spot onto a copy of one camera image, at the same pixel the text names."""
    spot = aim_pixel(camera, pose, depth_m)
    if spot is None or rgb is None:
        return rgb
    try:
        import cv2
    except ImportError:
        return rgb
    image = np.ascontiguousarray(np.asarray(rgb).copy())
    height, width = image.shape[:2]
    u, v = int(round(spot[0])), int(round(spot[1]))
    scale = max(1.0, width / 256.0)
    colour = (255, 0, 220)
    radius = int(round(9 * scale))
    cv2.circle(image, (u, v), radius, colour, max(1, int(round(1.5 * scale))))
    cv2.circle(image, (u, v), max(1, int(round(1.5 * scale))), colour, -1)
    for dx, dy in ((1, 0), (0, 1)):
        cv2.line(image, (u - dx * radius * 2, v - dy * radius * 2),
                 (u - dx * (radius + 2), v - dy * (radius + 2)), colour, max(1, int(scale)))
        cv2.line(image, (u + dx * (radius + 2), v + dy * (radius + 2)),
                 (u + dx * radius * 2, v + dy * radius * 2), colour, max(1, int(scale)))
    return image


def describe_aim_point(cameras: Dict[str, Any], pose, clearance: Optional[Dict[str, Any]] = None,
                       default_depth_m: float = 0.20,
                       surface_z_mm: Optional[float] = None) -> str:
    """Where, in each picture, the spot directly beneath the tool appears."""
    if pose is None or not cameras:
        return ""
    # Project to the TABLE, not to whatever happens to be under the tool.
    depth = aim_depth(pose, clearance, surface_z_mm, default_depth_m)

    target = np.asarray(pose.position_m, dtype=float) + geometry.approach_axis(pose) * depth
    lines = []
    for name, camera in cameras.items():
        height, width = camera.get("height"), camera.get("width")
        spot = aim_pixel(camera, pose, depth)
        if spot is None:
            continue
        u, v = spot

        # How much robot motion a slice of this picture is worth, per axis.
        scale = ""
        parts = []
        for label, offset in (("across", np.asarray([0.0, 0.01, 0.0])),
                              ("down the frame", np.asarray([0.01, 0.0, 0.0]))):
            probe = geometry.project(camera, target + offset)
            if probe is None:
                continue
            moved = np.hypot(probe[0] - u, probe[1] - v)
            span = float(width) if label == "across" else float(height)
            percent = moved / span * 100.0
            if percent > 0.05:
                parts.append("10% {} is roughly {:.0f} mm".format(label, 10.0 / percent * 10.0))
        if parts:
            scale = "; " + ", ".join(parts)
        lines.append("- {}: {:.0f}% across and {:.0f}% down{}".format(
            name, u / float(width) * 100.0, v / float(height) * 100.0, scale))
    if not lines:
        return ""
    return ("WHERE THE TOOL IS AIMING -- the spot directly below the tool, {:.0f} mm down, "
            "appears at:\n{}\nTo put the tool over something, move until that object sits on "
            "this spot. It is NOT the middle of the picture, and aiming at the middle stops "
            "the tool short.".format(depth * 1000.0, "\n".join(lines)))


class TablePlane:
    """The lowest surface the wrist has seen, in base z: a standing estimate of the table."""

    def __init__(self):
        self.z_mm: Optional[float] = None

    def update(self, clearance: Optional[Dict[str, Any]]) -> Optional[float]:
        if clearance and clearance.get("surface_z_mm") is not None and not clearance.get("blind"):
            seen = float(clearance["surface_z_mm"])
            if self.z_mm is None or seen < self.z_mm:
                self.z_mm = seen
        return self.z_mm






class ClearanceTracker:
    """Clearance by dead reckoning once the wrist depth camera can no longer see the surface."""

    # Wider than one sideways step (the model takes 40-50 mm), or a single correction over a rim
    # released the latch and the table was read again.
    def __init__(self, lateral_reset_mm: float = 80.0):
        self.last: Optional[Dict[str, float]] = None
        self.latched: Optional[Dict[str, float]] = None
        self.lateral_reset_mm = lateral_reset_mm

    def reset(self) -> None:
        self.last = None
        self.latched = None

    def update(self, clearance: Optional[Dict[str, Any]], pose) -> Optional[Dict[str, Any]]:
        if clearance is None or pose is None:
            return clearance
        axis = geometry.approach_axis(pose)       # z and xy are along and across the jaws
        if self.last and list(self.last.get("axis", axis)) != list(axis):
            self.reset()                  # a turn: what was ahead of the jaws is not now
        z_mm = -geometry.along(pose.position_m, axis) * 1000.0
        xy = tuple(geometry.across(np.asarray(pose.position_m, dtype=float), axis) * 1000.0)
        if self.latched is not None:
            moved = math.hypot(*(a - b for a, b in zip(xy, self.latched["xy"])))
            if moved > self.lateral_reset_mm:
                self.latched = None
        seen = clearance.get("mm")
        if seen is not None:
            surface = clearance.get("surface_z_mm")
            higher = (self.latched is not None and surface is not None
                      and float(surface) < self.latched["surface"] - 15.0)
            if higher:
                # the surface that was under the tool a moment ago has gone out of range and
                # the camera is now reading past it: carry the higher one
                clearance = dict(clearance)
                clearance["mm"] = None
                clearance["surface_dropped_out"] = True
                self.last = dict(self.latched)
            else:
                self.last = {"mm": float(seen), "z": z_mm, "surface": surface, "xy": xy,
                             "axis": axis}
                if clearance.get("near_min_range") and surface is not None:
                    if self.latched is None or float(surface) >= self.latched["surface"]:
                        self.latched = dict(self.last)
                return clearance
        if self.last:
            descended = self.last["z"] - z_mm
            clearance = dict(clearance)
            clearance["tracked_mm"] = round(self.last["mm"] - descended, 1)
            clearance["tracked_from_mm"] = round(self.last["mm"], 1)
            clearance["descended_mm"] = round(descended, 1)
            clearance["tracked_surface_z_mm"] = self.last.get("surface")
        return clearance


# ------------------------------------------------------------------- what to tell next
#
# The lines the NEXT proposal is written from, out of the cycles behind it. ``Cycle`` is read
# by its fields and never constructed, so nothing here imports the loop.

def _named(found) -> str:
    """What the cameras called the thing they aimed at."""
    label = (getattr(found, "label", "") or "").strip()
    return " (the cameras picked out {!r})".format(label) if label else ""


def _join(preface: str, history: str) -> str:
    """What the caller knows, then what this attempt has measured."""
    return "\n".join(part for part in (preface.strip(), history.strip()) if part)


def _refusals(cycle: Any, limit: int = 220) -> str:
    """The last thing a gate said no to in this cycle, short enough to carry."""
    kept = [str(a.get("error") or "") for a in (cycle.proposal or {}).get("attempts", [])
            if a.get("error")]
    if not kept:
        return ""
    last = kept[-1].strip().replace("\n", " ")
    return last if len(last) <= limit else last[:limit] + "..."


def _history(cycles: List[Any], keep: int = 3) -> str:
    """What has already been tried, so a model can notice it is not converging."""
    lines = []
    for cycle in cycles[-keep:]:
        # What was refused before the motion that ran.
        refused = _refusals(cycle)
        if refused:
            lines.append("- a proposal was REFUSED here: " + refused)
        if cycle.reach.get("text"):
            lines.append("- " + cycle.reach["text"])
        command = ((cycle.proposal or {}).get("proposal") or {}).get("command")
        # Both numbers, because one alone reads as a motion that merely fell short: a descent
        # told "stopped 30 mm short" after sliding twice as far sideways as it went down gets
        # "descend again", which is how a tool walks off the object it was measured over.
        moved = ["{:.0f} mm along{}".format(
            float(s.get("along_axis_mm") or 0.0),
            "" if not s.get("lateral_mm")
            else " and {:.0f} mm SIDEWAYS".format(abs(float(s.get("lateral_mm") or 0.0))))
            for s in cycle.steps if s.get("kind") in (None, "translate")]
        if not moved:
            moved = ["{:.0f} deg".format(float(s.get("turned_deg") or 0.0))
                     for s in cycle.steps if s.get("kind") == "rotate"]
        if not moved:
            # a close, an open, a wait: the distance is nothing and the message is everything
            moved = [str(s.get("message") or "").strip() for s in cycle.steps
                     if str(s.get("message") or "").strip()]
        if command or moved:
            lines.append("- proposed {} -> the robot measured {}".format(
                command if command else "nothing",
                ", then ".join(moved) if moved else "no motion"))
        # Why a motion was SHORTER than the reply asked for.
        for note in dict.fromkeys(str(step.get("note") or "").strip()
                                  for step in cycle.steps):
            if note:
                lines.append("  (" + note + ")")
        if cycle.batch.get("refused"):
            lines.append("  (THE BATCH WAS STOPPED after {} of its {} actions: {})".format(
                cycle.batch.get("stopped_at"), cycle.batch.get("actions"),
                cycle.batch["refused"]))
        elif cycle.stopped_by:
            lines.append("  (THIS MOTION WAS STOPPED PART WAY{}: {})".format(
                " -- watcher's finding: " + cycle.finding if cycle.finding else "",
                cycle.stopped_by))
            if "refused" in cycle.stopped_by:
                lines.append("  (THE ROBOT COULD NOT DO THAT: it is at the limit of its reach "
                             "in that direction. Do not propose it again -- move a different "
                             "way.)")
    return "\n".join(lines)


def edge_depth_mm(caps: Capabilities, pad_fallback_mm: float, own_height_mm: float) -> float:
    """How far below a rim, a wall or a handle the tool point goes before a close is made: a
    WHOLE pad face, so the pads are all the way down over the wall -- capped by what the thing
    has to give, so the pads are never driven into what it stands on. Half a pad under a top
    face the camera reads a little high shuts on air or grips only the lip."""
    pad = getattr(caps, "gripper_pad_length_m", None)
    whole = float(pad) * 1000.0 if pad else float(pad_fallback_mm)
    half = whole / 2.0
    return min(whole, max(float(own_height_mm) - half, 0.0))



def close_miss_mm(point, here, axis, approach=None):
    """A miss decomposed about the line the jaws close along: (across, along), millimetres,
    square to ``approach``."""
    if point is None or here is None or axis is None:
        return None, None
    n = min(len(point), len(here))
    off = geometry.across(np.asarray(point, dtype=float)[:n] - np.asarray(here, dtype=float)[:n],
                          approach) * 1000.0
    axis = np.asarray(axis, dtype=float)[:len(off)]
    if len(off) == 3:
        along = float(off @ axis)
        return float(np.linalg.norm(off - along * axis)), abs(along)
    return abs(float(off[0] * axis[1] - off[1] * axis[0])), abs(float(off @ axis))


def already_there_mm(loop, observation=None) -> Optional[float]:
    """How far the target is, when measured inside the published "already met" bar -- or, for
    a payload over what it goes on (``observation``), the allowance a release there has."""
    tolerance = loop.adapter.capabilities().arrival_tolerance_m
    allowed = loop.release_offset_allowed_mm(observation)[0] \
        if observation is not None and loop._placing(observation) else None
    bar = allowed if allowed is not None else None if tolerance is None else 2000.0 * tolerance
    if bar is None or loop._residual_mm is None or not loop._measured_since_lateral \
            or float(loop._residual_mm) > float(bar):
        return None
    return float(loop._residual_mm)



def _say_the_batch(deltas) -> str:
    """The batch that ran, in a few words, for the watcher asked about it afterwards."""
    said = []
    for delta in deltas or []:
        if delta.kind == types.GRIPPER:
            said.append("gripper {}".format(delta.gripper_state or ""))
        elif delta.kind == types.ROTATE:
            said.append("turn {:.0f} deg".format(np.degrees(float(delta.magnitude))))
        else:
            said.append("move {:.0f} mm{}".format(float(delta.magnitude) * 1000.0,
                                                  " until contact" if delta.push else ""))
    return ", then ".join(said) or "nothing"
