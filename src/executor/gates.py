"""What a reply is told and held to: the measurements the blocks are made of, and the two
limits a reply still meets -- the arm's reach and where a hand may tilt.

``Gates`` is the half of :class:`~src.executor.loop.ExecutorLoop` that measures: the numbers,
what the jaws face, how high things are, what is in them.
"""

from __future__ import annotations


from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

import numpy as np

from src.controller import types
from src.controller.clearance import EMPTY_BAND_FRACTION, EMPTY_BAND_M
from src.controller.convert import fit_to_reach, to_deltas
from src.controller.transforms import axis_angle_to_matrix
from src.controller.types import Pose

from . import geometry
from .context import close_miss_mm, edge_depth_mm, way
from .locate import HELD_UP_BY, _mm
from .models import placed_in, wants_release

@dataclass
class Observation:
    """What the executor gets to look at. Capture time is the camera's, not this machine's."""

    images: List[Any] = field(default_factory=list)
    names: List[str] = field(default_factory=list)
    capture_time: float = 0.0
    robot_text: str = ""
    pose: Optional[Pose] = None
    #: where the tool is aiming in each picture, in words
    legend: str = ""
    #: the surface under the tool, if the wrist depth camera sees it (or dead-reckons it)
    clearance_mm: Optional[float] = None
    #: ...and whether this frame SAW it, rather than carrying the last surface it saw
    clearance_seen: bool = False
    #: what the jaws report as numbers, not prose: the close and holding gates decide on it.
    grasp: Dict[str, Any] = field(default_factory=dict)
    #: the camera payloads -- intrinsic, cam2base, width, height and the DECODED image with no
    #: marker on it -- for the geometry, which measures pixels rather than reading a picture.
    cameras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Cycle:
    """One propose-and-run round, with everything it decided."""

    index: int
    proposal: Dict[str, Any] = field(default_factory=dict)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    supervision: List[Dict[str, Any]] = field(default_factory=list)
    stopped_by: str = ""
    finding: str = ""            # the watcher's finding behind a stop, if any
    #: WHERE THE CAMERAS PUT THE TARGET, and how far the tool is from it.
    reach: Dict[str, Any] = field(default_factory=dict)
    #: what one batch of actions did: how many were proposed, how many ran, what stopped it.
    batch: Dict[str, Any] = field(default_factory=dict)
    #: where this cycle's seconds went -- wall, moving, and model calls.
    timing: Dict[str, Any] = field(default_factory=dict)
    #: what a release measured: how far the jaws opened and why, and how far the payload was
    #: from the middle of what it went on.
    release: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"index": self.index, "proposal": self.proposal, "steps": self.steps,
                "supervision": self.supervision, "stopped_by": self.stopped_by,
                "finding": self.finding, "reach": self.reach, "located": self.reach,
                "batch": self.batch, "timing": self.timing,
                "release": self.release, "error": self.error}


class EpisodeOver(RuntimeError):
    """The simulator has stopped accepting motion, so there is nothing left to drive."""


# ------------------------------------------------------------------- saying it in words

#: ``src/schema/directions.yaml`` says what a direction word means and ``convert._axis_for``
#: resolves every worded action against it, so a word named here is one the runner accepts.
_DIRECTIONS: Optional[Dict[str, Any]] = None

#: ONE CORRECTION, as the gate says -- and what a correction is NOT, since "one action" was
#: quoted back as a licence to nudge a few millimetres instead of closing.
ONE_AT_A_TIME = ("THE TOOL IS IN THE FINE PHASE: there is less than 80 mm of air under it, "
                 "so this reply carries ONE CORRECTION -- which is not one action: up to one "
                 "move along each axis, then the descent, then the jaws, all in this same "
                 "reply. What it may not carry is a second correction along a line it has "
                 "already moved along: that one is decided by the look that comes after.")
#: A put the arm's radius has held off the middle of what it goes on, and that could not let
#: go -- in FROZEN_ENDING's own words first, which is what the mission reads a stuck attempt by.
WALL_ENDING = ("the tool has not moved and the jaws have not changed for {} cycles running: "
               "the arm's reach cut every move nearer where this step wants it short, and "
               "what it holds is not down on what it goes on, or is off the edge of it: come "
               "round to the near side and work from there")
#: ...and a step about WHERE that stopped because it is there, as measured.
ARRIVED_ENDING = ("arrived: nothing has moved for {} cycles running because what has to get "
                  "there is {:.0f} mm from the middle of {}, measured -- inside what counts as "
                  "already there, so the position this step asks for is met")


#: The camera whose boxes cannot be used: it moves with the arm, so a box drawn at plan time
#: is about a view that no longer exists.
WRIST = "wrist"


def _fixed_camera_boxes(boxes: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The planner's boxes, less any drawn on a camera that has moved since."""
    kept = {name: box for name, box in (boxes or {}).items()
            if name != WRIST and box is not None and len(list(box)) >= 4}
    return kept or None


def _tokens(text: str) -> List[str]:
    """Whole words, punctuation and underscores dropped: "yaw_right" is two."""
    return [w for w in ("".join(c if c.isalnum() else " " for c in str(text).lower())).split()
            if w]


def _unit(vector) -> Optional[np.ndarray]:
    """A unit vector, or None if the thing is not three finite numbers with a length."""
    try:
        axis = np.asarray(vector, dtype=float).reshape(3)
    except Exception:
        return None
    norm = float(np.linalg.norm(axis))
    return None if not np.isfinite(norm) or norm < 1e-9 else axis / norm


def _direction_table(directions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The ``directions:`` table, loaded once if the caller has none to hand."""
    global _DIRECTIONS
    if directions is not None:
        return directions.get("directions") or {}
    if _DIRECTIONS is None:
        from src.controller.convert import load_directions
        _DIRECTIONS = load_directions()
    return _DIRECTIONS.get("directions") or {}


def _rotation_table(directions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Every rotation word as the signed axis it means, exactly as ``convert._axis_for``
    folds the sign in, so one vector identifies one direction of turn."""
    out = {}
    for word, entry in ((directions or {}).get("rotations") or {}).items():
        axis = _unit(np.asarray(entry["axis"], dtype=float) * float(entry.get("sign", 1)))
        if axis is not None:
            out[word] = axis
    return out


def _word_for(axis, table: Dict[str, Any], sign: float = 1.0) -> Optional[str]:
    """The word in ``table`` whose vector is ``axis`` (``sign=-1``: the opposite of it)."""
    unit = _unit(axis)
    if unit is None:
        return None
    for word, vector in table.items():
        other = _unit(vector)
        if other is not None and sign * float(np.dot(unit, other)) > 0.99:
            return word
    return None


def _say_offset(offset_mm, directions: Optional[Dict[str, Any]] = None) -> str:
    """A base-frame offset in the robot's own words: "120 mm LEFT and 40 mm FORWARD"."""
    table = _direction_table(directions)
    said, offset = [], np.asarray(offset_mm, dtype=float).reshape(-1)
    for index in range(min(3, offset.shape[0])):
        size = float(offset[index])
        if abs(size) < 1.0:
            continue
        axis = np.zeros(3)
        axis[index] = 1.0 if size > 0 else -1.0
        word = _word_for(axis, table)
        said.append("{:.0f} mm {}".format(abs(size),
                                          (word or "along axis {}".format(index)).upper()))
    return " and ".join(said) if said else "where it already is"


# ---------------------------------------------------------------------- the gate chain


def chain_gates(*rules):
    """One gate from several: the first refusal wins."""
    def gate(proposal, deltas, observation):
        for rule in rules:
            if rule is None:
                continue
            refused = rule(proposal, deltas, observation)
            if refused:
                return refused
        return None
    return gate


# ------------------------------------------------------------------ reading a motion


def _closes(deltas) -> bool:
    """Does this batch shut the jaws?"""
    return any(d.kind == types.GRIPPER and d.gripper_state == "close" for d in deltas)


def _is_descent(deltas, axis=None) -> bool:
    """A motion along the jaws -- straight down, pointing down -- however the model spelled it:
    ``axis: [0, 0, -1]`` and ``direction: "down"`` are the same motion to the robot."""
    return any(d.kind == types.TRANSLATE and d.axis is not None
               and geometry.along(d.axis, axis) > 0.9 for d in deltas or [])


# ------------------------------------------------------------------- letting go


def when_the_jaws_work(loop, deltas, observation, state: str):
    """(the observation as it will be when this reply first works the jaws to ``state``, its
    own moves having run by then; the shift they make) -- or (``observation``, None) when
    they move the tool nowhere worth saying. A move that stops on contact goes an unknown way,
    except a descent before an open: it ends with what the jaws hold down on what it met."""
    shift = np.zeros(3)
    for delta in deltas or []:
        if delta.kind == types.GRIPPER and delta.gripper_state == state:
            break
        if delta.kind == types.TRANSLATE and delta.axis is not None and (not delta.push or (
                state == "open" and geometry.along(delta.axis, loop._approach(observation))
                > 0.9)):
            shift = shift + np.asarray(delta.axis, dtype=float) * float(delta.magnitude)
    pose = getattr(observation, "pose", None)
    if pose is None or float(np.linalg.norm(shift)) * 1000.0 < loop.NO_MOTION_MM:
        return observation, None
    return replace(observation, pose=pose.moved(shift),
                   clearance_mm=None if observation.clearance_mm is None
                   else float(observation.clearance_mm)
                   - geometry.along(shift, loop._approach(observation)) * 1000.0), shift


def where_the_release_would_land(loop, observation, height: bool = True) -> str:
    """Why opening the jaws HERE would not put the thing where the step wants it, or "" --
    judged against the allowance at the reach wall too (release_offset_allowed_mm); without
    ``height``, only how far off the middle of what it goes on (or in) it is."""
    # ...a part held from the side is a part of something standing: nothing drops
    if loop.gripper_only or not loop._placing(observation) or loop._level(observation):
        return ""
    clearance = observation.clearance_mm if height else None
    if clearance is not None and not loop._stopped_along_the_jaws(observation) \
            and clearance > loop.RELEASE_CLEARANCE_MM:
        # Nothing measures how far below the tool the payload hangs, and the tool is well up.
        return "it is {:.0f} mm of air under the tool and nothing measured how far below " \
               "the tool point, or to which side, what you are holding hangs".format(clearance)
    allowed, _ = loop.release_offset_allowed_mm(observation)
    offset = loop.release_offset_mm(observation)
    if offset is None or allowed is None or offset <= allowed:
        return ""
    return "it is {:.0f} mm from the middle of what it goes {}, and a release wants that " \
           "under {:.0f} mm".format(offset, "in" if loop.placing_in() else "on", allowed)


def release_gate(loop, proposal, deltas, observation) -> Optional[str]:
    """An open the model proposes, held to where the thing in the jaws would land: over the
    middle of what it goes on or in (fix7's release_gate, its offset half). How high it hangs
    is not asked: a release still goes by touch."""
    if not any(d.kind == types.GRIPPER and d.gripper_state == "open" for d in deltas or []):
        return None
    off = where_the_release_would_land(
        loop, when_the_jaws_work(loop, deltas, observation, "open")[0], height=False)
    if not off:
        return None
    return ("Opening the jaws there does not put what you are holding where this step wants "
            "it: {}. Letting go is the one action that cannot be taken back -- what falls out "
            "of the jaws goes where it falls. Carry it over the middle of what it goes on "
            "first: the offsets under WHERE THE TARGET IS are measured and name the direction. "
            "Open the jaws in the same reply if you like, and they will be judged from where "
            "the moves end.".format(off))


# ------------------------------------------------- a held thing turned against its stop

#: A held turn that got under this share of what it asked for, stopped, met the end of travel.
ROTATION_STOPPED_SHARE = 0.5


def note_turn(loop, result, cycle, observation) -> None:
    """Remember a lone turn made holding something: how far it went, and if it met a stop."""
    deltas = getattr(result, "deltas", None) or []
    if len(deltas) != 1 or deltas[0].kind != types.ROTATE or cycle.finding \
            or not loop._placing(observation):
        return
    axis, asked = _unit(deltas[0].axis), float(np.degrees(float(deltas[0].magnitude)))
    turned = sum(abs(float(s.get("turned_deg") or 0.0)) for s in cycle.steps
                 if s.get("kind") == "rotate")
    last = cycle.steps[-1] if cycle.steps else {}
    if axis is not None and asked > 0.0:
        loop._turns.append({"axis": axis, "asked": asked, "turned": turned, "stop": (
            turned < ROTATION_STOPPED_SHARE * asked and (
                last.get("outcome") == types.LIMITED or last.get("abort_reason") == "blocked"))})


def turn_gate(loop, proposal, deltas, observation) -> Optional[str]:
    """No turn again INTO a stop a held thing met in this subgoal: that is the end of its
    travel that way (fix7's rotation_gate; qwen-c3 L10: knobs turned on, then turned back)."""
    table = _rotation_table(getattr(loop.proposer, "directions", None))
    for delta in deltas:
        axis = _unit(delta.axis) if delta.kind == types.ROTATE and delta.axis is not None \
            else None
        stop = None if axis is None else next(
            (t for t in loop._turns if t["stop"] and float(np.dot(axis, t["axis"])) > 0.95), None)
        if stop is None:
            continue
        word = (_word_for(axis, table) or "that way").replace("_", " ")
        said = ("{} is against a stop: it was asked for {:.0f} degrees and the robot measured "
                "{:.0f} while holding it.".format(word, stop["asked"], stop["turned"]))
        back = sum(t["turned"] for t in loop._turns
                   if not t["stop"] and float(np.dot(axis, t["axis"])) < -0.95)
        if back:
            return ("{} The other way has already turned it {:.0f} degrees in this subgoal, so "
                    "there is no travel left either way. If that is what the subgoal wanted, "
                    "say done; if not, the thing in the jaws is not the part that turns -- let "
                    "go and take hold of the part that does.".format(said, back))
        return ("{} A thing that will not move that way is at the end of its travel that way, "
                "whatever the next command asks. If the subgoal wanted it where it already is, "
                "say done; otherwise turn it {} instead.".format(
                    said, (_word_for(axis, table, sign=-1.0) or "the other way").replace("_", " ")))
    return None


# ------------------------------------------------- the payload on its way out of the jaws


def jaw_resolution_mm(loop) -> float:
    """The narrowest thing these jaws can tell from air, in millimetres."""
    span = getattr(loop.adapter.capabilities(), "gripper_open_span_m", None)
    band = EMPTY_BAND_M * 1000.0
    return band if not span else min(band, EMPTY_BAND_FRACTION * float(span) * 1000.0)


def _watching(loop, observation) -> Optional[float]:
    """The gap the close settled at on the thing the jaws are carrying NOW, or ``None``."""
    if loop.gripper_only or loop._jaws_shut(observation) is not True:
        # The jaws are open: that width has been let go of, but is kept to judge the release.
        loop._watched_payload_mm = None
        loop._stale_payload_mm = loop._payload_width_mm
        return None
    took = loop._payload_width_mm
    if took is not None and took != loop._stale_payload_mm:
        # A width that is not the stale one is a fresh close: the jaws have something again.
        loop._watched_payload_mm = float(took)
    return loop._watched_payload_mm


# --------------------------------------------- the first motion after a wall grip is a test


def _a_wall_grip(loop, took: float) -> bool:
    """Is what the jaws have a WALL -- a rim, a lip, a handle -- rather than a body filling
    them?"""
    return bool(loop._aimed_at_edge or loop._payload_at_edge) or took < loop.pad_mm()


def _gently(loop, delta) -> float:
    """What speed a test segment runs at: never faster than the speed this robot uses when a
    command names none."""
    default = getattr(loop.adapter.capabilities(), "default_speed_m_s", None)
    speed = float(delta.speed or 0.0)
    if not default:
        return speed
    return float(default) if not speed else min(speed, float(default))


TEST_REST = "(the rest)"


def first_move_is_a_test(loop, proposal, deltas, observation) -> None:
    """Cut the first motion of a reply carrying a wall grip in two, in place: a short lead,
    and the rest of it behind that, which runs only if the grip held over the lead."""
    if loop._running or loop.gripper_only:
        return None                  # mid-batch the actions are already one at a time
    took = _watching(loop, observation)
    if took is None or not _a_wall_grip(loop, took):
        return None
    lead = loop.pad_mm() / 1000.0
    for index, delta in enumerate(deltas or []):
        if delta.kind != types.TRANSLATE or delta.axis is None or delta.push:
            continue
        if float(delta.magnitude) <= lead * 2.0:
            return None              # already short enough to be its own test
        said = ("the first {:.0f} mm of that move is an action of its own: the jaws are "
                "holding a wall at {:.1f} mm, and what that gap does over those first few "
                "millimetres decides whether the rest of the move runs at all"
                .format(lead * 1000.0, took))
        deltas[index:index + 1] = [
            replace(delta, magnitude=lead, speed=_gently(loop, delta), note=said,
                    label="{} (the first {:.0f} mm)".format(delta.label, lead * 1000.0)),
            replace(delta, magnitude=float(delta.magnitude) - lead,
                    label="{} {}".format(delta.label, TEST_REST))]
        for place, each in enumerate(deltas):
            each.index = place
        return None
    return None


def the_test_held(loop, proposal, deltas, observation) -> Optional[str]:
    """The rest of a test move runs only if the jaws still hold what they took."""
    if not any(TEST_REST in str(d.label or "") for d in deltas or []):
        return None
    took = _watching(loop, observation)
    grasp = getattr(observation, "grasp", None) or {}
    gap = grasp.get("opening_mm")
    if took is None:
        return None
    if grasp.get("holding") is False:
        leaving = "the robot now reports them holding nothing"
    elif gap is not None and float(gap) <= jaw_resolution_mm(loop):
        leaving = "they read {:.1f} mm now, which is as far as they shut on nothing".format(
            float(gap))
    else:
        return None
    return ("Stop the motion: the jaws took hold at {:.1f} mm and {}. The test move before "
            "this one is what showed it: the thing slid out from between the pads, so the rest "
            "of the move would carry air. Treat the jaws as empty: open them, look for the "
            "thing again, and take hold of it before anything is carried anywhere."
            .format(took, leaving))


# ------------------------------------------------- a long way sideways starts with a lift

#: a move sideways longer than this from a low tool shoves things (shorter ones are the rim
#: alignments, a wall hold's aim being 41-53 mm off the middle).
TRAVERSE_MM = 60.0
#: how far above what it would hit the tool, or what hangs in it, must already be for a long
#: move sideways to be left alone: the 60 mm the shoves stop at.
CLEAR_MM = 60.0
#: how far above it the tool is taken instead: the bottom of the band nothing was shoved in,
#: and no higher, because every mm up is one the next cycle comes back down.
TRAVEL_MM = 100.0


def _names_a_sliding_part(loop) -> bool:
    """Does this step name something that SLIDES, read up to where the words say it is: "the
    bowl in the top drawer" is a bowl."""
    said = _tokens((loop._target or "").strip() or "{} {}".format(loop._subgoal or "",
                                                               loop._criterion or ""))
    named = next((said[:at] for at, word in enumerate(said) if word in HELD_UP_BY), said)
    return bool(set(named) & set(loop.SLIDING_PARTS))


def lift_before_sideways(loop, deltas, observation):
    """(the reply, what was put into it): a long way sideways from a tool down among the things
    on the table bulldozes what is in the way, so a lift goes in before the first move
    sideways -- and, for a reply that works the jaws after its moves sideways, the same
    distance down after the last of them, so the jaws work where it meant (fix7's mend)."""
    caps = loop.adapter.capabilities()
    pose = getattr(observation, "pose", None)
    # ...and none with the jaws turned level: what they point at is not below them
    if pose is None or caps.table_z_m is None or loop.gripper_only or loop._level(observation):
        return deltas, ""
    held = loop._placing(observation)
    if not held and _names_a_sliding_part(loop) or step_says(loop, loop.LEANING_WORDS):
        return deltas, ""            # a push, a pull, a slide: meant to be down there
    # How far the reply's own ups and downs take the tool before its first move sideways: a
    # lift it already asked for is not asked for twice, and one it undoes with a descent first
    # is no lift at all -- so the lift goes in right before that first move, not at the top.
    sideways, before, first, last = 0.0, 0.0, None, None
    for at, delta in enumerate(deltas):
        if delta.kind != types.TRANSLATE or delta.axis is None or delta.push:
            continue
        axis = np.asarray(delta.axis, dtype=float)
        if abs(float(axis[2])) < 0.9:
            sideways += float(delta.magnitude) * float(np.linalg.norm(axis[:2])) * 1000.0
            first = at if first is None else first
            last = at
        elif first is None:
            before += float(delta.magnitude) * float(axis[2]) * 1000.0
    jaws = [i for i, d in enumerate(deltas) if d.kind == types.GRIPPER]
    # What the jaws hold, having met something -- a motion cut short, by touch -- is dragged
    # across it by ANY move sideways.
    resting = held and loop._payload_met
    if sideways <= (0.0 if resting else TRAVERSE_MM) or any(i < last for i in jaws):
        return deltas, ""            # jaws worked before or between the moves sideways
    # What it would hit: the thing this step is about as measured, and with nothing measured
    # whatever stands up to the height the approach came down from; never lower than where
    # that thing was located. While holding, only the receiver: the thing may be the payload,
    # and an opened drawer carried into was shut by the carry (qwen-c3 Goal, five cells).
    under = 0.0 if held else loop._object_height_mm(observation)
    if under is None and loop._origin is not None:
        under = (float(loop._origin.position_m[2]) - float(caps.table_z_m)) * 1000.0 - TRAVEL_MM
    point = (loop._last_point or {}).get("point") if not held or (
        loop._last_point or {}).get("words") == (loop._sensing(observation)[0] or "").strip(
            ).lower() else None
    if point is not None:
        under = max(under or 0.0, (float(point[2]) - float(caps.table_z_m)) * 1000.0)
    # ...and what a held payload met stands as high as the payload is: it rests at that height.
    height = 0.0 if resting else (float(pose.position_m[2]) - float(caps.table_z_m)) * 1000.0 \
        - max(0.0, under or 0.0)
    if height + before >= CLEAR_MM:
        return deltas, ""
    rise = TRAVEL_MM - height - before
    said = ("a {:.0f} mm lift was put in front of the {:.0f} mm sideways in that reply{}: {} "
            "would be {:.0f} mm above what it would sweep into, and a move that long that low "
            "pushes things across the table instead of passing over them".format(
                rise, sideways, ", and the tool is brought back down {:.0f} mm after it so the "
                "jaws work where the reply meant".format(rise) if jaws else "",
                "the tool", height + before))
    # ...each its own command: a lift past what one command may go is cut to it, and the way
    # back then did not fit at all (rigport2 193/0, step2AB 175/0 raised on it)
    lift, back = (to_deltas({"actions": [{"type": "move", "distance_mm": rise, "note": said,
                                           "axis": [0.0, 0.0, way]}]}, caps,
                            loop.proposer.directions)[0] for way in (1.0, -1.0))
    lift, back = replace(lift, label="lift over the table"), \
        replace(back, label="back down from over the table")
    deltas = list(deltas[:first]) + [lift] + list(deltas[first:last + 1]) \
        + ([back] if jaws else []) + list(deltas[last + 1:])
    return [replace(d, index=i) for i, d in enumerate(deltas)], said


def split_a_carry(loop, deltas, observation):
    """Cut a long CARRY into the legs the robot is going to run anyway, so that the gates
    between one action and the next are asked between them too."""
    most = float(getattr(loop.adapter.capabilities(), "max_translation_m", 0.0) or 0.0)
    if most <= 0.0 or _watching(loop, observation) is None:
        return deltas
    out, axis = [], loop._approach(observation)
    for delta in deltas or []:
        far = float(delta.magnitude)
        carrying = (delta.kind == types.TRANSLATE and delta.axis is not None
                    and not delta.push and geometry.along(delta.axis, axis) < 0.9
                    and far > most)
        if not carrying:
            out.append(replace(delta, index=len(out)))
            continue
        legs = int(np.ceil(far / most - 1e-9))
        for leg in range(legs):
            out.append(replace(delta, index=len(out), magnitude=min(most, far - leg * most),
                               label="{} ({} of {})".format(delta.label or "move",
                                                            leg + 1, legs)))
    return out


# ------------------------------------------------- the open the model proposes


def step_says(loop, words) -> bool:
    """Does this step's sentence or criterion use any of ``words``?"""
    return bool(set(_tokens("{} {}".format(loop._subgoal or "", loop._criterion or "")))
                & set(words))


# ------------------------------------------------------ a part taken from the side

#: What a side take says when the robot measured that it cannot be done -- which the verdict
#: and the replan read as such (planner/mission.py, planner/planner.py).
CANNOT_FROM_THE_SIDE = "cannot be taken from the side"
#: How far off vertical a turn has to get before the jaws point INTO an upright face: 45-60
#: degrees meets the face with the finger edges (measured on LIBERO's drawer scenes). The
#: robot's own Capabilities.tilt_reach_deg wins; this is the fallback where it declares none.
TILT_REACHES_DEG = 80.0


def face_out(part, body, directions: Dict[str, Any]) -> Optional[np.ndarray]:
    """Which way the upright face a part stands out of faces: of the robot's own words for
    across the table, the one pointing most from the middle of the thing it belongs to toward
    the part -- None with no body, or a part at its middle."""
    if body is None:
        return None
    away = _unit(np.append((np.asarray(part, dtype=float) - np.asarray(body, dtype=float))[:2],
                           0.0))
    level = [unit for unit in (_unit(vector) for vector in directions.values())
             if unit is not None and abs(float(unit[2])) < 0.5]
    return None if away is None or not level else max(level, key=lambda unit: float(unit @ away))


def side_take(loop, observation) -> Optional[Dict[str, Any]]:
    """For a step that takes a part FROM THE SIDE: which way the face it stands out of faces,
    the turn that points the jaws into it and the stand-off in front of it -- or why it cannot
    be done. The face is measured once a subgoal; the stand-off follows the located point."""
    caps, point = loop.adapter.capabilities(), (loop._last_point or {}).get("point")
    if getattr(observation, "pose", None) is None or point is None or caps.table_z_m is None \
            or not caps.tilt_limits_deg or not loop._side_step():
        return None
    point, axis = np.asarray(point, dtype=float), loop._approach(observation)
    if not loop._last_point.get("on_a_face"):     # one view: where it stands, not its height
        return {"cannot": "{} {}: nothing located where on its face the part is".format(
            loop._target or "it", CANNOT_FROM_THE_SIDE)}
    if loop._face_said is None:
        out = face_out(point, loop._last_point.get("body"),
                       _direction_table(getattr(loop.proposer, "directions", None)))
        table = _rotation_table(getattr(loop.proposer, "directions", None))
        # the turns this arm makes in full, by how squarely each points the jaws into that face
        turns = sorted((float(out @ axis_angle_to_matrix(table[word], np.pi / 2.0) @ axis), word)
                       for word, limit in caps.tilt_limits_deg.items()
                       if out is not None and word in table and limit >= loop.tilt_reach_deg())
        loop._face_said = {"out": out, "word": turns[0][1] if turns and turns[0][0] < -0.7
                           else None}
    side = dict(loop._face_said)
    out, low = side["out"], (point[2] - caps.table_z_m) * 1000.0
    floor = (caps.tilted_floor_m or 0.0) * 1000.0
    side["turned"] = out is not None and float(out @ axis) < -0.7
    # The jaws close UP-DOWN: from the floor a part up to a pad inside the half-gap below is
    # still between them (gpt6sol-goal-fix3 100: a handle read at 92-94 mm, truly 106-122).
    why = ("no two views outlined the thing it belongs to, so which way its face faces is not "
           "known") if out is None else (
        "it is {:.0f} mm above the table, and tilted this arm's tool point gets no lower than "
        "{:.0f} mm".format(low, floor) if low < floor - loop.jaw_half_gap() + loop.pad_mm()
        else "no turn this arm makes in full points the jaws into it"
        if side["word"] is None and not side["turned"] else "")
    if why:
        side["cannot"] = "{} {}: {}".format(loop._target or "it", CANNOT_FROM_THE_SIDE, why)
    else:
        side["stand"] = point + out * loop.SLIDE_STANDOFF_MM / 1000.0
        side["stand"][2] = max(float(side["stand"][2]), caps.table_z_m + floor / 1000.0)
    return side


def turn_back(pose, directions) -> str:
    """The turn that points tilted jaws straight down again, as the pose measures it --
    'rotate "roll_ccw" 90' -- or "" when no one rotation word makes it."""
    axis, table = np.asarray(pose.rotation, dtype=float)[:, 2], _rotation_table(directions)
    about = _unit(np.cross(axis, types.DOWN))
    word = None if about is None else max(table, key=lambda w: float(table[w] @ about))
    return "" if word is None or float(table[word] @ about) < 0.9 else 'rotate "{}" {:.0f}'.format(
        word, np.degrees(np.arccos(np.clip(-axis[2], -1.0, 1.0))))


def side_block(loop, observation) -> str:
    """TAKING IT FROM THE SIDE: the stand-off, the turn and the move in -- or, holding it with
    the jaws tilted, which way the pull goes; let go of, the way back to pointing down."""
    words, axis = getattr(loop.proposer, "directions", None), loop._approach(observation)
    if loop._placing(observation) and loop._level(observation):
        return ("THE PULL: the jaws point {} into the face of what they hold, so it opens with a "
                "pull {}, away from its body, along its line -- no lift, no turn -- which stops "
                "by itself where it stops giving; then open the jaws.".format(
                    way(axis, words), way(-axis, words)))
    if axis[2] != -1.0 and not loop._side_step():
        return "THE JAWS ARE TILTED, pointing {}: {}{} brings them back to pointing down.".format(
            way(axis, words), "back straight off {} first, clear of what is in front of them, "
            "then ".format(way(-axis, words)) if loop._level(observation) else "",
            turn_back(observation.pose, words) or "the turn that tilted them, undone,")
    side = side_take(loop, observation)
    if not side or side.get("cannot"):
        return "TAKING IT FROM THE SIDE: {}.".format(side["cannot"]) if side else ""
    into, out = way(-side["out"], words), way(side["out"], words)
    turn = "" if side["turned"] else (
        ", going across at this height and turning the jaws there, in open air -- rotate \"{}\" "
        "90 points them {}, into the face, closing UP-DOWN -- before coming down"
        .format(side["word"], into))
    return ("TAKING IT FROM THE SIDE: {} stands out of an upright face that faces {}. Come to "
            "the STAND-OFF point [{:.0f}, {:.0f}, {:.0f}] mm, {:.0f} mm in front of it and level "
            "with it -- {} of the tool point{}. From there move {} by {:.0f} mm with "
            "\"stop_on_contact\": true and close in that same reply: the move stops against the "
            "face with the part between the pads. The pull that opens it, {}, is a step of its "
            "own.".format(loop._target or "it", out, *(side["stand"] * 1000.0),
                          loop.SLIDE_STANDOFF_MM, _say_offset(
                              (side["stand"] - observation.pose.position_m) * 1000.0, words),
                          turn, into, loop.SLIDE_STANDOFF_MM + 2.0 * loop.pad_mm(), out))


class Gates:
    """The numbers a reply is judged against, and the gates that judge it."""
    _payload_across_mm: Optional[float] = None       # the held thing's outline width
    # ------------------------------------------------------------------ the numbers
    # Each one is a measurement; the line above each says what it is FOR.

    #: how much further out than it starts a move may end and still be inside the arm's radius:
    #: an arc across the line is no reach past it.
    REACH_TANGENT_M = 0.005
    #: how near an earlier empty close is "the same spot"
    EMPTY_CLOSE_RADIUS_M = 0.008
    #: and how far up the tool must come before it may close or descend again.
    RECOVERY_UP_M = 0.10
    #: a "height" under this is bare table or the clearance reading's own noise (~2.5 mm
    #: tracked, 3-5 mm of wrist extrinsic); over it, not the thing under the tool at all.
    MIN_OBJECT_MM = 10.0
    MAX_OBJECT_MM = 300.0
    #: used when nobody has measured this robot's jaws (Capabilities.gripper_open_span_m)
    DEFAULT_JAW_HALF_GAP_MM = 40.0
    #: what the pads need beside the object for the close to be a grip rather than a graze
    PAD_MARGIN_MM = 5.0
    #: the smallest sideways error worth holding a close to: one fixed view back-projected
    #: onto the table is worth 25-40 mm whatever anyone does.
    MIN_CLOSE_TOLERANCE_MM = 20.0
    #: a measured width under this is the outline seen end-on, not an object
    MIN_WIDTH_MM = 8.0
    #: and one wider than this many jaw-openings is not one either
    IMPLAUSIBLE_WIDTH_SPANS = 3.0
    #: how much a measured width may change between two looks before neither can be believed.
    WIDTH_JUMP_SHARE = 0.5
    #: how deep below its own top face the tool must be to close, as a share of its height.
    GRASP_BELOW_SHARE = 0.25
    #: and the smallest descent worth writing into a refusal
    #: how far short of the depth a close wants a close is let through -- the clearance
    #: reading's own noise, without which the gate has a boundary a descent cannot land on.
    LEVEL_SLACK_MM = 1.5
    MIN_USEFUL_DESCENT_MM = 3.0
    #: words for a feature standing up out of an object rather than the object's body.
    EDGE_PARTS = ("rim", "brim", "lip", "edge", "handle", "wall", "bar", "loop", "ear",
                  "flange", "ridge")
    #: under this much sideways travel a descent has not been shoved, it has leaned
    SKID_MM = 3.0
    #: under these a motion did nothing a camera or a measurement could see
    NO_MOTION_MM = 1.0
    NO_TURN_DEG = 0.5
    #: a motion that made less than this share of what it was told to met something
    BLOCKED_SHARE = 0.25
    #: how many of the surfaces seen over the outline the top face is the median of: enough
    #: that nothing a take's looks saw is dropped, few enough that a carried list stays small.
    FACES_KEPT = 12
    #: the thinnest a close can settle at and still be round some part of a thing this wide, as a
    #: share of its width, capped at what a thin wall settles at.
    THINNEST_SHARE = 0.04
    THINNEST_CAP_MM = 5.0
    #: how many corners an outline needs before its middle is worth taking.
    CENTROID_NEEDS_CORNERS = 3
    #: how far off the middle of the receiving thing the PAYLOAD may be when the jaws let go,
    #: as a share of that thing's own footprint radius.
    ON_OFFSET_SHARE = 0.25
    IN_OFFSET_SHARE = 0.5
    #: how much air under the TOOL is a drop, not a placement.
    RELEASE_CLEARANCE_MM = 30.0
    #: how much wider than the payload the jaws open to let it go.
    RELEASE_MARGIN_MM = 8.0
    #: how long the finger pads are: half of it is how far INSIDE a rim an edge grasp aims, and
    #: a whole one further in on a WHOLE outline's width, which reads long against a part's box.
    #: The robot's Capabilities.gripper_pad_length_m wins (pad_mm); this is the fallback.
    JAW_PAD_MM = 12.0
    OUTLINE_EDGE_PAD_MM = 12.0
    #: below this much air the approach is fine work: ONE action at a time, and the wrist is
    #: close enough to be worth a model call of its own.
    FINE_PHASE_MM = 80.0
    #: cycles of nothing moving that end the subgoal -- the STALL rule, what ends one getting
    #: nowhere under a mission clock -- and a safety cap that is no longer a budget.
    FROZEN_CYCLES, MAX_CYCLES = 2, 30
    #: words that make a criterion a question about WHERE something is, not what happened to
    #: it -- a step met by standing somewhere is met by the pose the tool is in.
    POSITION_CRITERION = ("over", "above", "under", "beneath", "aligned", "align", "centred",
                          "centered", "lined", "in line", "directly")
    #: words that say the step MEANS to let go from a height, so the clearance is the point of
    #: the motion and not a fault in it...
    DROP_WORDS = ("drop", "dropping", "tip", "tipping", "pour", "pouring", "fall")
    #: ...and the rest of the ways a step says it is about letting the thing GO
    LETTING_GO_WORDS = ("release", "let go", "place", "placing", "put it down", "put down",
                        "set down", "deposit", "into the", "onto the")
    #: the kind of thing that moves along its own track: a gripper can only lean on it
    SLIDING_PARTS = ("drawer", "door", "lid", "panel", "slider", "tray", "cover", "hatch",
                     "shutter")
    #: words that make a step LEAN on the thing rather than take it: its aim is a face's middle,
    #: not a grasp's edge. Not "shut": a shut thing is a SLIDING_PART, and shut jaws are a take.
    LEANING_WORDS = ("push", "pushing", "pushed", "press", "pressing", "pressed", "lean",
                     "leaning", "shove", "shoving", "slide", "sliding", "nudge")
    #: how far in front of a sliding face to stand before leaning on it.
    SLIDE_STANDOFF_MM = 90.0
    #: how high above the table a face is met, at most: LOW, because anything standing on the
    #: table has something to lean on down there whatever its height.
    #: the words that make a step a take FROM THE SIDE (planner.yaml's step of that name)
    SIDE_WORDS = "from the side"

    def _side_step(self) -> bool:
        return self.SIDE_WORDS in (self._subgoal or "").lower()

    def _level(self, observation=None) -> bool:
        """Are the jaws turned level, into an upright face, as a side take turns them?"""
        return -float(self._approach(observation)[2]) <= np.cos(
            np.radians(self.tilt_reach_deg()))

    # ---------------------------------------------------------------- what the jaws face

    def jaw_half_gap(self) -> float:
        """Half the open jaws, in millimetres: the widest sideways miss a close can survive."""
        span = getattr(self.adapter.capabilities(), "gripper_open_span_m", None)
        return self.DEFAULT_JAW_HALF_GAP_MM if not span else float(span) * 1000.0 / 2.0

    def held_in_view(self, observation, camera) -> Optional[List[float]]:
        """The box, in ``camera``'s pixels, round what the jaws hold -- its measured width
        across the jaws and as deep, at the tool point -- or None when nothing is held."""
        pose, axis = getattr(observation, "pose", None), self.jaw_axis(observation)
        across = self._payload_across_mm or self._payload_width_mm
        if camera is None or pose is None or axis is None or not self._placing(observation) \
                or not across:
            return None
        half, tool = across / 2000.0, np.asarray(pose.position_m, dtype=float)
        axis = np.append(axis, 0.0) if len(axis) == 2 else axis      # straight down: across is 2-D
        other = np.cross(axis, geometry.approach_axis(pose))
        seen = [geometry.project(camera, tool + sign * half * way)
                for way in (axis, other) for sign in (-1.0, 1.0)]
        if any(point is None for point in seen):
            return None
        uv = np.asarray(seen, dtype=float)
        return [uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()]

    def jaw_axis(self, observation) -> Optional[Any]:
        """The line the jaws close along in the base frame: across the table, or tilted, 3-D."""
        return self._jaw_axis_of(None if observation is None else observation.pose)

    @staticmethod
    def _jaw_axis_of(pose) -> Optional[Any]:
        if pose is None:
            return None
        axis = np.asarray(pose.rotation, dtype=float) @ np.array([0.0, 1.0, 0.0])
        flat = geometry.across(axis, geometry.approach_axis(pose))
        norm = float(np.linalg.norm(flat))
        return None if norm < 1e-6 else flat / norm

    def dragged(self, step, delta, observation) -> bool:
        """Was a move ACROSS the jaws stopped against something with a payload in them? The next
        one that way drags it out of the pads (gpt6sol 102, 109: two blocked legs, then lost), so
        the batch ends there."""
        return step.get("abort_reason") == "blocked" and delta.axis is not None \
            and self._placing(observation) and not self._level(observation) \
            and abs(geometry.along(delta.axis, self._approach(observation))) < 0.9

    def _approach(self, observation=None):
        """Where the jaws in ``observation`` point: straight down when nothing says."""
        return geometry.approach_axis(getattr(observation, "pose", None))

    def _miss(self, point, observation):
        """What is left between what has to arrive and ``point``, square to the approach axis:
        across the table, with the jaws pointing down."""
        here = self._here(observation, whole=True)
        if point is None or here is None:
            return None
        n = min(len(point), len(here))
        return geometry.across(np.asarray(point, dtype=float)[:n] - here[:n],
                               self._approach(observation))

    def _stopped_along_the_jaws(self, observation=None) -> bool:
        """Was the last motion along the jaws -- a descent -- stopped dead?"""
        return self._blocked_axis is not None \
            and geometry.along(self._blocked_axis, self._approach(observation)) > 0.9

    def widest_plausible_mm(self) -> float:
        """Past this a width is not a thing on the table: it is a box drawn round several of
        them, or a back-projection that landed on the far wall."""
        return min(self.MAX_OBJECT_MM, self.IMPLAUSIBLE_WIDTH_SPANS * self.jaw_half_gap() * 2.0)

    def _width_words(self) -> str:
        """Which target the remembered width belongs to: one object, one width."""
        return (self._last_point.get("words") or self._target or "").strip().lower()

    def target_width(self, observation) -> Any:
        """(millimetres across the jaws, where that number came from) -- or (None, why not)."""
        axis = self.jaw_axis(observation)
        across = geometry.extent_along(self._footprint, axis)
        span = geometry.longest_across(self._footprint)
        if across is None and span is None:
            return None, ""
        # Only an outline from straight above can say a thing is wider than the jaws.
        if len(self._footprint) < 4 and span > 2.0 * (self.jaw_half_gap() - self.PAD_MARGIN_MM):
            return None, "a side view's box read {:.0f} mm, which reads wide".format(span)
        last = self._last_width
        why = ""
        if across is not None and across < self.MIN_WIDTH_MM:
            why = ("measured only {:.0f} mm across the jaws, which is the outline seen "
                   "end-on rather than an object".format(across))
        elif across is not None and across > self.widest_plausible_mm():
            why = ("measured {:.0f} mm across the jaws, wider than anything these jaws could "
                   "be working near".format(across))
        elif across is not None and last and last.get("words") == self._width_words() \
                and last.get("mm"):
            if abs(across - last["mm"]) / last["mm"] > self.WIDTH_JUMP_SHARE:
                why = ("measured {:.0f} mm across the jaws, against {:.0f} mm for the same "
                       "thing a moment ago -- one object does not change width, so neither "
                       "reading can be believed".format(across, last["mm"]))
        how = ", from {}".format(self._footprint_method) if self._footprint_method else ""
        if not why and across is not None:
            # ...and the most it is PROVEN to measure: a reading less its outline's error bar
            least = across - float(self._footprint_error_mm or 0.0)
            if last and last.get("words") == self._width_words():
                least = max(least, float(last.get("least", least)))
            self._last_width = {"words": self._width_words(), "mm": across, "least": least}
            return across, "measured across the line the jaws close along" + how
        if span is None:
            return None, why
        said = "the outline's longest way across"
        return span, ((said + ", because the measurement across the jaws " + why)
                      if why else said) + how

    def wider_than_the_jaws(self, observation=None) -> bool:
        """Is the thing PROVABLY wider than the jaws open -- by some reading of it less its own
        error? An outline reads a solid wide from above and a bowl narrow from close up."""
        self.target_width(observation)
        return float((self._last_width or {}).get("least") or 0.0) > self.jaw_half_gap() * 2.0

    def _names_an_edge(self, *said: str) -> bool:
        """Do these words name a feature standing up out of an object (EDGE_PARTS)?"""
        words = _tokens(" ".join(said))
        return any(part in words for part in self.EDGE_PARTS)

    def _a_grasping_step(self) -> bool:
        """Will the jaws close on the thing this step is about?"""
        if self._wants_holding:
            return True
        return self._names_an_edge(self._target or "", self._last_point.get("words") or "")

    def _an_approach_to_take(self) -> bool:
        """Is this step only getting the tool somewhere over a thing the NEXT step will take
        hold of -- an `over_the_bowl` before a `take_the_bowl`? Aimed at the thing's middle,
        such a step puts the tool where a close on a thing wider than the jaws can never be
        made from, so it is aimed at the edge too. Excluded: a step that LEANS on the thing --
        a push, a press, a slide -- which wants the middle of a face and takes nothing."""
        if self._wants_holding:
            return False
        return not step_says(self, self.LEANING_WORDS)

    def edge_offset_mm(self, observation, grasping: Optional[bool] = None) -> Any:
        """How far OFF the middle of the thing a grasp of it has to be aimed, and why."""
        if self._placing(observation):
            return None, ""        # carrying it TO something: the middle is where it goes
        if not (self._a_grasping_step() if grasping is None else grasping) \
                and not self._an_approach_to_take():
            # A step that leans on the thing -- a push, a press -- wants its middle.
            return None, ""
        width, source = self.target_width(observation)
        if width is None or not self.wider_than_the_jaws(observation):
            return None, ""
        span = self.jaw_half_gap() * 2.0
        inward = self.pad_mm() / 2.0 + (0.0 if self._footprint_is_part
                                          else self.OUTLINE_EDGE_PAD_MM)
        return (max(0.0, width / 2.0 - inward),
                "it is about {:.0f} mm across ({}) and the jaws open {:.0f} mm, so no close "
                "centred on it can get round it -- the jaws have to straddle its wall{}"
                .format(width, source or "measured from the located outline", span,
                        "" if self._footprint_is_part
                        else ", aimed a pad further in because that width is the whole "
                             "thing's outline and not a named part's own box"))

    def edge_point(self, centre, observation):
        """The point on the footprint's edge to put the tool over, or None."""
        offset_mm, _ = self.edge_offset_mm(observation)
        axis = self.jaw_axis(observation)
        pose = None if observation is None else observation.pose
        if not offset_mm or axis is None or pose is None or centre is None:
            return None
        axis = np.asarray(axis, dtype=float)                  # 3-D, with the jaws tilted
        here = np.asarray(pose.position_m, dtype=float)[:len(axis)]
        middle = np.asarray(centre, dtype=float)[:len(axis)]
        side = 1.0 if float(np.dot(here - middle, axis)) >= 0.0 else -1.0
        # The side the tool is already on, so nothing crosses the thing to get there -- unless
        # that side lies outside the arm's radius and the other does not: things near the edge
        # of the reach are only taken by the rim nearer the base.
        reach = getattr(self.adapter.capabilities(), "reach_m", None)
        if reach is not None:
            radius = {sign: float(np.linalg.norm(
                (middle + sign * float(offset_mm) / 1000.0 * axis)[:2])) for sign in (1.0, -1.0)}
            if radius[side] > float(reach) >= radius[-side]:
                side = -side
        point = np.asarray(centre, dtype=float).copy()
        point[:len(axis)] = middle + side * float(offset_mm) / 1000.0 * axis
        return point

    def quarter_turn_for_reach(self, observation) -> str:
        """'rotate "yaw_left" 90' when neither rim along the line the jaws close along is inside
        the arm's reach and, turned a quarter about the vertical, the nearer one is -- or ""
        (gpt6sol-full-fix4 task 6: a bowl 800 mm out, both rims 804 mm out, 21 blocked moves)."""
        reach = getattr(self.adapter.capabilities(), "reach_m", None)
        offset_mm, _ = self.edge_offset_mm(observation)
        axis, point = self.jaw_axis(observation), (self._last_point or {}).get("point")
        if reach is None or not offset_mm or axis is None or point is None:
            return ""
        middle, along = np.asarray(point, dtype=float)[:2], np.asarray(axis, dtype=float)[:2]
        if float(np.linalg.norm(along)) < 0.5:              # tilted to close along the vertical
            return ""
        along = along / float(np.linalg.norm(along))
        nearer = lambda line: min(float(np.linalg.norm(  # noqa: E731
            middle + sign * float(offset_mm) / 1000.0 * line)) for sign in (1.0, -1.0))
        if nearer(along) <= float(reach) or nearer(np.array([-along[1], along[0]])) > reach:
            return ""
        word = _word_for(-types.DOWN, _rotation_table(
            getattr(self.proposer, "directions", None)))
        return 'rotate "{}" 90'.format(word) if word else ""


    def close_tolerance(self, observation) -> float:
        """How far off centre a close may be, in millimetres."""
        half_gap = self.jaw_half_gap()
        width, _ = self.target_width(observation)
        if width is None:
            return half_gap
        if self.wider_than_the_jaws(observation):
            # Round a thin wall, the bound is how well anything can aim.
            return self.MIN_CLOSE_TOLERANCE_MM
        room = half_gap - width / 2.0 - self.PAD_MARGIN_MM
        return min(half_gap, max(room, self.MIN_CLOSE_TOLERANCE_MM))

    def edge_grasp(self, observation=None) -> bool:
        """Is what this subgoal takes hold of a FEATURE rather than an object's body?"""
        return self._names_an_edge(self._target or "", self._last_point.get("words") or "",
                                   self._subgoal or "") or self.wider_than_the_jaws(observation)

    def grasp_depth_mm(self, height: float, observation=None) -> float:
        """How far below the measured top face the tool must be before a close is made."""
        if not self.edge_grasp(observation):
            return self.GRASP_BELOW_SHARE * height
        return edge_depth_mm(self.adapter.capabilities(), self.JAW_PAD_MM, height)


    # ------------------------------------------------------------- how high up things are


    def _note_the_face(self, observation) -> None:
        """How far the target's top face stands above the table, from what the wrist saw. A
        stopped descent sets none: what it rests on may be under the pads or under the hand."""
        table_z = self.adapter.capabilities().table_z_m
        pose = None if observation is None else observation.pose
        point = self._last_point.get("point")
        if self._top_face_mm is None and point is not None and pose is not None \
                and table_z is not None and not self._placing(observation) \
                and not self._side_step():
            # ...until it does, where the top edges of the two fixed views' boxes meet: a top
            # narrower than the patch the wrist reads under the jaws (a bottle's neck) reads as
            # the table, and gpt6sol 102 closed on nothing with no height to go by
            top = self._last_point.get("top")
            rise = None if top is None else (float(top[2]) - float(table_z)) * 1000.0
            if rise is not None and self.MIN_OBJECT_MM <= rise <= self.MAX_OBJECT_MM:
                self._top_face_mm = rise
                self._top_face_from = "where the top edges of the fixed views' boxes meet"
        # A face UNDER the tool, along the jaws: tilted, they do not point at one.
        if geometry.to_level(pose, table_z) is None \
                or not geometry.over_the_outline(self._footprint, pose.position_m,
                                                 self.jaw_half_gap() / 1000.0):
            return
        # Only a frame the camera SAW something in: a descent onto the thing is a run of blind
        # frames answering with the surface carried from the last one that saw it, and counting
        # those would make it the median whatever the frames from above had read.
        if not observation.clearance_seen or observation.clearance_mm is None:
            return
        face = (float(pose.position_m[2]) - self.adapter.capabilities().control_point_offset_m
                - float(table_z)) * 1000.0 - float(observation.clearance_mm)
        if not self.MIN_OBJECT_MM <= face <= self.MAX_OBJECT_MM:
            return
        # The MEDIAN of the surfaces seen over the outline, not the highest: the highest is the
        # one frame that caught a rim edge-on or a neighbour, and it puts the close too high.
        self._faces_seen_mm = (list(self._faces_seen_mm or []) + [face])[-self.FACES_KEPT:]
        self._top_face_mm = float(np.median(self._faces_seen_mm))
        self._top_face_from = "measured by the wrist camera while the tool was over it"

    def _above_the_face_mm(self, observation) -> Optional[float]:
        """How far the tool point is above the target's TOP FACE. Negative is below it."""
        face = self._top_face_mm
        clearance = None if observation is None else observation.clearance_mm
        up = geometry.to_level(getattr(observation, "pose", None),
                               self.adapter.capabilities().table_z_m)
        if face is None or up is None:
            return clearance
        return up * 1000.0 - face

    def _object_height_mm(self, observation) -> Optional[float]:
        """How tall the thing under the tool is, out of two numbers the robot already has."""
        if self._top_face_mm is not None:
            return self._top_face_mm                   # frozen: see _note_the_face
        clearance = None if observation is None else observation.clearance_mm
        up = geometry.to_level(getattr(observation, "pose", None),
                               self.adapter.capabilities().table_z_m)
        if clearance is None or up is None:
            return None
        height = up * 1000.0 - float(clearance)
        return None if height < self.MIN_OBJECT_MM or height > self.MAX_OBJECT_MM else height

    def _far_above(self, observation) -> bool:
        """Are the jaws shutting too far above whatever is under them to be taking it?"""
        above = self._above_the_face_mm(observation)
        height = self._object_height_mm(observation)
        if above is None or height is None:
            return False
        return above > max(height, self.jaw_half_gap())

    def where_the_tool_is(self, observation=None) -> str:
        """Where the tool point is relative to the top face under it, in one sentence."""
        above = self._above_the_face_mm(observation)
        height = self._object_height_mm(observation)
        # A number that was never measured is said to be missing, not left out: with no
        # commissioned surface every height here is unanswerable, and a close is refused.
        uncommissioned = ("" if self.adapter.capabilities().table_z_m is not None else
                          " This robot has no commissioned surface height, so nothing here is "
                          "measured against the table and a close will be refused until "
                          "something measures what is under the jaws.")
        if above is None:
            return ("Nothing can measure what is under the tool, so there is no top face to "
                    "count from." + uncommissioned)
        if height is None:
            return ("There is {:.0f} mm between the tool point and whatever is under it, and "
                    "nothing has measured how tall that is, so this is a clearance and not a "
                    "depth.{}".format(above, uncommissioned))
        need = self.grasp_depth_mm(height, observation)
        where = ("{:.0f} mm ABOVE its top face".format(above) if above > 0 else
                 "{:.0f} mm BELOW its top face".format(-above))
        return ("The thing this step is about stands {:.0f} mm above the table{} and the "
                "tool point is {}; a close on it wants the tool point at least {:.0f} mm "
                "below that face{}.".format(height,
                                      "" if not self._top_face_from
                                      else " (" + self._top_face_from + ")",
                                      where, need,
                                      ", which is a whole pad face: what is being taken hold "
                                      "of is an edge, and the pads have to be all the way "
                                      "down over it"
                                      if self.edge_grasp(observation) else ""))

    # ------------------------------------------------------------- what is IN the jaws

    def _widest_mm(self) -> Optional[float]:
        """The last width measured across the jaws, or the outline's longest way across."""
        return (self._last_width or {}).get("mm") or geometry.longest_across(self._footprint)

    def _too_thin_to_be_it(self, outcome) -> bool:
        """Did the jaws settle at a gap no part of the thing this step names could hold open?"""
        gap = (outcome.detail or {}).get("opening_mm")
        width = self._widest_mm()
        if gap is None or width is None:
            return False
        return float(gap) < min(self.THINNEST_CAP_MM, self.THINNEST_SHARE * float(width))

    def _note_step(self, outcome, pose_before, grasping: bool = True) -> None:
        """Bookkeeping from what the robot measured: what the jaws did, and where -- and a
        descent stopped dead, which spends the depth asked for earlier and owes a close look."""
        if outcome is None:
            return
        if outcome.pose_after is not None:
            self._pose_after = outcome.pose_after
        goal = outcome.step.goal
        if outcome.abort_reason == "blocked" and goal is not None and pose_before is not None \
                and (float(pose_before.position_m[2]) - float(goal.position_m[2])) * 1000.0 \
                > self.NO_MOTION_MM:
            self._looked_close = False
        if outcome.step.kind != types.GRIPPER or outcome.step.detail.get("state") != "close":
            return
        message = outcome.message or ""
        # The robot's own words for its own jaws.
        self._closed_holding = True if message.startswith("holding something") else \
            (False if "nothing held" in message else None)
        if self._closed_holding is True and self._too_thin_to_be_it(outcome):
            self._closed_holding, message = False, "nothing held"
        if self._closed_holding is True:
            self._measure_the_payload(outcome, pose_before)
        if grasping and ("nothing held" in message or self._closed_holding is False):
            where = outcome.pose_after.position_m if outcome.pose_after is not None \
                else (pose_before.position_m if pose_before is not None else None)
            if where is not None:
                self._empty_closes.append(np.asarray(where, dtype=float).copy())
                self._unrecovered_z = float(where[2])

    def _measure_the_payload(self, outcome, pose_before) -> None:
        """Everything about the thing in the jaws, taken at the moment they took it."""
        # A new thing in the jaws is a new placement: the last receiver would judge it wrong.
        self._receiver = {}
        pose = outcome.pose_after if outcome.pose_after is not None else pose_before
        gap_mm = (outcome.detail or {}).get("opening_mm")
        # Every close that holds writes its gap down and clears the spent mark: two grips of one
        # thing measure the same number, and a width equal to the spent one is watched.
        self._payload_width_mm = None if gap_mm is None else float(gap_mm)
        # ...and how wide the thing itself is, which a wall grip's gap is not: its outline as
        # measured when it was aimed at (a bowl held by the rim settles the pads 10-21 mm apart).
        last = self._last_width or {}
        self._payload_across_mm = max(float(gap_mm or 0.0), float(last.get("mm") or 0.0)) \
            if last.get("words") == self._width_words() else self._payload_width_mm
        self._stale_payload_mm = None
        # Where the thing hangs relative to the tool, along the line the jaws close along. A body
        # NARROWER than the jaws is squeezed to the tool's own middle by the close, so its
        # offset is zero whatever the cameras said (the difference is the locate error). A
        # WALL grip leaves the thing's middle where the fixed views' outline put it, off the
        # tool point -- not where a look from over the wall itself put the wall.
        offset, wall = 0.0, self._a_wall_hold(gap_mm)
        middle = geometry.outline_centroid(self._footprint, 2)
        if middle is None:
            middle = self._last_point.get("point")
        if wall and pose is not None and middle is not None:
            axis = self._jaw_axis_of(pose)
            middle = np.asarray(middle, dtype=float)[:2]
            here = np.asarray(pose.position_m, dtype=float)[:2]
            if axis is not None:
                offset = float(np.dot(middle - here, np.asarray(axis, dtype=float)[:2])) * 1000.0
            # A middle further from the tool than the thing's own half width, plus the error a
            # location carries, is a point the thing was shoved away from before the close:
            # where it hangs is then unknown, which is not under the tool.
            if abs(offset) > self._half_width_mm() + self.MIN_CLOSE_TOLERANCE_MM:
                offset = None
            elif self.wider_than_the_jaws():
                # Of a thing PROVEN wider than the jaws the middle is that half width in from where
                # the jaws aim, on the outline's side: a close-up outline reads a bowl narrow.
                inset = self.pad_mm() / 2.0 \
                    + self.OUTLINE_EDGE_PAD_MM * (not self._footprint_is_part)
                offset = float(np.copysign(self._last_width["least"] / 2.0 - inset, offset))
        self._payload_offset_mm = offset
        self._payload_at_edge = bool(self._aimed_at_edge or (wall and offset != 0.0))
        # ...and the thing has left the place it was located at: nothing is kept, or anchors a
        # look, on where it was.
        self._measured_since_lateral, self._last_point = False, {}

    def _half_width_mm(self) -> float:
        """Half of the widest way across the thing this step names, from the last measurement
        of it, or half the jaws' opening when nothing has measured it."""
        width = self._widest_mm()
        return (float(width) if width else self.jaw_half_gap() * 2.0) / 2.0

    def _a_wall_hold(self, gap_mm) -> bool:
        """Did the jaws take a WALL -- a rim, a lip -- and not a body that fills them? Any of:
        the aim was the edge; the jaws settled under one pad, which only a thin feature leaves
        them; or the thing is provably wider than the jaws open."""
        return bool(self._aimed_at_edge) or gap_mm is not None \
            and float(gap_mm) < self.pad_mm() or self.wider_than_the_jaws()

    def _note_what_the_payload_met(self, cycle, observation) -> None:
        """With the jaws full, a motion cut short met something with what they hold: kept, and
        across subgoals, until a cycle has risen off it."""
        if not self._placing(observation):
            return
        steps = cycle.steps or []
        rose = any(float((step.get("axis") or [0.0, 0.0, 0.0])[2]) > 0.9 and float(
            step.get("along_axis_mm") or 0.0) > self.NO_MOTION_MM for step in steps)
        self._payload_met = any(step.get("axis") and step.get("outcome") != types.DONE
                                for step in steps) or self._payload_met and not rose

    def _placing(self, observation) -> bool:
        """Is this step carrying something TO the thing named, rather than going to take it?"""
        return (getattr(observation, "grasp", None) or {}).get("holding") is True

    def payload_point(self, observation) -> Optional[Any]:
        """Where the middle of what is being carried is: xy, metres, base frame -- None when
        nothing measured where it hangs from the tool."""
        pose = None if observation is None else observation.pose
        if pose is None or self._payload_offset_mm is None:
            return None
        here = np.asarray(pose.position_m, dtype=float)[:2].copy()
        axis = self.jaw_axis(observation)
        if not self._payload_offset_mm or axis is None:
            return here
        return here + float(self._payload_offset_mm) / 1000.0 * np.asarray(axis, dtype=float)[:2]

    def placing_in(self) -> bool:
        """Does this step put the payload IN something, or ON it?"""
        return bool(placed_in(self._criterion or "", self._subgoal or ""))

    def release_offset_allowed_mm(self, observation=None) -> Any:
        """(how far off the receiver's middle a release may be, why), or (None, ""). At the
        reach wall, with the middle of what it goes ON past the arm's radius, its own radius."""
        radius = self._receiver.get("radius_mm")
        if not radius:
            return None, ""
        inside = self.placing_in()
        payload, reach = self.payload_point(observation), self.adapter.capabilities().reach_m
        if self._at_the_wall and not inside and payload is not None and reach is not None \
                and float(np.linalg.norm(np.asarray(observation.pose.position_m)[:2] + np.asarray(
                    self._receiver["centroid"])[:2] - payload)) > float(reach):
            return float(radius), "its middle is past this arm's reach, so anywhere on it will do"
        # IN, the payload's own near edge has to clear the rim, and not only its middle.
        allowed = float(radius) * self.IN_OFFSET_SHARE - float(self._payload_width_mm or 0.0) \
            / 2.0 if inside else float(radius) * self.ON_OFFSET_SHARE
        return max(self.MIN_CLOSE_TOLERANCE_MM, allowed), (
            "it measures {:.0f} mm across, and putting something {} it wants what you are "
            "holding {} of that".format(float(radius) * 2.0, "in" if inside else "on",
                                        "all of it inside the rim" if inside
                                        else "over the middle, not merely somewhere on it"))

    def release_offset_mm(self, observation) -> Optional[float]:
        """How far the PAYLOAD is from the middle of what it is being put down on."""
        centroid = self._receiver.get("centroid")
        payload = self.payload_point(observation)
        if centroid is None or payload is None:
            return None
        return float(np.linalg.norm(np.asarray(payload, dtype=float)[:2]
                                    - np.asarray(centroid, dtype=float)[:2])) * 1000.0

    def where_the_payload_is(self, observation) -> str:
        """Where the thing in the jaws is, in one sentence, or "" when they are empty."""
        if not self._placing(observation):
            return ""
        # How far below the tool it hangs is unknown: nothing reads what it stood on, so it is
        # let go of by touch.
        said = ["The jaws are holding something and nothing has measured how far below the "
                "tool point it hangs, so the air under IT is unknown: the clearance above is "
                "the tool's own."]
        offset = self.release_offset_mm(observation)
        allowed, why = self.release_offset_allowed_mm()
        if offset is not None and allowed is not None:
            said.append("It is {:.0f} mm from the middle of {} ({}), and a release wants that "
                        "under {:.0f} mm -- {}."
                        .format(offset, self._receiver.get("words") or "what it goes on",
                                self._receiver.get("from") or "measured", allowed, why))
        return " ".join(said)

    def _about_letting_go(self) -> bool:
        """Is this step about the thing in the jaws LEAVING them?"""
        words = "{} {}".format(self._subgoal or "", self._criterion or "").lower()
        if not words.strip():
            return True
        if wants_release(self._criterion) or wants_release(self._subgoal):
            return True
        if any(word in words for word in self.DROP_WORDS + self.LETTING_GO_WORDS):
            return True
        return "open" in words and any(word in words for word in
                                       ("jaw", "gripper", "finger", "grip"))

    def letting_go(self, delta, observation) -> bool:
        """Is this action an open of jaws that hold what the step lets go of?"""
        if delta.kind != types.GRIPPER or delta.gripper_state != "open":
            return False
        # The robot's own hold flag flickers, so a letting-go step whose payload is still
        # watched counts too: the width is still in the jaws.
        if self._placing(observation) or (self._about_letting_go()
                                          and _watching(self, observation) is not None):
            if self._payload_width_mm is not None:   # open only far enough to stop squeezing
                delta.open_to_mm = min(self.jaw_half_gap() * 2.0, float(
                    self._payload_width_mm) + self.RELEASE_MARGIN_MM)
            return True
        return False

    def note_the_release(self, delta, observation, cycle: Cycle) -> None:
        """Write down what a release was aimed at, measured where the jaws opened -- the moves
        in front of it having run -- what they opened to, and the allowance applied there."""
        allowed, _ = self.release_offset_allowed_mm(observation)
        offset = self.release_offset_mm(observation)
        centroid = self._receiver.get("centroid")
        cycle.release = dict(cycle.release, **{
            "asked_mm": _mm(delta.open_to_mm),
            "opened_to_mm": ((cycle.steps[-1] if cycle.steps else {}).get("detail")
                             or {}).get("opening_mm"),
            "payload_width_mm": _mm(self._payload_width_mm),
            "payload_offset_mm": _mm(self._payload_offset_mm),
            "tool_clearance_mm": _mm(observation.clearance_mm),
            # ...and a placement is measured no finer than the receiver it was measured against
            "offset_mm": None if allowed is not None and float(
                self._receiver.get("error_mm") or float("inf")) >= allowed else _mm(offset),
            "allowed_mm": _mm(allowed),
            "receiver": self._receiver.get("words") or "",
            "receiver_centroid_mm": None if centroid is None
            else [round(float(v) * 1000.0, 1) for v in np.asarray(centroid)[:2]],
            "placing_in": self.placing_in()})
        if self._at_the_wall:
            # held to the receiver's radius and not its middle: see release_offset_allowed_mm
            cycle.release["because"] = "at the reach wall"

    # ------------------------------------------------------------------ the gates
    # Each takes (proposal, deltas, observation) and returns a refusal or None.


    def _jaws_shut(self, observation) -> Optional[bool]:
        """Have the jaws stopped where a close left them? None when the robot cannot say."""
        grasp = getattr(observation, "grasp", None) or {}
        if grasp.get("holding") is True:
            return True
        fraction = grasp.get("closed_fraction")
        if fraction is not None:
            return float(fraction) < 0.5
        opening = grasp.get("opening_mm")
        if opening is not None:
            return float(opening) < 20.0
        return None


    def empty_pickup_gate(self, proposal, deltas, observation) -> Optional[str]:
        """After a close that shut on air: up 100 mm before closing or descending again."""
        closing, descending = _closes(deltas), _is_descent(deltas)
        here_z = None if observation is None or observation.pose is None \
            else float(observation.pose.position_m[2])
        if self._unrecovered_z is not None and here_z is not None \
                and here_z >= self._unrecovered_z + self.RECOVERY_UP_M - 0.005:
            self._unrecovered_z = None                    # the tool is back up: recovered
        if self._unrecovered_z is not None and (closing or descending):
            still = self.RECOVERY_UP_M * 1000.0 - (0.0 if here_z is None
                                                   else (here_z - self._unrecovered_z) * 1000.0)
            return ("The last close shut on nothing, and the tool is still down where it "
                    "happened. Recover before you {}: open the jaws if they are shut, then go "
                    "UP -- the tool has to be 100 mm above where that close happened ({:.0f} "
                    "mm still to go) so the object and the marker are back in the wrist view "
                    "-- re-align from up there, and only then come down and close again. A "
                    "close or a descent from here is refused until then."
                    .format("close" if closing else "descend", max(0.0, still)))
        if closing and observation is not None and observation.pose is not None:
            here = np.asarray(observation.pose.position_m, dtype=float)
            spot = self._near(here, self._empty_closes)
            if spot is None:
                return None
            if not any(float(np.linalg.norm(np.asarray(seen) - spot)) < 1e-9
                       for seen in self._relooked):
                # A look costs what the refusal costs and can MOVE the answer, where the
                # refusal only asks. The close follows it.
                self._relook_at = np.asarray(spot, dtype=float).copy()
                return None
            return ("A close at this very spot already shut on nothing ({:.0f} mm from "
                    "where the last empty close happened), and the look taken since did "
                    "not move the tool off it. Closing here again will shut on nothing "
                    "again: change the position or the height first, and say what you "
                    "changed.".format(float(np.linalg.norm(here - spot)) * 1000.0))
        return None

    def _near(self, here, spots):
        """The first of ``spots`` within an empty close's radius of ``here``, or None."""
        return next((spot for spot in spots
                     if float(np.linalg.norm(here - spot)) < self.EMPTY_CLOSE_RADIUS_M), None)

    def descent_floor_mm(self, height: float, observation=None) -> float:
        """How far below the top face a descent may go before it is doing harm, not good:
        twice the depth a close wants, so the recipe has a band and not a point -- and, for
        an edge, never further than the pads can go before they meet what the thing stands
        on -- the pad bottom being half a pad under the tool point."""
        if self.edge_grasp(observation):
            depth = self.grasp_depth_mm(height, observation)
            return min(2.0 * depth, max(depth, height - self.pad_mm() / 2.0))
        return height / 2.0

    def _as_it_will_be(self, deltas, observation):
        """Where the tool will BE when the close in this reply runs, and how far it comes down
        on the way."""
        at, shift = when_the_jaws_work(self, deltas, observation, "close")
        if shift is None:
            return observation, 0.0
        return at, geometry.along(shift, self._approach(at)) * 1000.0

    def pad_mm(self) -> float:
        """How long the pads are, as the robot declares it; JAW_PAD_MM where it does not."""
        pad = getattr(self.adapter.capabilities(), "gripper_pad_length_m", None)
        return float(pad) * 1000.0 if pad else self.JAW_PAD_MM

    def tilt_reach_deg(self) -> float:
        """How far round from straight down a tilted take reaches, as the robot declares it."""
        return float(getattr(self.adapter.capabilities(), "tilt_reach_deg", None)
                     or TILT_REACHES_DEG)

    def _hand_rests(self, observation) -> bool:
        """Did a stopped descent rest on the HAND, not the pads? The thing fits centred between
        the open pads -- half its width plus the miss along their line inside half the gap --
        so a pad cannot be on it; a thing wider than the jaws (a bowl's rim) still has one."""
        width = self.target_width(observation)[0]
        along = close_miss_mm(self._target_point(observation), self._here(observation, True),
                              self.jaw_axis(observation), self._approach(observation))[1]
        return None not in (width, along) and width / 2.0 + along < self.jaw_half_gap()

    def _depth_slack(self) -> float:
        """How far above the depth a close wants the close is let through anyway: the
        clearance reading's own noise, and in front of the jaws this arm's arrival
        tolerance too, the descent the gate judged having run."""
        if not self._running:
            return self.LEVEL_SLACK_MM
        arrival = self.adapter.capabilities().arrival_tolerance_m
        return self.LEVEL_SLACK_MM + (float(arrival) * 1000.0 if arrival
                                      else self.LEVEL_SLACK_MM)

    def close_depth_gate(self, proposal, deltas, observation) -> Optional[str]:
        """No close until the tool has come down to the depth a close wants, judged where the
        moves in front of it will have put the tool (fix7's residual_gate, its depth half)."""
        if not _closes(deltas):
            return None
        observation, descent_mm = self._as_it_will_be(deltas, observation)
        if self._far_above(observation) and not self._a_grasping_step():
            return None               # a fist, not a grasp: see _far_above
        above = self._above_the_face_mm(observation)
        height = self._object_height_mm(observation)
        # A descent stopped dead short of that depth has the pads resting ON the thing.
        stopped = self._stopped_along_the_jaws(observation)
        if height is None or above is None:
            # Nothing MEASURED under the tool: admitted with a look owed over it first, and
            # refused if the look taken there still measured nothing -- unless the descent the
            # refusal asks for has already stopped dead.
            pose, caps = getattr(observation, "pose", None), self.adapter.capabilities()
            if pose is None or stopped or not self._a_grasping_step():
                return None
            # ...a look DOWN, which tilted jaws have none of: they are held to the stop.
            level = caps.table_z_m is None or geometry.to_level(pose, caps.table_z_m)
            if (not self._running or self._relook_at is not None) and level is not None:
                if self._relook_at is None:
                    self._relook_at = np.asarray(pose.position_m, dtype=float).copy()
                return None
            if level is None:                # held to a move in along them that stops dead
                return None if not self._running and any(d.stop_on_contact and _is_descent(
                    [d], self._approach(observation)) for d in deltas) else (
                    "The jaws are tilted, so nothing measures how deep they are round what "
                    "they close on. Move in along them with \"stop_on_contact\": true until "
                    "the move stops dead, and close in that same reply.")
            if caps.table_z_m is None:
                return ("Nothing here has measured what is under the jaws: this robot was "
                        "never commissioned against a surface, so there is no height to count "
                        "a depth from and nothing can tell this close from one that shuts on "
                        "air. Come down ONTO the thing until the descent stops dead, and "
                        "close from there.")
            return ("Nothing has measured the top face of what this close is for -- not the "
                    "wrist on the way in, nor the look just taken -- so there is no depth to "
                    "judge it against, and a close judged against nothing shuts on air. Come "
                    "down ONTO it until the descent stops dead, then close.")
        need = self.grasp_depth_mm(height, observation)
        if above <= -need + self._depth_slack() or stopped and self._hand_rests(observation):
            return None
        if stopped:
            # A stop within a pad of the depth on a thing wider than the jaws is the pads
            # straddling its wall, and one refused here already did not move off it: said
            # twice at one spot, this refusal is a loop, not a correction (L10: 472 of them).
            here = np.asarray(observation.pose.position_m, dtype=float)
            if above + need <= self.pad_mm() and self.wider_than_the_jaws(observation) \
                    or self._near(here, self._stopped_at) is not None:
                return None
            self._stopped_at.append(here)
            return ("The descent was stopped dead {:.0f} mm short of the depth a close wants, "
                    "and what is under it is not known to fit centred between the pads: a pad "
                    "rests on it and a close shuts on nothing. Come up off it: the wrist looks "
                    "again from there, and WHERE THE TARGET IS says where to come down."
                    .format(above + need))
        # It ASKS for the far side of the band as a TOTAL from where the tool is: the reply
        # is a NEW one, and 11 of 13 sent an increment back whole.
        aim = self.descent_floor_mm(height, observation)
        total = descent_mm + max(above + aim, self.MIN_USEFUL_DESCENT_MM)
        said = self.where_the_tool_is(observation)
        if self._running:
            said = ("The moves in front of this close have run, and left it {:.0f} mm short of "
                    "the depth a close wants: {}{}".format(above + need, said[:1].lower(),
                                                           said[1:]))
        elif descent_mm >= self.NO_MOTION_MM:
            said = ("After the {:.0f} mm descent this reply carries, {}{}"
                    .format(descent_mm, said[:1].lower(), said[1:]))
        return ("{} A close from here shuts above the part it has to squeeze: the next reply "
                "comes down {:.0f} mm IN TOTAL from where the tool is now, which puts the tool "
                "point about {:.0f} mm BELOW the top face, and the close comes after it."
                .format(said, total, aim))

    def floor_gate(self, proposal, deltas, observation) -> Optional[str]:
        """No move sends the tool point under the robot's hard floor above the table
        (capabilities.floor_above_table_m), whatever a camera says is under it: the moves are
        added up in order, and one that would end below the floor refuses the reply."""
        caps = self.adapter.capabilities()
        pose = getattr(observation, "pose", None)
        floor_m = getattr(caps, "floor_above_table_m", None)
        if floor_m is None or caps.table_z_m is None or pose is None:
            return None
        floor = float(caps.table_z_m) + float(floor_m)
        start = z = float(pose.position_m[2])
        lowest = z
        for delta in deltas or []:
            if delta.kind == types.TRANSLATE and delta.axis is not None:
                z += float(delta.axis[2]) * float(delta.magnitude)
                lowest = min(lowest, z)
        if lowest >= floor - 0.001:
            return None
        height = (start - float(caps.table_z_m)) * 1000.0
        room = max(0.0, (start - floor) * 1000.0)
        return ("That brings the tool point to {:.0f} mm {} the table, and the table is a hard "
                "floor: the tool point goes no lower than {:.0f} mm above it. It is {:.0f} mm "
                "above the table now, so it may come down at most {:.0f} mm in all.".format(
                    abs(lowest - float(caps.table_z_m)) * 1000.0,
                    "above" if lowest >= float(caps.table_z_m) else "BELOW",
                    float(floor_m) * 1000.0, height, room))

    def reach_gate(self, proposal, deltas, observation) -> Optional[str]:
        """Hold a move to the radius this arm still goes where it is sent inside -- by
        SHORTENING it to that radius, which is what the controller already does with a
        move that asks for more travel than one command may. A move the radius cut at all is
        the reach WALL, and is noted for the release rules."""
        caps = self.adapter.capabilities()
        pose = None if observation is None else observation.pose
        if caps.reach_m is None or pose is None:
            return None
        asked = [d.magnitude for d in deltas]
        refused = fit_to_reach(deltas, np.asarray(pose.position_m, dtype=float)[:2],
                               caps.reach_m, self.REACH_TANGENT_M, caps.arrival_tolerance_m or 0.0)
        # This reply's own cut, not an earlier attempt's.
        self._at_the_wall = bool(refused) or [d.magnitude for d in deltas] != asked
        return refused


    def tilt_gate(self, proposal, deltas, observation) -> Optional[str]:
        """A roll or a pitch tilts the jaws for a take from the side only, and in open air: a
        hand turning off vertical sweeps round the tool point, and next to a located thing it
        stops against it. A turn back toward pointing down is any step's."""
        pose, point = getattr(observation, "pose", None), (self._last_point or {}).get("point")
        turns = [d for d in deltas or [] if d.kind == types.ROTATE and d.axis is not None
                 and abs(float(d.axis[2])) < 0.9]
        if pose is None or not turns:
            return None
        if not self._side_step():
            after = np.linalg.multi_dot([axis_angle_to_matrix(d.axis, d.magnitude)
                                         for d in reversed(turns)] + [pose.rotation])
            back = turn_back(pose, self.proposer.directions)
            return None if after[2, 2] < float(pose.rotation[2][2]) - 0.01 else (
                "A roll or a pitch tilts the jaws, which only a step taking a part from the side "
                "does. {}".format("{} brings them back to pointing down.".format(back) if back
                                  and pose.rotation[2][2] > -0.99 else "Leave them as they are."))
        if point is None:
            return None
        near = float(np.linalg.norm(np.asarray(point, dtype=float)[:3] - pose.position_m)) * 1e3
        clear = self.SLIDE_STANDOFF_MM - 2000.0 * float(
            self.adapter.capabilities().arrival_tolerance_m or 0.0)
        if near >= clear:
            return None
        stand = (side_take(self, observation) or {}).get("stand")
        return ("That turn tilts the jaws {:.0f} mm from what this step is about, inside the "
                "{:.0f} mm a turning hand sweeps: it stops against it part way. Turn in open "
                "air, {}.".format(near, clear, "a hand's length clear of it" if stand is None
                                  else "above the STAND-OFF point -- {} of the tool point".format(
                                      _say_offset((stand - pose.position_m)[:2] * 1000.0,
                                                  getattr(self.proposer, "directions", None)))))


