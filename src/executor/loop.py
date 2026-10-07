"""One subgoal, driven to completion: look, say where the target is, run a batch, measure."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.controller import runner, types
from src.controller.convert import ConversionError, to_deltas
from src.controller.types import Capabilities
from src.schema.actions import ActionError

from .context import (_history, _join, _named, _say_the_batch, already_there_mm,
                      describe_progress, describe_reach, describe_since_last, describe_since_start)
from . import geometry
from .gates import (ARRIVED_ENDING, ONE_AT_A_TIME, WRIST, Cycle, EpisodeOver, Gates, Observation,
                    WALL_ENDING, _closes, _fixed_camera_boxes, _is_descent, _say_offset, _tokens,
                    chain_gates, first_move_is_a_test, note_turn, release_gate, side_block,
                    side_take, turn_gate, lift_before_sideways, split_a_carry, the_test_held,
                    where_the_release_would_land)
from .locate import WRIST_USEFUL_BELOW_MM, locate, part_in_a_sentence, part_phrase
from .models import receiving_words, wants_holding
from .proposal import Proposer, ProposalResult
from .supervision import MotionSupervisor


class ExecutorLoop(Gates):
    """One subgoal's cycle, and the state it carries across one."""
    #: Everything that is about ONE subgoal, as it starts: a value, or the type a fresh one is
    #: made from. ``_reset_subgoal_state`` sets all of it; on a loop built without a constructor
    #: -- one rule under test -- each is made on first read.
    _SUBGOAL_STATE = {
        # where closes shut on air or stopped short, the spots looked at again since, the spot
        # a close is about to be made at again, and the height the tool has to come back up past
        "_empty_closes": list, "_relooked": list, "_relook_at": None, "_unrecovered_z": None,
        # what the last close REPORTED, out of the robot's own message; what the step wants
        "_closed_holding": None, "_wants_holding": False, "_stopped_at": list, "_turns": list,
        # where this subgoal started and the last step ended; the sideways slides of its descents
        "_origin": None, "_pose_after": None,
        # cycles running that ended with the tool where it started and the jaws as they were,
        # and those the reach cut a move of the reply short in
        "_frozen": 0, "_frozen_carried": 0, "_frozen_asked": False,
        "_at_the_wall": False, "_walled": 0,
        # what this subgoal has learnt to add to the run-up of a contact move
        # how far the tool is from the located point, and whether that point still describes it
        "_residual_mm": None, "_measured_since_lateral": False,
        # the planner's words and boxes for the thing this subgoal is about
        "_target": "", "_target_boxes": None, "_subgoal": "", "_criterion": "",
        # the looks taken, where the tool stood when the last one answered, the last located
        # point, and the face measured for a push (a measurement, not a sentence)
        "_looked_close": False, "_wrist_looked": False, "_looked_from": None,
        "_last_point": dict, "_face_said": None,
        # where the located thing meets the table -- the only width there is -- whose box, and
        # the error bar of the look that outlined it
        "_footprint": None, "_footprint_method": "", "_footprint_is_part": False,
        "_footprint_error_mm": None,
        # how far its top face stands above the table, and the surfaces the wrist saw over the
        # outline, whose median the face is
        "_top_face_mm": None, "_top_face_from": "", "_faces_seen_mm": list,
        # where the tool aims / what it holds, and the widths measured across the thing
        "_aimed_at_edge": False, "_payload_at_edge": False, "_last_width": dict,
        # the last command, so a repeat of it can be recognised, and the axis the last motion
        # was stopped along
        "_blocked_axis": None,
        # how well the last location was pinned down, and in words where that came from
        "_error_mm": None, "_error_class": "", "_disagreement": "", "_ambiguous": "",
        # what the payload is being put down ON, when a look has measured it this subgoal
        "_receiver": dict,
    }
    #: ...and what lasts the loop's whole life, made the same way.
    _LOOP_STATE = {"_spent": dict}

    gripper_only: bool = False
    #: what is in the jaws, which crosses subgoal boundaries (see carry_over)
    _payload_width_mm: Optional[float] = None
    _payload_offset_mm: Optional[float] = 0.0
    #: ...and whether it has met something since the tool last rose: a move sideways drags it
    _payload_met: bool = False
    #: the payload watch: the gap the close settled at, the one let go of, and whether the hold
    #: rule has just asked for the open that follows
    _watched_payload_mm: Optional[float] = None
    _stale_payload_mm: Optional[float] = None
    _model_calls: int = 0
    #: set while a rule is being asked about an action that is ABOUT to run, with the rest of
    #: its reply already run and measured, rather than about a reply being admitted.
    _running: bool = False

    def __getattr__(self, name):
        start = ExecutorLoop._SUBGOAL_STATE.get(name, ExecutorLoop._LOOP_STATE.get(name, _NONE))
        if start is _NONE:
            raise AttributeError(name)
        value = start() if callable(start) else start
        object.__setattr__(self, name, value)
        return value

    def __init__(self, adapter, proposer: Proposer, supervisor: MotionSupervisor,
                 observe: Callable[[], Observation], capability_text: str,
                 max_cycles: int = 8, gripper_only: bool = False):
        self.adapter = adapter
        self.proposer = proposer
        self.supervisor = supervisor
        # Every look is offered to the face path, not just a cycle's boundaries: the frame taken
        # BETWEEN a lateral leg and its descent is the one high enough to see a top face.
        def noted() -> Observation:
            observation = observe()
            self._note_the_face(observation)
            return observation
        self.observe = noted
        self.capability_text = capability_text
        self.max_cycles = min(int(max_cycles), self.MAX_CYCLES)
        # Set for a subgoal whose whole content is working the gripper.
        self.gripper_only = gripper_only
        self.cancelled = threading.Event()
        # What this loop has spent, so a cycle can say what it cost.
        self._model_calls = 0
        self._spent: Dict[str, Any] = {}
        self._reset_subgoal_state()

    def _reset_subgoal_state(self) -> None:
        """Everything that is about ONE subgoal, back to how a subgoal starts."""
        for name, start in self._SUBGOAL_STATE.items():
            setattr(self, name, start() if callable(start) else start)

    def _note_spent(self, role: str, seconds: float, calls: int) -> None:
        """What a model call cost, by role, so a run file can show where the clock went."""
        self._model_calls += int(calls)
        self._spent[role + "_s"] = self._spent.get(role + "_s", 0.0) + float(seconds)
        self._spent[role + "_calls"] = self._spent.get(role + "_calls", 0) + int(calls)

    def cancel(self) -> None:
        self.cancelled.set()

    # ------------------------------------------ where the target is, measured, as text

    def _locate(self, target: str, observation: Observation, part: Optional[bool] = None,
                boxes: Optional[Dict[str, Any]] = None, hints: Optional[Dict[str, Any]] = None,
                wrist_only: bool = False):
        """Ask the cameras where ``target`` is -- or, with ``boxes``, ask nobody and do the
        geometry on the box the planner already drew."""
        caps = self.adapter.capabilities()
        pose = observation.pose
        height = self._object_height_mm(observation)
        top_z_m = None if height is None or caps.table_z_m is None \
            else float(caps.table_z_m) + height / 1000.0
        kept = self._last_point or {}
        last_seen = kept.get("point") if kept.get("words") == target.strip().lower() else None
        air = last_seen is not None and self._aim_shut_on_air()
        asked = time.monotonic()
        # The wrist is asked only looking DOWN at the table -- holding too: over a receiver it is
        # the one close view of it (the xArm replay: receivers from the fixed views alone were
        # 49-364 mm off at 6 of 6 real releases; the simulator's baskets 50-125 mm).
        cameras = {name: view for name, view in (observation.cameras or {}).items()
                   if name != WRIST or not (self._side_step()
                                            or self._approach(observation)[2] != -1.0)}
        found = locate(self.proposer.client, self.proposer.prompts, cameras,
                       target, table_z_m=caps.table_z_m, top_z_m=top_z_m,
                       tool_z_m=None if pose is None else float(pose.position_m[2]),
                       part=self._acts_on_what_it_names() if part is None else part,
                       part_word=part_in_a_sentence(self._subgoal, target)
                       if self._wants_holding else "",
                       boxes=boxes, last_seen=last_seen, wrist_only=wrist_only,
                       hint_boxes=None if last_seen is None else hints,
                       wrist_within_mm=max(self.jaw_half_gap() * 2.0, float(
                           (last_seen is not None and kept.get("error_mm")) or 0.0)),
                       held_px=self.held_in_view(observation, cameras.get(WRIST)),
                       tallest_mm=(getattr(caps, "max_object_height_m", None) or 0) * 1e3 or None)
        self._note_spent("locate", time.monotonic() - asked, int(found.calls))
        if found.ok and self._side_step() and found.part_point is not None:   # its face's middle
            found.point_base, found.disagreement = found.part_point, ""
        elif found.ok and self._side_step() and (self._last_point or {}).get("on_a_face"):
            found.error = "no two views met on its face"        # nothing else over a face point
        # An empty close is evidence against the point that aimed it, not licence to chase a
        # look-alike across the table: what the sticky rule was written for.
        near = air and found.ok and float(np.linalg.norm(
            (np.asarray(found.point_base, dtype=float)
             - np.asarray(last_seen, dtype=float))[:2])) * 1000.0 \
            <= max(float(found.error_mm or 0.0), float(kept.get("error_mm") or 0.0))
        found = self._sticky(target, found, override=wrist_only or near)
        found.footprint = geometry.about_its_point(found.footprint, found.point_base)
        return found

    def _aim_shut_on_air(self) -> bool:
        """Has a close aimed from the point now held already shut on NOTHING?"""
        point = (self._last_point or {}).get("point")
        return point is not None and any(
            float(np.linalg.norm((np.asarray(spot, dtype=float) - point)[:2])) * 1000.0
            <= self.jaw_half_gap() for spot in self._empty_closes)

    def _acts_on_what_it_names(self) -> bool:
        """Will this step DO something to the thing it names, or only go to it?"""
        if self._wants_holding:
            return True
        words = "{} {}".format(self._subgoal or "", self._criterion or "").lower()
        return any(word in words for word in ("turn", "twist", "rotate", "press"))

    def _sticky(self, target: str, found, override: bool = False):
        """One subgoal, one object: the look kept stands till a better one moves it."""
        kept = self._last_point
        if not found.ok:
            return found
        point = np.asarray(found.point_base, dtype=float)
        if kept and not override and kept.get("words") == target.strip().lower() \
                and not self._takes_over(kept, "point", point, found.error_mm):
            moved = float(np.linalg.norm((point - kept["point"])[:2])) * 1000.0
            found.point_base = kept["point"].copy()
            if moved <= self.NO_MOTION_MM:
                return found      # the same answer again: nothing to say about it
            found.jumped_mm, found.error_mm = moved, kept.get("error_mm")
            found.error_class = kept.get("error_class") or found.error_class
            found.method = (
                "{} -- but the answer that came back was {:.0f} mm from where these words "
                "were found before, with a wider error bar, so THE PREVIOUS LOCATION WAS KEPT{}"
                .format(kept["method"], moved, "" if kept.get("waiting") is None else
                        " -- another look putting it in the same new place will take it"))
            return found
        self._last_point = {"words": target.strip().lower(), "point": point.copy(),
                            "method": found.method, "error_mm": found.error_mm,
                            "error_class": found.error_class, "waiting": None,
                            "on_a_face": found.point_base is found.part_point,
                            "body": found.part_body, "top": found.top_point}
        return found

    def _takes_over(self, kept, key, point, error_mm) -> bool:
        """Does a new look replace the one kept under ``key``: an error bar no wider than the
        kept one's, or a second look putting it where this one did?"""
        bar = float("inf") if error_mm is None else float(error_mm)
        if kept.get(key) is None or bar <= float(kept.get("error_mm") or float("inf")):
            return True
        point, waiting = np.asarray(point, dtype=float)[:2], kept.get("waiting")
        if waiting is not None and float(np.linalg.norm(point - waiting)) * 1000.0 \
                <= max(bar, self.NO_MOTION_MM):
            return True
        moved = float(np.linalg.norm(point - np.asarray(kept[key], dtype=float)[:2])) * 1000.0
        kept["waiting"] = None if moved <= bar else point.copy()
        return False

    def _note_receiver(self, words: str, found, middle, judged: bool = False) -> None:
        """Remember what the payload is being put down on: its middle and its size -- unless
        the one kept is the better look (see _takes_over), which a look ``judged`` against
        the point kept for these words already passed."""
        words = (words or "").strip()
        if not judged and self._receiver.get("words") == words \
                and not self._takes_over(self._receiver, "centroid", middle, found.error_mm):
            return
        span = geometry.longest_across(found.footprint)
        corners = 0 if found.footprint is None else int(np.asarray(found.footprint).shape[0])
        self._receiver = {"words": words, "error_mm": found.error_mm, "waiting": None,
                          "centroid": np.asarray(middle, dtype=float)[:2].copy(),
                          "radius_mm": None if span is None else span / 2.0,
                          "from": ("the middle of its outline"
                                   if corners >= self.CENTROID_NEEDS_CORNERS
                                   else "where the cameras say it stands")}

    def _target_point(self, observation):
        """Where the tool -- or what it carries -- has to get to: xy, metres, base frame."""
        point = (self._last_point or {}).get("point")
        if self._placing(observation):       # the receiver, never what is held
            centroid = self._receiver.get("centroid")
            return None if centroid is None else np.asarray(centroid, dtype=float)[:2]
        side = side_take(self, observation) if point is not None else None
        if point is None or side and side.get("stand") is not None and not side["turned"]:
            return None if point is None else side["stand"]     # in front of it, not at it
        point = np.asarray(point, dtype=float)
        aim = self.edge_point(point, observation)
        self._aimed_at_edge = aim is not None
        return point if aim is None else np.asarray(aim, dtype=float)

    def _here(self, observation, whole: bool = False):
        """What has to arrive: the tool (in 3-D if ``whole``), or, jaws full, what hangs in them."""
        pose = None if observation is None else observation.pose
        if pose is None:
            return None
        carried = self.payload_point(observation) if self._placing(observation) else None
        return np.asarray(pose.position_m, dtype=float)[:3 if whole else 2] if carried is None \
            else np.asarray(carried, dtype=float)[:2]

    def _refresh_residual(self, observation) -> None:
        """How far the tool still is from the located point -- arithmetic, every cycle."""
        miss = self._miss(self._target_point(observation), observation)
        if miss is not None:
            self._residual_mm = float(np.linalg.norm(miss)) * 1000.0

    def _in_the_fine_phase(self, observation) -> bool:
        """Air under the tool, AND something within reach of a correction to be fine about."""
        clearance = None if observation is None else observation.clearance_mm
        if clearance is None or float(clearance) > self.FINE_PHASE_MM:
            return False
        return self._residual_mm is None or float(self._residual_mm) <= self.FINE_PHASE_MM

    def _sensing(self, observation) -> Any:
        """(the words to find, the planner's boxes for them): while the jaws hold something where
        it is going, never it; taken from the side, the part named, which no box round it says."""
        if self._placing(observation):
            return receiving_words(self._criterion or "", self._subgoal or "",
                                   held=self._target or ""), None
        part = part_phrase(self._subgoal, self.SIDE_WORDS) if self._side_step() else ""
        return (part, None) if part else (self._target, self._target_boxes)

    def _should_sense(self, observation) -> str:
        """Why this cycle needs a fresh look, or "" for "the point it has is still good"."""
        words, _ = self._sensing(observation)
        if not words:
            return ""
        if self._last_point.get("words") != words.strip().lower() or self._side_step() \
                and not self._last_point.get("on_a_face"):      # ...on its face, from the side
            return "nothing has located it yet"
        if not self._measured_since_lateral:
            return "the world has been shoved since the last look"
        if self._in_the_fine_phase(observation) and not self._looked_close:
            # The one place a model call earns itself: down here the wrist sees the thing bigger
            # and squarer than any fixed camera, and the last 20 mm decide every empty close.
            return "the tool is down among the objects and the wrist can see it"
        if self._carried_further_than_a_close_survives(observation):
            # A correction that big carries the last answer's error with it: look again.
            return "the tool has crossed further than a close survives since the last look"
        return ""

    def _carried_further_than_a_close_survives(self, observation) -> bool:
        """Has the tool carried the last answer further across the table than the miss a
        close can survive?"""
        here = self._here(observation)
        if here is None or self._looked_from is None \
                or not self._in_the_fine_phase(observation):
            return False
        bar = max(self.MIN_CLOSE_TOLERANCE_MM, self.close_tolerance(observation))
        gone = float(np.linalg.norm(here - np.asarray(self._looked_from, dtype=float))) * 1000.0
        return gone > bar

    def _sense_target(self, observation, cycle: Cycle, emit, wrist: str = "",
                      owed: str = "") -> None:
        """Look for the target and write down where it is."""
        why = wrist or owed or self._should_sense(observation)
        words, given = self._sensing(observation)
        if not why or not words:
            return
        close = bool(wrist) or self._in_the_fine_phase(observation)
        known = self._last_point.get("words") == words.strip().lower()
        # The planner's boxes are a picture from before anything moved: a first look's only.
        boxes = given if given and not close and not known else None
        found = self._locate(words, observation, boxes=boxes, hints=given,
                             wrist_only=bool(wrist))
        if close and found.ok:
            # Only a look that ANSWERED spends the fine phase's one model call.
            self._looked_close = True
        cycle.reach = {"target": words, "because": why,
                       "asked_a_model": boxes is None,
                       "from_plan_boxes": boxes is not None,
                       "located": found.as_dict()}
        emit({"event": "reached", "cycle": cycle.index, "target": words,
              "because": why, "asked_a_model": boxes is None, "located": found.as_dict()})
        if not found.ok:
            cycle.reach["text"] = "could not find {!r}: {}".format(words, found.error)
            return
        if geometry.measures_more(found.footprint, self._footprint, found.view_name == WRIST):
            self._footprint, self._footprint_method = found.footprint, found.footprint_method
            self._footprint_is_part, self._footprint_error_mm = found.part_outline, found.error_mm
        self._looked_from = self._here(observation)
        self._disagreement = found.disagreement
        self._ambiguous = found.ambiguous or ""
        self._error_mm, self._error_class = found.error_mm, found.error_class
        self._measured_since_lateral = True
        if self._placing(observation) and not getattr(found, "jumped_mm", None):
            middle = geometry.outline_centroid(found.footprint, self.CENTROID_NEEDS_CORNERS)
            self._note_receiver(words, found, middle if middle is not None
                                else found.point_base, judged=known)
        self._note_the_face(observation)
        self._refresh_residual(observation)
        cycle.reach["residual_mm"] = None if self._residual_mm is None \
            else round(self._residual_mm, 1)
        cycle.reach["text"] = "located {!r}{}: {}{}".format(
            words, _named(found), found.method,
            "" if self._residual_mm is None
            else ", {:.0f} mm from the tool".format(self._residual_mm))

    def _wrist_look(self, observation, cycle: Cycle, emit) -> None:
        """One look straight down, before the first descent of a grasp comes down."""
        caps, pose = self.adapter.capabilities(), observation.pose
        was = (self._last_point or {}).get("point")
        up = geometry.to_level(pose, caps.table_z_m)       # a look DOWN: none, tilted
        if self._wrist_looked or not self._wants_holding or was is None or up is None \
                or up * 1000.0 >= WRIST_USEFUL_BELOW_MM:
            return
        self._wrist_looked = True
        before = dict(cycle.reach)
        self._sense_target(observation, cycle, emit,
                           wrist="the tool is over it and about to come down on it, and a "
                                 "descent cannot be corrected once it is under way")
        cycle.reach = dict(cycle.reach, before=dict(
            before, point_mm=[round(float(v) * 1000.0, 1) for v in np.asarray(was)]))
        self._follow_the_look(was, observation, cycle, emit)


    def _look_again_before_the_close(self, observation, cycle: Cycle, emit) -> None:
        """A close is about to run where one already shut on air: the answer it was aimed from
        is looked at again, and the tool follows what that look moved."""
        spot, self._relook_at = self._relook_at, None
        was = (self._last_point or {}).get("point")
        here = None if observation is None or observation.pose is None \
            else np.asarray(observation.pose.position_m, dtype=float)
        if spot is None or was is None or here is None \
                or self._near(here, [spot]) is None:
            return                    # the reply's own moves took the tool off that spot
        self._relooked.append(np.asarray(spot, dtype=float).copy())
        self._sense_target(observation, cycle, emit,
                           wrist="a close at this very spot already shut on nothing, so the "
                                 "answer it was aimed from is the thing to look at again")
        self._follow_the_look(was, observation, cycle, emit)

    def _follow_the_look(self, was, observation, cycle: Cycle, emit) -> None:
        """Move the tool by what that look moved the answer, before the descent runs."""
        now = (self._last_point or {}).get("point")
        if now is None:
            return
        shift = (np.asarray(now, dtype=float) - np.asarray(was, dtype=float))[:2]
        far = float(np.linalg.norm(shift)) * 1000.0
        # Under the look's own error bar nothing moves; over the half-gap the jaws close from,
        # the two answers are not about one thing and a blind sidestep will not fix it.
        if far < max(self.NO_MOTION_MM, float(self._error_mm or 0.0)) or far > self.jaw_half_gap():
            return
        command = {"actions": [{"type": "move", "axis": [float(shift[0]), float(shift[1]), 0.0],
                                "distance_mm": far,
                                "note": "the look taken before this descent put the thing "
                                        "{:.0f} mm from where the tool was aimed, and the "
                                        "tool was moved by that before coming down"
                                        .format(far)}]}
        try:
            deltas = to_deltas(command, self.adapter.capabilities(), self.proposer.directions)
        except (ActionError, ConversionError) as exc:                   # pragma: no cover
            cycle.error = str(exc)
            return
        emit({"event": "followed_the_look", "cycle": cycle.index, "mm": round(far, 1)})
        self._run_one(deltas[0], observation, cycle, emit)
        if self._pose_after is not None:
            # The look was answered from here now, not from where it was asked.
            self._looked_from = np.asarray(self._pose_after.position_m, dtype=float)[:2]

    def _target_block(self, observation) -> str:
        """WHERE THE TARGET IS, in the robot's own words and millimetres."""
        miss = self._miss(self._target_point(observation), observation)
        if miss is None or observation.pose is None:
            return ""
        directions = getattr(self.proposer, "directions", None)
        words, _ = self._sensing(observation)
        # ...from what has to arrive: said so, or a model corrects for a hang already counted.
        held = self._placing(observation) and self.payload_point(observation) is not None
        said = ["WHERE THE TARGET IS",
                "{}: {} of {}.".format(words or "the target", _say_offset(miss * 1e3, directions),
                                       "the middle of what the jaws hold, measured off the tool "
                                       "point" if held else "the tool point")]
        above = self._above_the_face_mm(observation)
        # While placing, the face kept is the payload's own, not the top of what it goes on.
        if above is not None and self._top_face_mm is not None and not self._placing(observation):
            said.append("Its top face is {:.0f} mm {} the tool point.".format(
                abs(above), "BELOW" if above > 0 else "ABOVE"))
        said.append("These are measured, not estimated: copy them into your move distances "
                    "and use these direction words.")
        caps = self.adapter.capabilities()
        radius = describe_reach(caps, observation.pose)
        if radius:
            said.append(radius)
        tolerance = caps.arrival_tolerance_m
        if tolerance:
            said.append("Anything under {:.0f} mm of it is already met: this arm counts "
                        "itself arrived within {:.0f} mm, so a move that small changes "
                        "nothing anyone can measure -- go down and work the jaws instead of "
                        "correcting it again.".format(tolerance * 2000.0, tolerance * 1000.0))
        offset_mm, why = self.edge_offset_mm(observation)
        if offset_mm:
            said.append("The {:.0f} mm the jaws need to sit off its middle, along the line "
                        "they close along ({}), is already taken off those numbers: they are "
                        "what is LEFT to remove. Move them as written and add nothing to "
                        "them.".format(offset_mm, why))
            turn = self.quarter_turn_for_reach(observation)
            if turn:
                said.append("NEITHER RIM along that line is inside this arm's reach. {} turns the "
                            "jaws to close along the other line, whose near rim is: turn first, "
                            "and these offsets are then measured for it.".format(turn))
        if self._error_mm is not None:
            said.append("(from {}, within about {:.0f} mm)".format(
                (self._last_point or {}).get("method") or "the cameras", self._error_mm))
        if self._disagreement:
            said.append("BUT TWO VIEWS DISAGREE: {}.".format(self._disagreement))
        if self._ambiguous:
            said.append("BUT MORE THAN ONE THING FITS THOSE WORDS: {}".format(self._ambiguous))
        return said[0] + "\n" + " ".join(said[1:])

    # ------------------------------------------------- what survives a subgoal boundary

    def carry_over(self) -> Dict[str, Any]:
        """What this subgoal measured that the next one may still be entitled to use."""
        return {"target": (self._target or "").strip().lower(),
                **{name: getattr(self, "_" + name)
                   for name in _CARRIED_AS_IS + _CARRIED_IF_MEASURED},
                "frozen": self._frozen, "error_class": self._error_class,
                "fresh": self._measured_since_lateral,
                "last_point": dict(self._last_point) if self._last_point else {},
                "disagreement": self._disagreement,
                "empty_closes": [np.asarray(spot, dtype=float).copy()
                                 for spot in self._empty_closes],
                # the outline, so the step that CLOSES still knows how wide the thing is: such a
                # grasp measures in its first half and closes in its second
                "footprint": None if self._footprint is None
                else np.asarray(self._footprint, dtype=float).copy(),
                "footprint_method": self._footprint_method,
                "footprint_is_part": self._footprint_is_part,
                # ...and how tall, measured once: the second half is too close to see anything
                "top_face_mm": self._top_face_mm,
                "faces_seen_mm": list(self._faces_seen_mm or []),
                "top_face_from": self._top_face_from,
                # ...and what is in the jaws: the take and the put name different objects
                "payload": {key: getattr(self, name) for key, name in _PAYLOAD.items()},
                # ...and what it is being put down on, for the same reason
                "receiver": dict(self._receiver) if self._receiver else {}}

    def _adopt(self, carry: Optional[Dict[str, Any]], target: str) -> str:
        """Keep the last subgoal's measurement when this one is about the same object."""
        payload = (carry or {}).get("payload") or {}
        if payload:
            # Taken whatever the words say: a put names the thing it goes ON, not what is held;
            # and with it the LET GO mark, without which a restored width reads as a close made.
            for key, name in _PAYLOAD.items():
                setattr(self, name, payload.get(key))
        if (carry or {}).get("receiver"):
            self._receiver = dict(carry["receiver"])
        words = (target or "").strip().lower()
        if not carry or not words or (carry.get("target") or "") != words:
            return ""
        self._target_boxes = None      # the plan's picture is older than the last step's looks
        for name in _CARRIED_AS_IS:
            setattr(self, "_" + name, carry.get(name))
        self._error_class = carry.get("error_class") or ""
        self._measured_since_lateral = bool(carry.get("fresh"))
        self._last_point = dict(carry.get("last_point") or {})
        self._disagreement = carry.get("disagreement") or ""
        if carry.get("footprint") is not None:
            self._footprint = np.asarray(carry["footprint"], dtype=float).copy()
            self._footprint_method = carry.get("footprint_method") or ""
            self._footprint_is_part = bool(carry.get("footprint_is_part"))
        if carry.get("top_face_mm") is not None:
            self._top_face_mm = float(carry["top_face_mm"])
            self._top_face_from = carry.get("top_face_from") or ""
        for name in _CARRIED_IF_MEASURED:
            if carry.get(name) is not None:
                setattr(self, "_" + name, float(carry[name]))
        self._faces_seen_mm = [float(v) for v in carry.get("faces_seen_mm") or []]
        self._empty_closes = [np.asarray(spot, dtype=float).copy()
                              for spot in carry.get("empty_closes") or []]
        # A retry of the same object is one approach continued: a frozen count restarting at
        # every boundary never reaches its limit over renamed subgoals.
        self._frozen_carried = int(carry.get("frozen") or 0)
        if self._residual_mm is None:
            return ""
        return ("the last step was about this same object, and it measured the tool {:.0f} mm "
                "from it{}".format(self._residual_mm,
                                   "" if self._measured_since_lateral
                                   else " before the tool was moved sideways"))

    # ------------------------------------------------------------ the endings code settles

    def _held_it(self, cycle: Cycle, emit, before=None) -> bool:
        """Is this step finished because the robot says it is holding the thing?"""
        if not self._wants_holding or self._closed_holding is not True:
            return False
        if self._placing(before):
            # It was already holding when this cycle began, so the close took nothing new: a lift
            # and a carry say "still holding" too, and a re-close would end them at their start.
            self._closed_holding = None
            return False
        self._closed_holding = None
        width, source = self.target_width(before)
        ending = HELD_IN_CODE
        if self._payload_width_mm is not None and width is not None:
            ending = HELD_WITH_A_WIDTH.format(
                gap=self._payload_width_mm, width=width,
                source=source or "measured from the located outline")
            cycle.proposal["held"] = {"gap_mm": round(float(self._payload_width_mm), 1),
                                      "target_width_mm": round(float(width), 1),
                                      "target": self._target}
        cycle.stopped_by = ending
        cycle.proposal["done"] = True
        cycle.proposal["done_in_code"] = ending
        emit({"event": "held", "cycle": cycle.index, "because": ending})
        return True

    def _let_go_instead(self, observation, cycle: Cycle, emit) -> bool:
        """A put-down step that has stopped moving still has one thing left to do: open --
        at the reach wall, anywhere on what it goes on and not only over its middle."""
        if observation is None or self.cancelled.is_set():
            return False
        if not self._placing(observation) or not self._about_letting_go():
            return False
        if where_the_release_would_land(self, observation):
            return False               # still in the air, or off to one side: opening drops it
        command = {"actions": [{"type": "gripper", "state": "open",
                                "note": "the step is about letting it go, it is down and over "
                                        "the middle, and nothing has moved for cycles"}]}
        try:
            deltas = to_deltas(command, self.adapter.capabilities(), self.proposer.directions)
        except (ActionError, ConversionError) as exc:                   # pragma: no cover
            cycle.error = str(exc)
            return False
        cycle.release = {"in_code": True}
        for delta in deltas:
            if self._run_one(delta, observation, cycle, emit):
                break
        cycle.stopped_by = RELEASED_IN_CODE
        return True

    def _positional_criterion(self) -> bool:
        said = (self._criterion or "").lower()
        return any(word in _tokens(said) or word in said
                   for word in self.POSITION_CRITERION)

    def _ask_instead_of_freezing(self, cycle: Cycle) -> bool:
        """A frozen pose on a step about POSITION is a question, not an ending."""
        if self._frozen_asked or not self._positional_criterion():
            return False
        self._frozen_asked = True
        cycle.stopped_by = (
            "nothing has moved for {} cycles running and what this step asks for is about "
            "WHERE the tool is, which a motion cannot be the only answer to. Look again: if "
            "the tool is where the step wanted it, say so and say done. If it is not, say "
            "what is still wrong and command the one motion that would fix it."
            .format(self._frozen))
        return True

    def _frozen_out(self, cycle: Cycle, emit, observation=None) -> bool:
        """Has this subgoal stopped being able to change anything -- or arrived, as measured?"""
        run, walled = max(self._frozen, self._walled), self._walled >= self.FROZEN_CYCLES
        if run < self.FROZEN_CYCLES:
            return False
        if self._let_go_instead(observation, cycle, emit):
            return True
        arrived = already_there_mm(self, observation) if self._positional_criterion() else None
        if walled and arrived is None:
            cycle.stopped_by = WALL_ENDING.format(self._walled)
            emit({"event": "frozen", "cycle": cycle.index, "cycles": run,
                  "because": "at the reach wall"})
            return True
        if arrived is None and self._ask_instead_of_freezing(cycle):
            return False
        cycle.stopped_by = FROZEN_ENDING.format(run) if arrived is None else \
            ARRIVED_ENDING.format(run, arrived, self._sensing(observation)[0] or "it")
        emit({"event": "frozen", "cycle": cycle.index, "cycles": run})
        return True

    # ------------------------------------------------------------------ the loop

    def run(self, subgoal: str, on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
            criterion: str = "", target: str = "", carry: Optional[Dict[str, Any]] = None,
            history_preface: str = "", target_boxes: Optional[Dict[str, Any]] = None):
        """Drive one subgoal."""
        def emit(event: Dict[str, Any]) -> None:
            if on_event is not None:
                on_event(event)

        cycles: List[Cycle] = []
        previous: Optional[Observation] = None
        caps: Capabilities = self.adapter.capabilities()
        self._reset_subgoal_state()
        self._target = (target or "").strip()
        # The rules read the step's own sentence: what a retry appends after a blank line is
        # for the model, and its words are not the step's.
        self._subgoal, self._criterion = subgoal or "", criterion or ""
        self._wants_holding = wants_holding(criterion or "")
        self._target_boxes = _fixed_camera_boxes(target_boxes)
        kept = self._adopt(carry, target)
        if kept:
            emit({"event": "carried", "carried": kept})
        # What a reply is held to: the robot's hard floor above the table, the arm's reach, a
        # tilt only where a hand may turn, no close or descent again where one shut on air until
        # the tool has come back up, no close short of the depth a close wants, no open off the
        # receiver's middle, and the first move after a wall grip cut to a test the rest waits on.
        rules = (turn_gate, release_gate, the_test_held, first_move_is_a_test)
        self.proposer.gate = chain_gates(
            getattr(self.proposer, "gate", None), self.floor_gate, self.reach_gate,
            self.tilt_gate, self.empty_pickup_gate, self.close_depth_gate,
            *(partial(r, self) for r in rules))

        # One cycle is one propose call and its batch, so the budget counts propose calls.
        index, spent = -1, 0
        while spent < self.max_cycles:
            index += 1
            if self.cancelled.is_set():
                break
            over = getattr(self.adapter, "episode_over", None)
            if callable(over) and over():
                counted = getattr(self.adapter, "ticks", None)
                ticks = counted() if callable(counted) else None
                emit({"event": "episode_over", "cycle": index, "ticks": ticks})
                raise EpisodeOver(
                    "the simulator stopped accepting motion part way through this subgoal "
                    "(it reached its episode step horizon{}), so nothing commanded from here "
                    "can move: the run is over, whatever the plan still says"
                    .format("" if ticks is None else ": " + ", ".join(
                        "{} {}".format(k, v) for k, v in sorted(ticks.items()))))
            cycle = Cycle(index=index)
            began, calls_before = time.monotonic(), self._model_calls
            propose_s, result = 0.0, None
            self._at_the_wall = False
            self._spent = {"locate_s": 0.0, "locate_calls": 0,
                           "supervise_s": 0.0, "supervise_calls": 0}

            def stamp(cycle=cycle, began=began, calls_before=calls_before) -> None:
                """Where this cycle's seconds went, by role."""
                if cycle.timing:
                    return
                cycle.timing = {"wall_s": round(time.monotonic() - began, 2),
                                "motion_s": round(sum(float(step.get("elapsed_s") or 0.0)
                                                      for step in cycle.steps), 2),
                                "model_calls": self._model_calls - calls_before,
                                "propose_s": round(propose_s, 2),
                                "propose_calls": len(getattr(result, "attempts", None) or []),
                                "locate_s": round(self._spent["locate_s"], 2),
                                "locate_calls": self._spent["locate_calls"],
                                "supervise_s": round(self._spent["supervise_s"], 2),
                                "supervise_calls": self._spent["supervise_calls"]}
                emit({"event": "timing", "cycle": cycle.index, **cycle.timing})

            observation = self.observe()
            if self.cancelled.is_set():         # the clock or a STOP landed during that look
                break
            cycles.append(cycle)
            if self._origin is None:
                self._origin = observation.pose
            for line in (describe_since_start(self._origin, observation.pose),
                         describe_since_last(previous, observation)):
                if line:
                    observation.robot_text = observation.robot_text + "\n" + line
            # Where the target is and how far the tool still has to go.
            self._refresh_residual(observation)
            self._sense_target(observation, cycle, emit)
            # Where the tool point is against the top face under it, said BEFORE a close is
            # proposed: the sign of that number separated every grasp that held from every empty.
            observation.robot_text = (observation.robot_text + "\n"
                                      + self.where_the_tool_is(observation))
            for block in (self._target_block(observation), side_block(self, observation),
                          self.where_the_payload_is(observation)):
                if block:
                    observation.robot_text = observation.robot_text + "\n" + block
            cycle.stopped_by = (side_take(self, observation) or {}).get("cannot", "")
            if cycle.stopped_by:
                stamp()
                break
            emit({"event": "observed", "cycle": index,
                  "capture_time": observation.capture_time,
                  "images": list(observation.names)})

            # The two numbers the phase is decided on, kept so a run can be read back.
            cycle.batch = {"clearance_mm": observation.clearance_mm,
                           "residual_mm": self._residual_mm}
            if self.gripper_only:
                forbid = ("This subgoal is ONLY about the gripper. Do not move the arm: the "
                          "tool is already where it needs to be. Answer with a \"gripper\" "
                          "action, or with \"done\" if the gripper is already as the subgoal "
                          "wants it.")
            else:
                forbid = _watcher_verdict(cycles[:-1])
                if self._in_the_fine_phase(observation):
                    forbid = _join(forbid, ONE_AT_A_TIME)

            asked = time.monotonic()
            result = self.proposer.propose(subgoal, observation,
                                           observation.robot_text, caps, self.capability_text,
                                           criterion=criterion, target=self._target,
                                           history=_join(history_preface,
                                                         _history(cycles[:-1])),
                                           legend=observation.legend, forbid=forbid)
            propose_s = time.monotonic() - asked
            self._model_calls += len(result.attempts or [])
            cycle.proposal = result.as_dict()
            emit({"event": "proposed", "cycle": index, **cycle.proposal})

            if not result.ok:
                # Every retry inside one propose call was refused -- at the wall, perhaps.
                if not self._let_go_instead(observation, cycle, emit):
                    cycle.error = result.error
                stamp()
                break
            if result.done and not result.deltas:
                # ...unless nothing has moved for so long that "it is met" is about nothing.
                run = self._frozen_carried + self._frozen + 1
                cycle.stopped_by = FROZEN_ENDING.format(run) \
                    if self._frozen_carried and run >= self.FROZEN_CYCLES \
                    else "the model says the subgoal is met"
                stamp()
                break
            if not result.deltas:
                cycle.error = "the proposal carried no actions and did not claim it was done"
                stamp()
                break
            # A reply carrying BOTH a last action and "done" runs the action and then stops.
            last_action = bool(result.done)

            stopped = self._run_batch(subgoal, result, observation, cycle, emit)
            self._note_cycle(result, cycle, observation)
            if _moved_the_world(result.deltas, cycle, self._last_point.get("error_mm")):
                # Something was leaned on or dragged: the last answer is about where it WAS.
                self._measured_since_lateral = False
            stamp()
            previous = observation
            spent += 1
            if self._held_it(cycle, emit, observation):
                break
            if stopped or self.cancelled.is_set():
                break
            if self._frozen_out(cycle, emit, observation):
                break
            if last_action:
                cycle.stopped_by = _done_after_the_last_action
                break
        return cycles

    # ------------------------------------------------------------------ one batch

    def _run_batch(self, subgoal: str, result: ProposalResult, before: Observation,
                   cycle: Cycle, emit) -> bool:
        """Run the actions of one reply back to back, with the gates between them."""
        # The legs a long carry is going to be run in, cut ABOVE the chain rather than under
        # it: see gates.split_a_carry. A reply that carries nothing comes back unchanged.
        deltas, lifted = lift_before_sideways(self, list(result.deltas), before)
        if lifted:
            cycle.batch = dict(cycle.batch or {}, mended=lifted)
            emit({"event": "mended", "cycle": cycle.index, "because": lifted})
        deltas = split_a_carry(self, deltas, before)
        started = time.monotonic()
        residual_before = self._residual_mm
        gate = getattr(self.proposer, "gate", None)
        stopped = False
        for index, delta in enumerate(deltas):
            if self.cancelled.is_set():
                return True
            # Every action after the first is admitted against what the robot looks like NOW, not
            # the picture the reply was written from -- a close, with its own descent measured.
            now = before if not index else self.observe()
            refused = None
            if index:
                self._refresh_residual(now)
                refused = self._judge(gate, result.proposal, delta, now)
            if not refused and _closes([delta]) and self._relook_at is not None:
                # The gate armed a look over a close: it is taken, and the close judged again
                # from where that look left the tool.
                self._look_again_before_the_close(now, cycle, emit)
                now = self.observe()
                refused = self._judge(gate, result.proposal, delta, now)
            if refused:
                cycle.batch = dict(cycle.batch or {}, stopped_at=index, refused=refused)
                cycle.stopped_by = refused
                emit({"event": "batch_stopped", "cycle": cycle.index, "at": index,
                      "because": refused})
                stopped = True
                break
            if _is_descent([delta], self._approach(now)):
                self._wrist_look(now, cycle, emit)
            if self._run_one(delta, now, cycle, emit):
                return True
            last = cycle.steps[-1] if cycle.steps else {}
            if self.dragged(last, delta, now) or str(last.get("outcome") or "") \
                    not in (types.DONE, types.LIMITED, types.CONTACT):
                stopped = True
                break
        cycle.batch = dict(cycle.batch or {}, actions=len(deltas), ran=len(cycle.steps),
                           motion_s=round(time.monotonic() - started, 2))
        if not stopped or self.cancelled.is_set():
            # A batch that ran to the end, or was cancelled, is not looked at again here: the
            # next cycle observes for itself. The watcher is for a batch that went wrong.
            return False
        after = self.observe()
        self._refresh_residual(after)
        grew = (residual_before is not None and self._residual_mm is not None
                and self._residual_mm > residual_before + self.NO_MOTION_MM)
        self._ask_the_watcher(subgoal, result, before, after, cycle, emit, grew)
        return False

    def _judge(self, gate, proposal, delta, now) -> Optional[str]:
        """One action of a running batch against the gates, the moves in front of it run."""
        if not gate:
            return None
        self._running = True
        try:
            return gate(proposal, [delta], now)
        finally:
            self._running = False

    def _run_one(self, delta, before: Observation, cycle: Cycle, emit) -> bool:
        """One action, measured. True when the subgoal should end (a cancel or a fault)."""
        grasping = self._a_grasping_step() or not self._far_above(before)
        releasing = self.letting_go(delta, before)
        for step in runner.plan(self.adapter, [delta]).steps:
            if self.cancelled.is_set():
                return True
            emit({"event": "step_start", "label": step.label})
            try:
                outcome = self.adapter.run_step(step)
            except Exception as exc:                          # a fault is a result too
                cycle.error = str(exc)
                emit({"event": "step_error", "error": str(exc)})
                return True
            self._note_step(outcome, before.pose, grasping=grasping)
            # What the delta asked for, written onto what the robot measured of it: the AXIS, so
            # a step that got nowhere reads on its own, and what the controller said of it.
            measured = dict(outcome.as_dict(), **(
                {"axis": [round(float(v), 4) for v in np.asarray(delta.axis, dtype=float)]}
                if delta.kind == types.TRANSLATE and delta.axis is not None
                and not delta.push else {}))
            if delta.note:
                measured["note"] = delta.note
            if outcome.abort_reason == "blocked" and delta.axis is not None:
                self._blocked_axis = np.asarray(delta.axis, dtype=float)   # met it mid-batch
            cycle.steps.append(measured)
            emit({"event": "step_end", **measured})
            if outcome.outcome != types.DONE:
                # Short of the goal, refused, or stopped on something.
                cycle.stopped_by = "{}: {}".format(outcome.outcome, outcome.message)
                break
        if releasing:
            self.note_the_release(delta, before, cycle)
            emit({"event": "letting_go", "cycle": cycle.index, **cycle.release})
        return False

    def _ask_the_watcher(self, subgoal: str, result: ProposalResult, before: Observation,
                         after: Observation, cycle: Cycle, emit, grew: bool) -> None:
        """One supervision call, after the fact, about a batch that did not go as told."""
        self.supervisor.begin(from_capture=before.capture_time)
        commanded_mm = sum(d.magnitude * 1000.0 for d in result.deltas
                           if d.kind == types.TRANSLATE)
        asked = time.monotonic()
        check = self.supervisor.check(
            subgoal=subgoal, motion=_say_the_batch(result.deltas),
            expectation=(result.proposal.expect if result.proposal else "") or "not stated",
            progress=describe_progress(cycle.steps, commanded_mm),
            robot_text=after.robot_text, observation=after, before=before,
            legend=after.legend)
        self._note_spent("supervise", time.monotonic() - asked, 1)
        if check is None:
            return
        cycle.supervision.append(check.as_dict())
        cycle.finding = check.finding
        emit({"event": "supervision", "after_the_fact": True,
              "because": "the gap grew" if grew else "the batch was stopped",
              **check.as_dict()})

    # -------------------------------------------------------- what the cycle measured

    def _note_cycle(self, result, cycle: Cycle, observation=None) -> None:
        """Remember the command, and whether the robot measured anything come of it."""
        moved = sum(abs(float(s.get("moved_mm") or 0.0)) for s in cycle.steps)
        turned = sum(abs(float(s.get("turned_deg") or 0.0)) for s in cycle.steps)
        # A cycle that moved neither the tool -- or stopped short of what it asked for -- nor the
        # jaws changed nothing a criterion is about.
        still = (moved < self.NO_MOTION_MM or self._stopped_short(result, cycle)) \
            and turned < self.NO_TURN_DEG and not self._jaws_changed(result, observation)
        # ...and a PUT whose move the arm's radius cut short is not still but at the WALL,
        # where what is left to try is letting go on the receiver short of its middle.
        walled = still and self._at_the_wall and self._placing(observation)
        self._frozen = self._frozen + 1 if still and not walled else 0
        self._walled = self._walled + 1 if walled else 0
        # ...and, separately, whether it got nowhere ALONG ITS OWN AXIS: a blocked descent leans
        # a couple of millimetres sideways, so "did it move at all" says yes to nothing. With
        # the jaws full any block is kept -- a descent that met something is let go of by touch.
        self._note_what_the_payload_met(cycle, observation)
        note_turn(self, result, cycle, observation)
        self._blocked_axis = None
        for step in cycle.steps or []:
            if step.get("abort_reason") != "blocked" or not step.get("axis"):
                continue
            if abs(float(step.get("along_axis_mm") or 0.0)) < self.NO_MOTION_MM \
                    or self._placing(observation):
                self._blocked_axis = np.asarray(step["axis"], dtype=float)


    def _stopped_short(self, result, cycle: Cycle) -> bool:
        """Did this cycle's motion stop dead, or travel a fraction of what it was told to?"""
        if any(step.get("abort_reason") == "blocked" for step in cycle.steps or []):
            return True
        asked = sum(float(getattr(d, "magnitude", 0.0) or 0.0)
                    for d in (getattr(result, "deltas", None) or [])
                    if d.kind == types.TRANSLATE) * 1000.0
        got = sum(abs(float(step.get("along_axis_mm") or 0.0)) for step in cycle.steps or [])
        return asked > self.MIN_USEFUL_DESCENT_MM and got < self.BLOCKED_SHARE * asked

    def _jaws_changed(self, result, observation) -> bool:
        """Did this cycle's command actually change what the jaws are doing?"""
        states = [d.gripper_state for d in (getattr(result, "deltas", None) or [])
                  if d.kind == types.GRIPPER and d.gripper_state]
        if not states:
            return False
        shut = self._jaws_shut(observation)
        if shut is None:
            return True                   # nobody can say it did not
        return not all((state == "close") == shut for state in states)


#: What crosses a subgoal boundary on the same object as ``_<name>`` <-> ``carry[name]``: as it
#: stands, and -- a height -- only when it was measured.
_CARRIED_AS_IS = ("residual_mm", "error_mm", "last_width")
_CARRIED_IF_MEASURED = ("footprint_error_mm",)
#: ...and what is in the jaws, whatever the next step names: its key in the carry, and its name.
_PAYLOAD = {"width_mm": "_payload_width_mm", "across_mm": "_payload_across_mm",
            "offset_mm": "_payload_offset_mm", "at_edge": "_payload_at_edge",
            "met": "_payload_met", "stale_mm": "_stale_payload_mm"}


#: what a state table has no entry for
_NONE = object()


# ------------------------------------------------------------------ reading a motion


def _lateral_mm(cycle: Cycle) -> float:
    """How far sideways the robot measured itself going during this motion."""
    return sum(abs(float(step.get("lateral_mm") or 0.0)) for step in cycle.steps)


def _moved_the_world(deltas, cycle: Cycle, error_mm: Optional[float] = None) -> bool:
    """Could this cycle have moved the THING, rather than only the tool? A skid inside the kept
    point's own error bar is one no look can tell from where it was: looks again only drift."""
    if any(d.kind == types.TRANSLATE and (d.push or d.stop_on_contact) for d in deltas or []):
        return True
    return _is_descent(deltas) and _lateral_mm(cycle) >= max(ExecutorLoop.SKID_MM, error_mm or 0.0)


@dataclass
class Move:
    """One ``move`` action as the model wrote it: a word, or an axis, never both."""

    word: Optional[str] = None
    axis: Optional[List[float]] = None

    @property
    def name(self) -> str:
        return self.word or "axis {}".format(self.axis)


def _moves(cycle: Cycle) -> List[Move]:
    """Every ``move`` in a cycle's proposed command, in the order it was written."""
    actions = (((cycle.proposal or {}).get("proposal") or {}).get("command") or {}) \
        .get("actions") or []
    out: List[Move] = []
    for action in actions:
        if action.get("type") != "move":
            continue
        word, axis = action.get("direction"), action.get("axis")
        if word or axis is not None:
            out.append(Move(word=word or None,
                            axis=list(axis) if axis is not None else None))
    return out


def _watcher_verdict(cycles: List[Cycle]) -> str:
    """The restriction a wrong-direction or overshoot finding puts on the NEXT proposal."""
    if not cycles:
        return ""
    last = cycles[-1]
    moves = _moves(last)
    if not moves or last.finding not in ("wrong_direction", "overshoot"):
        return ""
    name = moves[0].name
    if last.finding == "wrong_direction":
        return ("THE WATCHER JUDGED YOUR LAST MOTION (\"{}\") TO HAVE TAKEN THE TARGET AWAY "
                "FROM THE MARKER: that direction is wrong. Do not propose \"{}\" again now. "
                "Re-read WHERE THE TARGET IS and copy its words; say in your assessment which "
                "it is and why.".format(name, name))
    return ("THE WATCHER JUDGED YOUR LAST MOTION (\"{}\") TO HAVE CROSSED TO THE OTHER SIDE "
            "OF THE MARKER: you have gone past it. The next move is the opposite of \"{}\", "
            "and smaller than the last one.".format(name, name))


# --------------------------------------------------------------- the endings, by name

#: What a cycle that ran the model's own last action and then took its word for it is called.
_done_after_the_last_action = "the model ran one last action and says the subgoal is met"

#: The marker the verifier looks for in what the executor says ended the step.
HELD_MARKER = "the jaws closed and the robot reports it is holding something"
HELD_IN_CODE = (HELD_MARKER + ", which is what this step asked for -- settled here, not asked "
                "again")
#: ...and the same ending with the number that says WHAT is in the jaws: a close settles at about
#: the width it closed on, so a gap far from the measured width is a gap round something else.
HELD_WITH_A_WIDTH = (HELD_MARKER + " -- the pads settled {gap:.0f} mm apart, and the thing this "
                     "step names measures about {width:.0f} mm across the line they close "
                     "along ({source})")
FROZEN_ENDING = ("the tool has not moved and the jaws have not changed for {} cycles running: "
                 "nothing this step asks about can have changed, so it is NOT met. Something "
                 "other than another look at the same pose has to happen next")
RELEASED_IN_CODE = ("this step is about putting the thing down, the robot measured it as "
                    "standing on the surface below it and over the middle of what it is being "
                    "put on, and nothing had moved for cycles: the jaws were opened here -- to "
                    "the width of what they were holding, not to full travel -- rather than the "
                    "step ending still holding it")
