"""Semantic command in, deltas at the tool out.

    {"type": "move", "direction": "forward", "distance_mm": 60}
"""

from __future__ import annotations

import functools
import math
import os
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from ..schema.actions import (Command, GRIPPER, HOME, ROTATE, TRANSLATE, WAIT,
                             parse_command)
from .transforms import unit
from .types import Capabilities, TcpDelta
from . import types

DIRECTIONS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "schema", "directions.yaml")

DEFAULT_SPEED_M_S = 0.0125
DEFAULT_ROTATION_SPEED_RAD_S = math.radians(8.0)
# How far one CONTACT move may ask for -- a move carrying "stop_on_contact", which is how
# something gets pushed now that "push" is not a type of its own.
PUSH_LIMIT_M = 0.25


class ConversionError(ValueError):
    """A command that parses but cannot be turned into a delta."""


@functools.lru_cache(maxsize=None)
def load_directions() -> Dict[str, Any]:
    """The direction table, read once. Shared: callers must not mutate it."""
    with open(DIRECTIONS_PATH) as handle:
        return yaml.safe_load(handle)


def _axis_for(directions: Dict[str, Any], word: Optional[str],
              explicit, kind: str) -> np.ndarray:
    if explicit is not None:
        return unit(explicit)
    table = directions["rotations" if kind == ROTATE else "directions"]
    aliases = directions.get("aliases") or {}
    canonical = aliases.get(word, word)
    if canonical not in table:
        # Name every word that WOULD have worked, the other spellings included: a refusal that
        # lists only the canonical six leaves a model that wrote "clockwise" guessing at which
        # of yaw_left and yaw_right its own word meant.
        offered = sorted(set(table) | {other for other, to in aliases.items() if to in table})
        raise ConversionError(
            "{!r} is not a {} word; directions.yaml offers {}".format(
                word, "rotation" if kind == ROTATE else "direction", ", ".join(offered))
        )
    value = table[canonical]
    if kind == ROTATE:
        return unit(np.asarray(value["axis"], dtype=float) * float(value.get("sign", 1)))
    return unit(value)


def to_deltas(command, capabilities: Optional[Capabilities] = None,
              directions: Optional[Dict[str, Any]] = None) -> List[TcpDelta]:
    """Parse if needed, then resolve every action into a :class:`TcpDelta`."""
    if not isinstance(command, Command):
        command = parse_command(command)
    directions = directions if directions is not None else load_directions()

    speed_cap = capabilities.max_speed_m_s if capabilities else None
    turn_cap = capabilities.max_rotation_speed_rad_s if capabilities else None

    deltas: List[TcpDelta] = []
    for action in command.actions:
        common = dict(index=action.index, label=action.describe(), note=action.note,
                      stop_on_contact=bool(action.stop_on_contact))

        if action.kind == TRANSLATE:
            contact = bool(action.stop_on_contact)
            distance = action.distance_mm / 1000.0
            if contact and distance > PUSH_LIMIT_M:
                raise ConversionError(
                    "action {}: a move with stop_on_contact may ask for at most {:.0f} mm "
                    "and this one asks for {:.0f}. It carries on until something stops it, "
                    "so the distance is how far it is allowed to go if nothing does -- ask "
                    "for less, and move again if it is still not home."
                    .format(action.index, PUSH_LIMIT_M * 1000, action.distance_mm))
            fallback = ((capabilities.default_speed_m_s if capabilities else None)
                        or DEFAULT_SPEED_M_S)
            # The robot's own default, for a contact move as much as for any other: a slower
            # push is not a gentler one here -- the servo tracks a fraction of what it is
            # given, so half the speed is half the travel per control tick and a segment that
            # runs out of ticks reports a shortfall it never had.
            speed = (action.speed_mm_s / 1000.0) if action.speed_mm_s else fallback
            if speed_cap:
                speed = min(speed, speed_cap)
            if speed <= 0:
                raise ConversionError("action {}: speed must be positive".format(action.index))
            # ``push`` as well as ``stop_on_contact``, because it is ``delta.push`` that both
            # adapters read to decide that ending against something is the RESULT: LIBERO's
            # run_step reads step.detail["push"] and returns CONTACT for it, and LIMITED
            # ("blocked") without it.
            deltas.append(TcpDelta(kind=types.TRANSLATE,
                                   axis=_axis_for(directions, action.direction, action.axis,
                                                  TRANSLATE),
                                   magnitude=distance, speed=speed, push=contact, **common))

        elif action.kind == ROTATE:
            fallback = ((capabilities.default_rotation_speed_rad_s if capabilities else None)
                        or DEFAULT_ROTATION_SPEED_RAD_S)
            speed = (math.radians(action.rot_speed_deg_s) if action.rot_speed_deg_s
                     else fallback)
            if turn_cap:
                speed = min(speed, turn_cap)
            if speed <= 0:
                raise ConversionError("action {}: speed must be positive".format(action.index))
            deltas.append(TcpDelta(kind=types.ROTATE,
                                   axis=_axis_for(directions, action.direction, action.axis,
                                                  ROTATE),
                                   magnitude=math.radians(action.angle_deg), speed=speed,
                                   **common))

        elif action.kind == GRIPPER:
            deltas.append(TcpDelta(kind=types.GRIPPER, gripper_state=action.command, **common))

        elif action.kind == HOME:
            if capabilities is not None and not capabilities.supports_home:
                raise ConversionError(
                    "action {}: {} has no home pose".format(action.index, capabilities.name))
            deltas.append(TcpDelta(kind=types.HOME, **common))

        elif action.kind == WAIT:
            deltas.append(TcpDelta(kind=types.WAIT, seconds=action.seconds, **common))

        else:                                        # unreachable: the schema enumerates them
            raise ConversionError("action {}: unknown kind {!r}".format(action.index, action.kind))

    return _check_totals(deltas)


TOTAL_TRANSLATION_LIMIT_M = 0.4
TOTAL_ROTATION_LIMIT_RAD = math.radians(90.0)


def _check_totals(deltas: List[TcpDelta]) -> List[TcpDelta]:
    """Hold one command to what it may ask for, and hand back the part of it that fits."""
    turn = sum(d.magnitude for d in deltas if d.kind == types.ROTATE)
    if turn > TOTAL_ROTATION_LIMIT_RAD:
        raise ConversionError(
            "the command asks for {:.0f} deg of rotation in total, over the {:.0f} deg one "
            "command may ask for".format(np.degrees(turn), np.degrees(TOTAL_ROTATION_LIMIT_RAD)))
    homed = False
    for delta in deltas:
        if delta.kind == types.HOME:
            homed = True
        elif homed and delta.kind in (types.TRANSLATE, types.ROTATE):
            raise ConversionError(
                "action {} moves after a home in the same command. Where home leaves the "
                "robot is the robot's business, so this cannot be previewed past it -- send "
                "the motion as a second command.".format(delta.index))
    return _what_fits(deltas)


def how_far_fits(here: np.ndarray, axis: np.ndarray, limit: float) -> float:
    """How far along ``axis`` from ``here`` the tool may go and still be inside ``limit`` of
    where the command started."""
    along = float(here @ axis)
    inside = limit * limit - float(here @ here) + along * along
    return max(0.0, math.sqrt(max(0.0, inside)) - along)


#: what a shortened move says to do with the part of it that did not run.
ANOTHER_COMMAND = "send the rest as another command"


def shorten(deltas: List[TcpDelta], index: int, fits: float, said: str,
            advice: str = ANOTHER_COMMAND) -> List[TcpDelta]:
    """Cut the move at ``index`` to ``fits`` metres and drop what was written to follow it."""
    delta = deltas[index]
    dropped = len(deltas) - index - 1
    delta.magnitude = fits
    delta.note = "{}. {} was shortened to {:.0f} mm{}; {}."\
        .format(said, delta.label, fits * 1000,
                "" if not dropped else
                " and the {} action(s) after it were not run".format(dropped), advice)
    delta.label = "{} (shortened to {:.0f} mm: {})".format(delta.label, fits * 1000, said)
    del deltas[index + 1:]
    return deltas


def fit_to_reach(deltas: List[TcpDelta], here: np.ndarray, limit: float,
                 tangent: float = 0.0, arrival: float = 0.0) -> Optional[str]:
    """Hold the moves of one command to a radius around the base, in place: a shortened move
    drops what was written after it."""
    index = 0
    while index < len(deltas):
        delta = deltas[index]
        if delta.kind != TRANSLATE or delta.axis is None:
            index += 1
            continue
        if delta.push:
            break            # where a contact move ends is whatever stops it, not its length
        axis = np.asarray(delta.axis, dtype=float)[:2]
        out, end = float(np.linalg.norm(here)), here + axis * delta.magnitude
        ends = float(np.linalg.norm(end))
        if ends <= max(limit, out + tangent):
            here, index = end, index + 1
            continue
        sideways = float(np.linalg.norm(axis))          # the horizontal part of a slanted move
        # FLOORED to the millimetre a command is written in: a distance rounded up lands
        # outside the radius it was measured against, and the model that sends back exactly
        # what it was advised is refused for obeying.
        fits = float(np.floor(how_far_fits(here, axis / sideways, limit) * 1000.0))
        said = ("{:.0f} mm this way ends {:.0f} mm out from the base, and this arm only goes "
                "where it is sent out to about {:.0f} mm; it is {:.0f} mm out now"
                .format(delta.magnitude * 1000.0, ends * 1000.0, limit * 1000.0, out * 1000.0))
        if fits >= 1.0:
            shorten(deltas, index, fits / 1000.0 / sideways, said,
                    "what is left of it is outside that radius, so do not send it again -- "
                    "come round to the near side of the thing and work from there")
            return None
        if ends <= limit + arrival:
            here, index = end, index + 1
            continue
        return ("{}, so there is nothing of that move left to run. Further out a motion "
                "comes back having travelled a fraction of what it asked for and having "
                "wandered off its line with nothing in its way, which reads exactly like "
                "something stopping it. Come back in and work from closer.".format(said))
    return None


def _what_fits(deltas: List[TcpDelta]) -> List[TcpDelta]:
    """The moves of this command up to the point where they leave the travel limit."""
    limit, here = TOTAL_TRANSLATION_LIMIT_M, np.zeros(3)
    for index, delta in enumerate(deltas):
        if delta.kind != types.TRANSLATE:
            continue
        axis = unit(delta.axis)
        end = here + axis * delta.magnitude
        if float(np.linalg.norm(end)) <= limit + 1e-9:
            here = end
            continue
        asked = _resultant(deltas)
        fits = how_far_fits(here, axis, limit)
        said = ("the moves in this command compose to {:.0f} mm from where the tool stands "
                "-- the straight-line distance of all of them together, which is what the "
                "{:.0f} mm one command may ask for is measured against, not the sum of the "
                "distances written on them".format(asked * 1000, limit * 1000))
        if fits * 1000.0 < 1.0:
            raise ConversionError(
                "{}. The moves before action {} already use all of it, so there is nothing "
                "left for this one: send it as a second command."
                .format(said, delta.index))
        return shorten(deltas, index, fits, said)
    return deltas


def _resultant(deltas: List[TcpDelta]) -> float:
    """How far the composed moves of a command put the tool from where it started."""
    total = np.zeros(3)
    for delta in deltas:
        if delta.kind == types.TRANSLATE:
            total = total + unit(delta.axis) * delta.magnitude
    return float(np.linalg.norm(total))
