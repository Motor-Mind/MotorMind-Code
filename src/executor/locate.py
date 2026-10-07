"""Where is the thing the subgoal names?"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import geometry

# Two rays aimed at one object passed 3-24 mm apart across the probe's five scenes; two aimed at
# different objects passed hundreds of millimetres apart.
MAX_SKEW_MM = 50.0
#: The top edges of two views' boxes are ONE point -- a narrow top -- only this close.
TOP_SKEW_MM = 20.0
#: ...and on a box at least this many times as tall as it is wide in both.
TALL = 1.5
# Below this the wrist is close enough for its own answer to be worth having: the probe measured
# 24 mm at 100 mm above the table.
WRIST_USEFUL_BELOW_MM = 150.0
# Words that name a PART of something rather than the thing itself.
SIDE_PARTS = ("handle", "rim", "edge", "face", "front", "side", "lip", "spout", "grip")
TOP_PARTS = ("knob", "lid", "cap", "button", "top")
#: ...and the SIDE parts that are the top face's own edge -- an opening's rim -- wherever the box
#: drawn round them sits against the box of what they belong to.
TOP_EDGE_PARTS = ("rim", "lip")
#: How far apart two boxes may be and still be one thing and a part of it, as a share of the
#: part's own size.
TOUCHES_SHARE = 0.5
#: ...and how much bigger than the part its body may be, in picture area.
BODY_AT_MOST_TIMES = 25.0
#: How far above the table a part of something standing on it can be.
PART_HEIGHT_LIMIT_MM = 400.0
#: Words that put the thing ON or IN something else, which is what holds a thing up off the
#: table -- "the bowl on the wooden cabinet", "inside the top drawer" -- unless a direction
#: follows rather than a thing: "on the left", "in front of".
HELD_UP_BY = ("on", "in", "inside", "atop", "upon", "within")
_A_DIRECTION = ("left", "right", "front", "side", "middle", "centre", "center", "foreground",
                "background")
#: ...or the table the heights are measured from -- a "surface" can be a shelf's -- or
#: the thing itself: "written on it".
_THE_TABLE = ("table", "tabletop", "floor", "it", "them")


#: Words that may stand between the article and the part word and still leave the phrase
#: headed by the part: "the NEAR side of the bowl", "the FRONT face of the drawer".
PART_MODIFIER = ("near", "far", "front", "back", "left", "right", "top", "bottom", "upper",
                 "lower", "outer", "inner", "near-side", "far-side")
_ARTICLE = ("the", "a", "an", "its", "his", "her", "their", "this", "that")


def heads_with_part(phrase: str) -> str:
    """The part word this phrase is ABOUT, or "" -- an article and one modifier allowed."""
    text = " ".join(str(phrase or "").split())
    head = text.partition(" of ")[0] if " of " in text else text
    words = [w.strip(",.").lower() for w in head.split() if w.strip(",.")]
    while words and words[0] in _ARTICLE:
        words.pop(0)
    if len(words) > 1 and words[0] in PART_MODIFIER:
        words.pop(0)
    # ...and ONE adjective of any kind, where the part word is all that is left after it.
    if len(words) == 2 and a_part_word(words[1]):
        words.pop(0)
    return a_part_word(words[0]) if words else ""


def a_part_word(word: str) -> str:
    """That word as one of the parts, singular -- or "" if it is not one."""
    stem = str(word or "").strip(",.'").lower().rstrip("s")
    for part in SIDE_PARTS + TOP_PARTS:
        if part.rstrip("s") == stem:
            return part
    return ""


def part_in_a_sentence(words: str, target: str = "") -> str:
    """The part a STEP says it will close on, out of its own sentence, or ""."""
    said = [w.strip(",.").lower() for w in str(words or "").split() if w.strip(",.")]
    # The target's HEAD noun phrase, up to its first preposition.
    head: List[str] = []
    for word in str(target or "").split():
        word = word.strip(",.'").lower()
        if word in _PREPOSITION:
            break
        if len(word) > 2 and word not in _ARTICLE and word not in PART_MODIFIER:
            head.append(word.rstrip("s"))
    about = set(head)
    for index, word in enumerate(said[:-1]):
        if said[index + 1] != "of" or not a_part_word(word):
            continue
        whose = {w.rstrip("s") for w in said[index + 2:index + 8]}
        if about and not (about & whose):
            continue              # a part OF something else: where a thing is, not its part
        return a_part_word(word)
    return ""


def part_phrase(sentence: str, until: str) -> str:
    """"the <part> of <what it is part of>" as ``sentence`` says it, up to ``until`` or the end
    of its clause -- the words a camera can box that part by -- or ""."""
    for found in re.finditer(r"\b(?=(\w+) of (.+?)(?:[,;.]| and | then | {}|$))".format(
            re.escape(until)), " ".join(str(sentence or "").lower().split())):
        if a_part_word(found.group(1)):
            return "the {} of {}".format(a_part_word(found.group(1)), found.group(2).strip())
    return ""


#: Words that put whatever follows them in a clause OF THEIR OWN.
_PREPOSITION = ("of", "with", "on", "in", "at", "by", "against", "beside", "near", "from",
                "under", "over", "behind", "inside", "into", "to")


def _ends_with_part(phrase: str) -> str:
    """"the mug handle", "the pot's handle" -- a part named as the head noun of a compound."""
    words = [w.strip(",.").lower() for w in str(phrase or "").split() if w.strip(",.")]
    if len(words) < 3 or any(w in _PREPOSITION for w in words) or words[-2] in _ARTICLE:
        return ""
    return a_part_word(words[-1])


def names_a_part(words: str) -> str:
    """The part word in ``words``, or "" -- the one test of "is this a part of something"."""
    return heads_with_part(words) or _ends_with_part(words)


@dataclass
class Located:
    """Where the target is, and how sure the arithmetic behind that is."""

    point_base: Optional[np.ndarray] = None
    method: str = ""                      # in words, for the history and the claim
    cameras: List[str] = field(default_factory=list)
    label: str = ""
    candidates: List[str] = field(default_factory=list)
    skew_mm: Optional[float] = None
    # Two views that answered about two different things, in words.
    disagreement: str = ""
    # The answer stands but something here argues against it: the views disagreed, its base is
    # hidden, or a part could not be told from its body.
    doubtful: bool = False
    # How far this point may be from the middle of the thing, in millimetres, and in words where
    # that number comes from.
    error_mm: Optional[float] = None
    error_class: str = ""
    # How far a THIRD camera -- the wrist, which is only ever asked from down low -- put the
    # thing from the answer that stands.
    confirmed_mm: Optional[float] = None
    # The chosen box's bottom edge falls inside another candidate's box, so it is that other
    # thing's outline and not where this one stands. See geometry.hiding_the_base.
    occluded: bool = False
    # Where the located box meets the table, in the base frame (metres): two points from an
    # oblique view, four from straight above.
    footprint: Optional[np.ndarray] = None
    # WHICH view measured that outline and on what plane, in words.
    footprint_method: str = ""
    # The box that was chosen, in the pixels of the view that chose it, and that view's own
    # payload.
    box_px: Optional[List[float]] = None
    #: every box the view that answered came back with, as (label, pixels)
    inventory: List[Tuple[str, List[float]]] = field(default_factory=list, repr=False)
    view: Optional[Dict[str, Any]] = field(default=None, repr=False)
    view_name: str = ""
    # Which part of something was asked for, and how far up the answer puts it.
    part: str = ""
    part_z_mm: Optional[float] = None
    # ...and, from two views, where the MIDDLES of their boxes meet: a part on an upright face
    # (a drawer's handle) is where that face's middle is, not where the thing stands.
    part_point: Optional[np.ndarray] = None
    # ...and where the middles of the boxes round the thing it belongs to meet: the part stands
    # out of that thing's face on the side away from this point.
    part_body: Optional[np.ndarray] = None
    # ...and where the middles of the two boxes' TOP edges meet, when they do: how high the
    # thing's top stands, from pictures alone.
    top_point: Optional[np.ndarray] = None
    # Whether ``footprint`` is the PART's own box read on the part's own plane, rather than the
    # whole thing's outline where it meets the table.
    part_outline: bool = False
    # How many views were shown where the caller's own point falls in their picture, and in how
    # many of them the box that came back covers that spot.
    hinted: int = 0
    hint_kept: int = 0
    # Set by the executor when it kept an earlier location for these words instead of this
    # answer: how far apart the two were. Lives here so the run file carries it.
    jumped_mm: Optional[float] = None
    # Other things in the same picture that these words fit as well as the answer does, and that
    # stand closer to it than this answer's own error bar -- so no measurement here can say
    # which of them was meant.
    look_alikes: List[Tuple[str, float]] = field(default_factory=list)
    ambiguous: str = ""
    #: Set when the words asked for a PART and nothing here could tell it from the thing it
    #: belongs to.
    part_unresolved: bool = False
    calls: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.point_base is not None and not self.error

    def as_dict(self) -> Dict[str, Any]:
        return {"point_mm": None if self.point_base is None
                else [round(float(v) * 1000.0, 1) for v in self.point_base],
                "footprint_mm": None if self.footprint is None
                else [[round(float(v) * 1000.0, 1) for v in point[:2]]
                      for point in self.footprint],
                "footprint_method": self.footprint_method,
                "method": self.method, "cameras": list(self.cameras), "label": self.label,
                "candidates": list(self.candidates), "skew_mm": _mm(self.skew_mm),
                "disagreement": self.disagreement, "doubtful": self.doubtful,
                "error_mm": _mm(self.error_mm),
                "error_class": self.error_class, "occluded": self.occluded,
                "confirmed_mm": _mm(self.confirmed_mm),
                "look_alikes": [[label, round(float(mm), 1)]
                                for label, mm in self.look_alikes],
                "ambiguous": self.ambiguous, "part_unresolved": self.part_unresolved,
                "box_px": None if self.box_px is None
                else [round(float(v), 1) for v in self.box_px],
                "view_name": self.view_name, "part": self.part,
                "part_z_mm": _mm(self.part_z_mm), "part_outline": self.part_outline,
                "part_point_mm": None if self.part_point is None
                else [round(float(v) * 1000.0, 1) for v in self.part_point],
                "jumped_mm": _mm(self.jumped_mm),
                "hinted": self.hinted, "hint_kept": self.hint_kept,
                "calls": self.calls, "error": self.error}


def _mm(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(value, 1)


def locate(client, prompts: Dict[str, Any], frames: Dict[str, Any], target_words: str,
           table_z_m: Optional[float] = None, tool_z_m: Optional[float] = None,
           wrist: str = "wrist", top_z_m: Optional[float] = None,
           part: Optional[bool] = None,
           part_word: str = "",
           boxes: Optional[Dict[str, Sequence[float]]] = None,
           last_seen: Optional[Sequence[float]] = None,
           hint_boxes: Optional[Dict[str, Sequence[float]]] = None,
           wrist_only: bool = False, wrist_within_mm: Optional[float] = None,
           held_px: Optional[Sequence[float]] = None,
           tallest_mm: Optional[float] = None) -> Located:
    """Find what ``target_words`` describes and say where it is in the base frame: the fixed
    views give its EXTENT, the wrist from down low its POSITION (see _wrist_position). A wrist
    box no wider than ``held_px`` -- what the jaws hold, in the wrist's pixels -- is that, not
    the thing (a bowl in the jaws was once taken for the plate's middle, 200 mm off)."""
    views = {name: camera for name, camera in (frames or {}).items()
             if camera.get("image") is not None and geometry.usable(camera)}
    fixed = [name for name in views if name != wrist]
    if not fixed:
        return Located(error="no fixed camera with a calibrated pose is available, so a "
                             "pixel in these pictures cannot be turned into a position")

    out = Located()
    # Is this a part of something?
    named = part_word or names_a_part(target_words)
    out.part = named if part is None or part else ""
    readings: List[Tuple[str, Tuple[float, float], List[float],
                        Optional[List[float]]]] = []
    # What each view called the thing, and what else it listed.
    said: Dict[str, Tuple[str, List[Tuple[str, List[float]]]]] = {}
    # The wrist, in two roles that are not the same question.
    close_in = tool_z_m is not None and table_z_m is not None and (tool_z_m - (
        table_z_m if top_z_m is None else top_z_m)) * 1000.0 < WRIST_USEFUL_BELOW_MM
    order = [] if wrist_only else sorted(fixed)
    # Every view that will be asked whatever the others answer is asked at once; the wrist
    # from up high only when no fixed view found it, so after them.
    asked = _ask_at_once(client, prompts, views, [name for name in order + (
        [wrist] if wrist in views and close_in else []) if (boxes or {}).get(name) is None],
                         target_words, last_seen, hint_boxes)

    def reading_of(name):
        given = (boxes or {}).get(name)
        if given is not None:
            return _from_box(views[name], given, target_words, out)
        reading, own = asked[name] if name in asked else _ask_at_once(
            client, prompts, views, [name], target_words, last_seen, hint_boxes)[name]
        _take(out, own, reading)
        return reading

    for name in order:
        reading = reading_of(name)
        said[name] = (out.label, list(out.inventory))
        if reading is not None:
            readings.append((name,) + reading)
    wrist_reading = None
    if wrist in views and (close_in or not readings):
        reading = reading_of(wrist)
        if reading is not None and not (held_px is not None and _longest_px(reading[1]) <= (
                HELD_WIDER * _longest_px(held_px))):
            wrist_reading = (wrist,) + reading

    if not readings and wrist_reading is None:
        if wrist_only:
            out.error = ("the camera on the wrist could not pick out {!r} from over it"
                         .format(target_words.strip()))
            return out
        out.error = out.error or (
            "{!r} is not visible from here: no fixed camera could pick it out{}. Nothing in "
            "these pictures can be reached for until the view changes -- raise the tool so "
            "more of the table is in the wrist view and look again, or move the arm out of "
            "the fixed camera's line of sight -- or say what tells it apart from the things "
            "that ARE in the pictures."
            .format(target_words.strip(),
                    ", and neither could the camera on the wrist" if wrist in views
                    else " and there is no wrist camera to ask"))
        return out

    if len(readings) >= 2:
        # Two FIXED views, triangulated: no table height is needed and the skew between the rays
        # is a free check.
        (name_a, pixel_a, _, _), (name_b, pixel_b, _, _) = readings[0], readings[1]
        boxes = [box for _, _, box, _ in readings[:2]]

        def meet(pixels, within_mm=MAX_SKEW_MM):      # None where they pass further apart
            point, apart = geometry.triangulate(views[name_a], pixels[0], views[name_b], pixels[1])
            return (point if point is not None and apart * 1000.0 <= within_mm else None), apart
        middle, apart = meet([geometry.box_middle(box) for box in boxes])
        agree = middle is not None
        if out.part and agree:
            out.part_point = middle           # the middles, where the bottoms need not agree
            bodies = [_holder_box(said[name][1], box) for name, _, box, _ in readings[:2]]
            if all(body is not None for body in bodies):
                out.part_body = meet([geometry.box_middle(body) for body in bodies])[0]
        point, skew = geometry.triangulate(views[name_a], pixel_a, views[name_b], pixel_b)
        crossing = "" if point is None or skew * 1000.0 > MAX_SKEW_MM else \
            _crossed_where_nothing_stands(views, readings[0], readings[1], point, table_z_m,
                                          top_z_m, bool(out.part) or held_up(target_words))
        # A bottom edge something in front supplied is not this thing's own: where the middles
        # meet is (gpt6sol 104, twice: the bottoms 55 mm apart or read onto the table apart,
        # the middles on the raised top it went on, one view on the table 150 mm short of it).
        # A top edge is ONE point in both views only on a thing taller than it is wide; over
        # a bowl each view's top edge is its far rim, and they meet high over the middle.
        if all(abs(box[3] - box[1]) >= TALL * abs(box[2] - box[0]) for box in boxes):
            out.top_point = meet([((box[0] + box[2]) / 2.0, min(box[1], box[3]))
                                  for box in boxes], TOP_SKEW_MM)[0]
        by_middles = agree and (point is None or skew * 1000.0 > MAX_SKEW_MM or bool(crossing)) \
            and any(hidden is not None for _, _, _, hidden in readings[:2])
        if by_middles:
            point, skew, crossing = middle, apart, ""
        if point is not None:
            out.skew_mm = skew * 1000.0
            if out.skew_mm <= MAX_SKEW_MM and not crossing:
                # Two views that answered, so the error bar is MEASURED: how far apart the two
                # rays passed.
                out.error_mm = max(geometry.CENTRED_ERROR_MM, out.skew_mm)
                out.error_class = "how far apart the two views that answered passed"
                if table_z_m is not None:
                    # Outlined where the rays met -- what the thing stands on, which is
                    # never below the table -- and not on the table under a thing held up.
                    out.footprint = geometry.box_on_plane_corners(
                        views[name_a], readings[0][2], max(float(table_z_m), float(point[2])),
                        top_z_m)
                    out.footprint_method = _outline_method(views[name_a], name_a, top_z_m)
                out.point_base, out.cameras = point, [name_a, name_b]
                out.method = "{} + {} triangulated{}".format(name_a, name_b, (
                    ", by the middles of their boxes: the bottom edge of one is something in "
                    "front of it") if by_middles else "")
                # The footprint above is measured in name_a's pixels, so name_a's reading is
                # the one this answer is made of and its words are the ones that describe it.
                _adopt(out, said, name_a)
                # ...not while holding: over the receiver the wrist sees mostly the payload, and two
                # fixed views that agree are the better answer (rig/port: bowls 34-82 mm off plates)
                if wrist_reading is not None and held_px is None:
                    _wrist_position(out, views[wrist], wrist, wrist_reading[2], table_z_m,
                                    top_z_m, last_seen, wrist_within_mm)
                return out
            # Two fixed views that disagree are probably looking at two different things, and
            # averaging two objects gives a point on neither: the reading kept is the one
            # nearer where the thing was found before.
            out.doubtful = True
            readings = [_nearest_to(views, readings[:2], last_seen, table_z_m, top_z_m)]
            out.disagreement = (crossing or (
                "the {} and {} views disagree: their rays pass {:.0f} mm apart, over the "
                "{:.0f} mm two views of one object stay within, so they are probably looking "
                "at different things".format(name_a, name_b, out.skew_mm, MAX_SKEW_MM))) + (
                " -- the {} view's answer is the one below".format(readings[0][0]))

    if table_z_m is None:
        out.error = ("only the {} view answered and this robot has no commissioned table "
                     "height, so a pixel cannot be turned into a position"
                     .format(readings[0][0] if readings else wrist))
        return out

    # The fixed view's own answer, preferred over the wrist's wherever it exists; the wrist's
    # when nothing fixed could see it, and then with no hidden base to correct for.
    name, _, box_px, hidden = readings[0] if readings else wrist_reading[:3] + (None,)
    view = views[name]
    # With a measured height the MIDDLE of the box is the thing's centroid and reads onto
    # the plane half way up it exactly; without one the bottom edge and its constant is all
    # there is, and where something in front has supplied that bottom edge the box is not
    # the thing's outline at all, so the correction below has to keep its own rule.
    point = geometry.box_on_plane(view, box_px, float(table_z_m),
                                  None if hidden is not None else top_z_m)
    if point is None:
        out.error = "the {} view's ray does not meet the table plane".format(name)
        return out
    # Held to where it was found before as _wrist_position holds it: nothing checks it here.
    if name == wrist and last_seen is not None and wrist_within_mm is not None \
            and _apart_mm(point, last_seen) > wrist_within_mm:
        out.error = ("the {} view put it {:.0f} mm from where it was found before, further "
                     "than the jaws span".format(wrist, _apart_mm(point, last_seen)))
        return out
    # From straight down an outline needs its top face's height: without one the wrist's box is
    # laid on the table, bigger than the thing (a wine bottle read 90 mm against a true 45).
    out.footprint = None if name == wrist and top_z_m is None and geometry.looking_down(view) \
        else geometry.box_on_plane_corners(view, box_px, float(table_z_m), top_z_m)
    out.footprint_method = "" if name == wrist and out.footprint is None else \
        _outline_method(view, name, top_z_m)
    out.error_mm, out.error_class = geometry.plane_error_mm(view, box_px, float(table_z_m),
                                                            top_z_m, tallest_mm)
    if name == wrist and not close_in:
        # The wrist is the only camera that could name the target and the tool is still a long
        # way up, which is the one arrangement this module has measured as bad: from there its
        # answer disagreed with the fixed view's by 81 to 534 mm on six missions of six.
        out.error_mm = max(out.error_mm or 0.0, geometry.UNCHECKED_FROM_HEIGHT_MM)
        out.error_class = ("the only view that could see it, from up high, with nothing to "
                           "check it against")
    out.point_base, out.cameras = point, [name]
    out.box_px, out.view, out.view_name = list(box_px), view, name
    # Before _look_alikes, which reads out.inventory through out.view: both have to be
    # this view's or it is measuring one camera's boxes with another's calibration.
    _adopt(out, said, name)
    out.method = "{} view on the table plane".format(name) if name != wrist else \
        "{} view alone on the table plane -- {}".format(
            wrist, "asked on its own, from over the thing and close to it" if wrist_only
            else "no fixed camera could see {!r} from here, so nothing checks this one"
                 .format(target_words.strip()))
    if out.disagreement:
        out.method += ", the two fixed views having disagreed {}".format(
            "by {:.0f} mm".format(out.skew_mm or 0.0) if (out.skew_mm or 0.0) > MAX_SKEW_MM
            else "about where it stands")
    if name != wrist:
        _look_alikes(out, target_words, float(table_z_m))
    if hidden is not None and not geometry.looking_down(view):
        # From straight above there is nothing to correct: the whole footprint
        # lies under the box and its middle is the answer already, so a thing in
        # front of it cannot have supplied the edge that was measured.
        _base_out_of_sight(view, point, hidden, float(table_z_m), out)
    if name != wrist and wrist_reading is not None and _wrist_position(
            out, views[wrist], wrist, wrist_reading[2], table_z_m, top_z_m, last_seen,
            wrist_within_mm):
        return out
    # Last, because the test a part has to pass is against this view's error bar and the
    # bar is not final until everything above has had its say.
    return _as_a_part(out, view, box_px, target_words, float(table_z_m), top_z_m)


def held_up(words: str) -> bool:
    """Do these words put the thing on or in something, rather than beside it?"""
    said = [w.strip(",.;:()'\"").lower() for w in str(words or "").split()]
    said = [w for w in said if w]
    for index, word in enumerate(said[:-1]):
        after = [w for w in said[index + 1:index + 4] if w not in _ARTICLE][:2]
        if word in HELD_UP_BY and after and after[0] not in _A_DIRECTION \
                and not set(after) & set(_THE_TABLE):
            return True
    return False


def _crossed_where_nothing_stands(views: Dict[str, Any], first, second, point,
                                  table_z_m: Optional[float], top_z_m: Optional[float],
                                  held: bool) -> str:
    """Why two rays that DID meet are still not on one thing, in words -- or ""."""
    if table_z_m is None:
        return ""
    (name_a, _, box_a, _), (name_b, _, box_b, _) = first, second
    on_a = geometry.box_on_plane(views[name_a], box_a, float(table_z_m))
    on_b = geometry.box_on_plane(views[name_b], box_b, float(table_z_m))
    if on_a is None or on_b is None:
        return ""
    apart_mm = float(np.linalg.norm((on_a - on_b)[:2])) * 1000.0
    up_mm = (float(point[2]) - float(table_z_m)) * 1000.0
    # A small skew only says the rays meet SOMEWHERE. Read onto the table, one thing standing
    # on it lands in one place from both views (measured 1-54 mm apart); a thing held up lands
    # apart, by an amount that grows with the height the rays meet at, and that height is only
    # believable if something says what holds it there -- two views on two look-alikes can
    # meet well above the table with a skew of a few millimetres.
    highest = PART_HEIGHT_LIMIT_MM if held else max(
        MAX_SKEW_MM, 0.0 if top_z_m is None else (float(top_z_m) - float(table_z_m)) * 1000.0)
    if apart_mm <= MAX_SKEW_MM or 0.0 <= up_mm <= highest:
        return ""
    return ("the {} and {} views disagree: their rays meet {:.0f} mm {} the table, where "
            "{}, and read onto the table the two boxes are {:.0f} mm apart, so they are "
            "probably two different things"
            .format(name_a, name_b, abs(up_mm), "above" if up_mm > 0 else "below",
                    "nothing standing on it can be" if up_mm < 0 else
                    "nothing in the words holds it up", apart_mm))


def _nearest_to(views, readings, last_seen, table_z_m, top_z_m):
    """Of two readings that disagree, the one read onto the table nearer where the thing was
    found before -- the first, with nothing found before to hold them to."""
    if last_seen is None or table_z_m is None:
        return readings[0]
    at = [geometry.box_on_plane(views[name], box, float(table_z_m),
                                None if hidden is not None else top_z_m)
          for name, _, box, hidden in readings]
    return min(zip(readings, at), key=lambda pair: float("inf") if pair[1] is None
               else _apart_mm(pair[1], last_seen))[0]


def _wrist_position(out: Located, camera: Dict[str, Any], name: str,
                    box_px: Sequence[float], table_z_m: Optional[float],
                    top_z_m: Optional[float], last_seen, within_mm: Optional[float]) -> bool:
    """The two cameras apart is the error bar that costs nothing; the wrist's point is the
    answer within ``within_mm`` of where it was last seen -- or, with nothing seen before, within
    the fixed view's own error bar if that is wider: a close-in wrist box is not overruled by a
    point known no better than the distance between them (the rig: 165 of 1147 target looks
    thrown away against one far view). The outline stays the fixed view's."""
    if top_z_m is None and geometry.looking_down(camera):
        return False          # a box seen from above with no top measured is laid on the table
    at = geometry.box_on_plane(camera, box_px, float(table_z_m), top_z_m)
    if at is None:
        return False
    if last_seen is None and within_mm is not None:
        within_mm = max(within_mm, float(out.error_mm or 0.0))
    gap_mm = _apart_mm(at, out.point_base)
    out.error_mm = max(geometry.CENTRED_ERROR_MM, gap_mm)
    out.error_class = "how far apart the two views that answered put it"
    from_anchor_mm = _apart_mm(at, out.point_base if last_seen is None else last_seen)
    said = (name, gap_mm, from_anchor_mm, "found before" if last_seen is not None else "located")
    if within_mm is None or from_anchor_mm > within_mm:
        out.confirmed_mm = gap_mm
        out.method += ("; the {} view put it {:.0f} mm from there and {:.0f} mm from where it "
                       "was {}, and was not used".format(*said))
        return False
    out.point_base, out.cameras, out.confirmed_mm = at, [name], from_anchor_mm
    out.method += ("; so the {} view's answer was taken instead -- from over it the wrist "
                   "gives the position, and it put the thing {:.0f} mm from where the fixed "
                   "view did and {:.0f} mm from where it was {}".format(*said))
    return True


#: A wrist box must be this much wider than what the jaws hold to be something else.
HELD_WIDER = 1.3


def _longest_px(box: Sequence[float]) -> float:
    return max(abs(float(box[2]) - float(box[0])), abs(float(box[3]) - float(box[1])))


def _apart_mm(point, other) -> float:
    """How far apart two points are across the table, in millimetres."""
    return float(np.linalg.norm((np.asarray(point, dtype=float)
                                 - np.asarray(other, dtype=float))[:2])) * 1000.0


def _outline_method(camera: Dict[str, Any], name: str, top_z_m: Optional[float]) -> str:
    """Which view measured the outline and on which plane, in one phrase for the history."""
    if geometry.looking_down(camera):
        return ("the {} view looking straight down, measured where the box read on the table "
                "and on the top face overlap".format(name)) if top_z_m is not None else \
            ("the {} view looking straight down at a top face whose height nothing has "
             "measured, laid on the table".format(name))
    return ("the {} view from the side, where only the bottom edge of the box touches the "
            "table, so it says how wide the thing is and not how deep".format(name))


def _base_out_of_sight(camera: Dict[str, Any], point, hidden: Sequence[float], z_m: float,
                       out: Located) -> None:
    """The chosen box's bottom edge is another object's outline: widen the error bar to the
    strip the hidden base could be in. The point itself stands."""
    out.occluded, out.doubtful = True, True
    front = geometry.footprint_on_plane(camera, hidden, z_m)
    eye = np.asarray(camera["cam2base"], dtype=float)[:3, 3]
    said = ("; BUT the bottom edge of that box falls inside another thing's box, so it is "
            "that thing's outline and not where this one stands")
    if front is None or float(np.linalg.norm((point - eye)[:2])
                              - np.linalg.norm((front - eye)[:2])) <= 0.0:
        out.method += said + " -- the position is a guess at a hidden base"
        out.error_mm = max(out.error_mm or 0.0, geometry.PLANE_ERROR_FLOOR_MM * 2.0)
        return
    span_mm = float(np.linalg.norm((point - front)[:2])) * 1000.0
    out.error_mm = max(out.error_mm or 0.0, span_mm / 2.0)
    out.error_class = "the width of the strip its base could be hiding in"
    out.method += (said + ", and the base is somewhere between that thing and the lowest of "
                   "this one anyone can see -- a strip {:.0f} mm deep, so the position above "
                   "may be anywhere in it".format(span_mm))


# --------------------------------------------------------------------------- asking

def _adopt(out: Located, said: Dict[str, Tuple[str, List[Tuple[str, List[float]]]]],
           name: str) -> None:
    """Take the label and inventory of the view whose geometry this answer is made of."""
    if name in said:
        out.label, out.inventory = said[name][0], list(said[name][1])


def _from_box(camera: Dict[str, Any], box: Sequence[float], target_words: str, out: Located):
    """The same reading :func:`_point_at` returns, out of a box somebody already drew."""
    width, height = int(camera["width"]), int(camera["height"])
    if not out.label:
        out.label = target_words.strip()
    box_px = geometry.box_to_px(box, width, height)
    return (geometry.to_pixels(geometry.box_aim(camera, box), width, height), box_px, None)


def _hint(camera: Dict[str, Any], name: str, last_seen, boxes) -> Tuple[str, Optional[List[float]]]:
    """What to tell a view that is being asked AGAIN about something already found once."""
    spot = None
    pixel = None if last_seen is None else geometry.project(camera, last_seen)
    if pixel is not None:
        spot = [pixel[0] * geometry.NORMALISED_SPAN / float(camera["width"]),
                pixel[1] * geometry.NORMALISED_SPAN / float(camera["height"])]
        if not all(0.0 <= v <= geometry.NORMALISED_SPAN for v in spot):
            spot = None
    box = (boxes or {}).get(name)
    if box is not None and (spot is None or (box[0] <= spot[0] <= box[2]
                                             and box[1] <= spot[1] <= box[3])):
        spot = list(geometry.box_middle(box))
        where = "inside the box [{:.0f}, {:.0f}, {:.0f}, {:.0f}]".format(*box[:4])
    elif spot is not None:
        where = "at about ({:.0f}, {:.0f})".format(*spot)
    else:
        return "", None
    return ("WHERE IT WAS LAST SEEN\n"
            "Something already looked once and put the thing being looked for {} of this "
            "picture, in the same coordinates you answer in. It is asked again because the "
            "scene may have changed or that answer gone stale, not because it was wrong. "
            "Unless it has plainly moved, choose THAT SAME object, the one nearest that spot, "
            "not a look-alike -- but it only chooses among things whose printing you cannot "
            "read; printing naming another product overrides it.".format(where), spot)


def _ask_at_once(client, prompts: Dict[str, Any], views: Dict[str, Any], names: List[str],
                 target_words: str, last_seen, hint_boxes) -> Dict[str, Tuple[Any, Located]]:
    """Each named view asked -- all at the same time, by a client that can take that --
    each into a Located of its own."""
    own = {name: Located() for name in names}
    workers = len(names) if getattr(client, "concurrent", False) else 1

    def ask(name):
        return _point_at(client, prompts, views[name], target_words, own[name],
                         hint=_hint(views[name], name, last_seen, hint_boxes))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        replies = list(pool.map(ask, names))
    return {name: (reply, own[name]) for name, reply in zip(names, replies)}


def _take(out: Located, own: Located, reading) -> None:
    """What one view's ask wrote into its own Located, written into ``out`` -- as asking it
    straight into ``out`` would have, in the order the views are read."""
    out.calls += own.calls
    out.hinted += own.hinted
    out.hint_kept += own.hint_kept
    out.candidates += [label for label in own.candidates if label not in out.candidates]
    if reading is not None:
        out.label, out.inventory = own.label, own.inventory


def _point_at(client, prompts: Dict[str, Any], camera: Dict[str, Any],
              target_words: str, out: Located,
              hint: Tuple[str, Optional[List[float]]] = ("", None)):
    """One camera: ask for every object and choose the target."""
    width, height = int(camera["width"]), int(camera["height"])
    # The size the picture is SENT at, which the model sees -- not the render's.
    shrink = min(1.0, float(getattr(client, "image_side", 0) or 1e9) / max(width, height))
    prompt = prompts["locate"]["text"].format(width=round(width * shrink),
                                              height=round(height * shrink),
                                              target=target_words.strip(), hint=hint[0])
    reply = client.ask_json(prompt, images=[camera["image"]],
                            system=prompts["locate"]["system"])
    out.calls += 1
    objects, chosen = _read(reply.data)
    if not objects:
        return None
    for label, _ in objects:
        if label not in out.candidates:
            out.candidates.append(label)
    box = _box_for(objects, chosen)
    if box is None:
        return None
    out.label = chosen
    if hint[1] is not None:
        out.hinted += 1
        out.hint_kept += int(box[0] <= hint[1][0] <= box[2] and box[1] <= hint[1][1] <= box[3])
    hidden = geometry.hiding_the_base(box, [other for label, other in objects
                                            if label != chosen])
    box_px = geometry.box_to_px(box, width, height)
    hidden_px = None if hidden is None else geometry.box_to_px(hidden, width, height)
    # The whole inventory, in this view's pixels.
    out.inventory = [(label, geometry.box_to_px(other, width, height))
                     for label, other in objects]
    return geometry.to_pixels(geometry.box_aim(camera, box), width, height), box_px, hidden_px


_IGNORED = {"the", "a", "an", "of", "on", "in", "and", "with", "box", "can", "one",
            "object", "thing", "item", "left", "right", "front", "back", "near", "far"}


def _shares_a_word(label: str, words: str) -> bool:
    """Could a reader of ``words`` mean this label too?"""
    wanted = {w for w in _tokens(words) if w not in _IGNORED}
    return bool(wanted & {w for w in _tokens(label) if w not in _IGNORED})


def _tokens(text: str) -> List[str]:
    return ["".join(c for c in word if c.isalnum())
            for word in str(text).lower().replace("_", " ").split()]


#: What the sentence about two look-alikes starts with.
AMBIGUOUS = "TWO THINGS HERE MATCH THOSE WORDS"


#: How much of the smaller box has to lie inside the larger one for the two to be ONE thing -- a
#: mug and its handle, a pot and its lid -- rather than two things standing close together.
INSIDE_SHARE = 0.5
#: ...and how much smaller than the larger box the smaller one has to be to be a PART of it
#: rather than its neighbour. A handle, a lid, a knob is a fraction of the thing it belongs to.
PART_AT_MOST = 0.6


def _one_thing(one: Sequence[float], two: Sequence[float]) -> bool:
    """Is the smaller of these two boxes INSIDE the other?"""
    a = (min(one[0], one[2]), min(one[1], one[3]), max(one[0], one[2]), max(one[1], one[3]))
    b = (min(two[0], two[2]), min(two[1], two[3]), max(two[0], two[2]), max(two[1], two[3]))
    across = min(a[2], b[2]) - max(a[0], b[0])
    down = min(a[3], b[3]) - max(a[1], b[1])
    if across <= 0 or down <= 0:
        return False
    areas = sorted([(a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])])
    smaller, larger = areas[0], areas[1]
    if smaller <= 0 or smaller > PART_AT_MOST * larger:
        # Two boxes of much the same size are two things of much the same size, however much
        # they overlap: things standing side by side overlap in the picture and are still two.
        return False
    return (across * down) / smaller >= INSIDE_SHARE


def _look_alikes(out: Located, target_words: str, table_z_m: float) -> None:
    """Fill in the OTHER things these words fit, when no measurement here can rule them out."""
    if out.view is None or out.box_px is None or out.point_base is None:
        return
    bar = out.error_mm if out.error_mm is not None else geometry.PLANE_ERROR_FLOOR_MM
    here = np.asarray(out.point_base, dtype=float)[:2]
    for label, box in out.inventory or []:
        if not label or label == out.label or len(box) < 4:
            continue
        if not _shares_a_word(label, target_words) or _one_thing(box, out.box_px):
            continue
        point = geometry.box_on_plane(out.view, box[:4], float(table_z_m))
        if point is None:
            continue
        apart = float(np.linalg.norm(np.asarray(point, dtype=float)[:2] - here)) * 1000.0
        if apart <= bar:
            out.look_alikes.append((str(label), apart))
    if not out.look_alikes:
        return
    out.ambiguous = (
        "{}: the answer is {!r}, and {} in the same picture -- {} -- which these words fit "
        "just as well and which nothing here can tell from it, because {} stands {} and this "
        "position is only known to about {:.0f} mm. Say which one is meant by something that "
        "is true of ONE of them: what is written on it, its colour, its shape, or what it is "
        "standing next to."
        .format(AMBIGUOUS, out.label or target_words.strip(),
                "another thing" if len(out.look_alikes) == 1 else "other things",
                ", ".join("{!r} {:.0f} mm away".format(label, mm)
                          for label, mm in out.look_alikes),
                "it" if len(out.look_alikes) == 1 else "the nearest of them",
                "{:.0f} mm away".format(min(mm for _, mm in out.look_alikes)),
                bar))


def _read(data: Any):
    """The objects and the chosen label, from a reply."""
    if isinstance(data, list):
        data = {"objects": data}
    if not isinstance(data, dict):
        return [], ""
    objects: List[Tuple[str, List[float]]] = []
    for entry in data.get("objects") or []:
        if not isinstance(entry, dict):
            continue
        box = entry.get("bbox_2d") or entry.get("bbox") or entry.get("box_2d")
        label = str(entry.get("label") or entry.get("name") or "").strip()
        numbers = _numbers(box)
        if label and len(numbers) >= 4:
            objects.append((label, numbers[:4]))
    chosen = str(data.get("target") or "").strip()
    if objects and not _box_for(objects, chosen):
        chosen = ""
    return objects, chosen


def _box_for(objects: List[Tuple[str, List[float]]], label: str) -> Optional[List[float]]:
    if not label:
        return None
    for name, box in objects:
        if name.strip().lower() == label.strip().lower():
            return box
    return None


def _numbers(value: Any) -> List[float]:
    if isinstance(value, dict):
        value = [value.get(k) for k in ("x1", "y1", "x2", "y2")]
    if not isinstance(value, (list, tuple)):
        return []
    out = []
    for item in value:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            return []
    return out


# ------------------------------------------------------------------ a part of something

def _as_a_part(out: Located, camera: Dict[str, Any], box_px: Sequence[float],
               target_words: str, table_z_m: float, top_z_m: Optional[float]) -> Located:
    """Re-read a finished location as a PART, when the words asked for one."""
    if not out.part or out.point_base is None or out.error:
        return out
    point, why, plane, body = _part_reading(camera, box_px, out.part, table_z_m, top_z_m, out,
                                            target_words)
    if point is None:
        # NOT an error.
        whole = None if body is None else geometry.box_on_plane(camera, body, float(table_z_m),
                                                               top_z_m)
        if whole is not None:
            out.point_base = whole
        out.part_unresolved = True
        out.doubtful = True
        out.method += "; asked for as a PART and not resolved: " + why
        return out
    out.point_base = point
    out.method += "; read as a PART: {}".format(why)
    # How wide the PART is, off the part's own box -- from straight above, on the top face the
    # robot measured, which is the highest thing a box drawn from over it can be round: read
    # lower, a stove knob's grip came back 90 mm across and was aimed at as a wall (task 107).
    outline = _part_outline(camera, box_px, ("flat", float(top_z_m)) if plane[0] == "flat"
                            and top_z_m is not None else plane)
    if outline is not None:
        out.footprint, out.part_outline = outline, True
        out.footprint_method = ("the box drawn round the part itself, read on {}"
                                .format("the face it sits on" if plane[0] == "flat"
                                        else "the upright plane through the thing it "
                                             "belongs to"))
    return out


def _part_outline(camera: Dict[str, Any], box_px: Sequence[float], plane):
    """The part's own box as points in the base frame, on the plane the part was read on."""
    if plane is None:
        return None
    x1, y1, x2, y2 = (float(v) for v in box_px[:4])
    corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    if plane[0] == "flat":
        points = [geometry.on_plane(camera, u, v, float(plane[1])) for u, v in corners]
    else:
        points = [geometry.on_upright_plane(camera, u, v, plane[1]) for u, v in corners]
    if any(point is None for point in points):
        return None
    return np.asarray(points, dtype=float)


#: How far down into the body's box a part may start and still count as sitting ON it, as a
#: share of the body's own height in the picture.
ON_TOP_SHARE = 0.25


def _sits_on_top(part: Sequence[float], body: Sequence[float]) -> bool:
    """Is this part's box sitting ON the body's box, rather than drawn down inside it?"""
    part_bottom = max(float(part[1]), float(part[3]))
    body_top, body_bottom = min(float(body[1]), float(body[3])), max(float(body[1]),
                                                                    float(body[3]))
    return part_bottom <= body_top + ON_TOP_SHARE * (body_bottom - body_top)


def _holder_box(inventory: Sequence[Tuple[str, Sequence[float]]],
                box_px: Sequence[float]) -> Optional[List[float]]:
    """The biggest box in the same reply with the part's middle inside it: what the part
    stands out of, however big -- a cabinet, not the drawer front drawn round the handle."""
    u, v = geometry.box_middle(box_px)
    held = [[min(o[0], o[2]), min(o[1], o[3]), max(o[0], o[2]), max(o[1], o[3])]
            for _, o in inventory or [] if min(o[0], o[2]) <= u <= max(o[0], o[2])
            and min(o[1], o[3]) <= v <= max(o[1], o[3])]
    return max(held, key=lambda o: (o[2] - o[0]) * (o[3] - o[1]), default=None)


def _body_box(inventory: Sequence[Tuple[str, Sequence[float]]], box_px: Sequence[float],
              words: str = "") -> Optional[List[float]]:
    """The box of the thing the chosen box is a part OF, out of the same reply."""
    x1, y1, x2, y2 = (float(v) for v in box_px[:4])
    left, right, top, bottom = min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)
    margin = TOUCHES_SHARE * max(right - left, bottom - top)
    best, best_rank = None, None
    for label, other in inventory or []:
        ox1, oy1, ox2, oy2 = (float(v) for v in other[:4])
        o_left, o_right = min(ox1, ox2), max(ox1, ox2)
        o_top, o_bottom = min(oy1, oy2), max(oy1, oy2)
        area = (o_right - o_left) * (o_bottom - o_top)
        mine = (right - left) * (bottom - top)
        if area <= mine or area > BODY_AT_MOST_TIMES * mine:
            # No bigger than the part is not a body; hugely bigger is the furniture it stands
            # against, and reading a part against THAT puts it wherever the furniture is.
            continue
        if o_left > right + margin or o_right < left - margin \
                or o_top > bottom + margin or o_bottom < top - margin:
            continue                                  # nowhere near it: a different thing
        # Two things can be behind a part in the picture and only one of them is what it is part
        # OF.
        named = _shares_a_word(label, words)
        rank = (0 if named else 1, area)
        if best_rank is None or rank < best_rank:
            best, best_rank = [o_left, o_top, o_right, o_bottom], rank
    return best


def _part_reading(camera: Dict[str, Any], box_px: Sequence[float], part: str,
                  table_z_m: float, top_z_m: Optional[float], out: Located,
                  words: str = ""):
    """Where the PART in this box is, whether it was told apart from its body, and that body."""
    middle = geometry.box_middle(box_px)
    body = _body_box(out.inventory, box_px, words)
    if body is None:
        # Nothing else in the picture is attached to this box.
        return None, ("the part was not resolved: the picture came back with one box under "
                      "the part's name and nothing it could be a part OF, which is what a "
                      "box round the WHOLE thing looks like"), None, body
    # The plane stands where the body STANDS, which is what box_on_plane answers and what
    # its footprint correction is calibrated for.
    anchor = geometry.box_on_plane(camera, body, float(table_z_m), top_z_m)
    # WHICH plane the part is read on is a fact about where it sits, measured from the two
    # boxes -- except a rim, which IS the top face's edge: from over it its box is the whole
    # opening, drawn down inside whatever it stands on, and read lower it comes back wide.
    on_a_top_face = part in TOP_EDGE_PARTS or _sits_on_top(box_px, body)
    if geometry.looking_down(camera):
        # From straight down the answer is read on a HORIZONTAL plane, whichever face the part
        # is on, and the only question is how high that plane is.
        if on_a_top_face and top_z_m is not None:
            face_z, how = float(top_z_m), ("read from straight above, on the top face this "
                                           "robot measured, which is the face its own box "
                                           "sits on")
        elif top_z_m is not None:
            face_z = (float(table_z_m) + float(top_z_m)) / 2.0
            how = ("read from straight above, halfway up the face of the thing it belongs to: "
                   "its box is drawn down inside that thing's own box rather than sitting on "
                   "top of it, and from here the height barely moves the answer")
        else:
            face_z, how = float(table_z_m), ("read from straight above, on the table: nothing "
                                             "has measured how far up the face it is on goes")
        point = geometry.on_plane(camera, middle[0], middle[1], face_z)
        plane = ("flat", face_z)
    else:
        point = None if anchor is None else geometry.on_upright_plane(camera, *middle, anchor)
        plane = ("upright", anchor)
        how = ("read onto the upright plane standing through the thing it belongs to"
               + ("" if on_a_top_face else ", because its box is drawn down inside that thing's "
                                           "own box rather than sitting on top of it"))
    if anchor is None or point is None:
        return None, "it or its body meets no plane this view can measure on", None, body
    bar = out.error_mm if out.error_mm is not None else geometry.PLANE_ERROR_FLOOR_MM
    across_mm = float(np.linalg.norm((point - anchor)[:2])) * 1000.0
    up_mm = (float(point[2]) - float(table_z_m)) * 1000.0
    if part not in TOP_PARTS and across_mm <= bar:
        return None, ("the part was not resolved: it came back {:.0f} mm from the middle of "
                      "the thing it belongs to, inside the {:.0f} mm this view can tell "
                      "apart, and a {} is not in the middle of it"
                      .format(across_mm, bar, part)), None, body
    if not 0.0 <= up_mm <= PART_HEIGHT_LIMIT_MM:
        return None, ("the part was not resolved: reading it against the thing it seems to "
                      "belong to puts it {:.0f} mm above the table, which is not somewhere a "
                      "part of something standing on this table can be -- the two boxes are "
                      "not one thing and a part of it".format(up_mm)), None, body
    out.part_z_mm = up_mm
    return point, "{}, {:.0f} mm up and {:.0f} mm from the body's own point".format(
        how, up_mm, across_mm), plane, body
