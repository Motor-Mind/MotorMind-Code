"""What the planner is allowed to say, and nothing more."""

from __future__ import annotations

import re
from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator, \
    model_validator

from src.executor.models import wants_holding


def _word(value):
    """Enum fields arrive as free text; normalise the shape, never the meaning."""
    return value.strip().lower() if isinstance(value, str) else value


def _tokens(text):
    return re.findall(r"[a-z']+", str(text or "").lower())


# --------------------------------------------------------------------------- what the words mean
#
# Small word tests over a sentence the planner wrote. "Is this about the payload" is one
# question with one answer for the whole system, so it is asked of the executor's copy
# (:func:`src.executor.models.wants_holding`) rather than of a second list of words here.

#: What a subgoal about a CONTROL that rotates says -- a knob, a tap, a dial, a lever that
#: swings.
_TURNING = ("turn", "turns", "turned", "turning", "rotate", "rotates", "rotated", "rotating",
            "twist", "twists", "twisted", "twisting", "wind", "winds", "unscrew", "screw")



def wants_turn(text: str) -> bool:
    """True when the step is about TURNING something -- read from the planner's own sentence."""
    return any(word in _tokens(text) for word in _TURNING)


#: The beginnings of every word that moves the ARM.
_MOTION = ("mov", "reach", "descen", "lower", "rais", "lift", "rise", "risen", "rose", "come",
           "cam", "bring", "brought", "get", "got", "retreat", "withdraw", "approach", "look",
           "carr", "push", "pull", "press", "slid", "turn", "rotat", "travel", "align",
           "hover", "advanc", "cross", "shift", "swing", "tilt", "yaw", "put", "drop",
           "centre", "center", "position", "reposition", "drive", "sweep", "nudge", "insert")
#: ...and the short ones that have to match whole, because their first letters are the start
#: of words that move nothing: "complete", "goal", "in place", "this step".
_MOTION_WORD = ("go", "goes", "going", "went", "back", "away", "toward", "towards", "across")

#: A noun that means the jaws themselves, and a verb that only they can do.
_JAW_NOUN = ("jaw", "jaws", "gripper", "grippers", "gripper's", "finger", "fingers", "grip")
_JAW_VERB = ("open", "opens", "opened", "close", "closes", "closed", "shut", "release",
             "releases", "releasing", "squeeze", "squeezes", "loosen", "loosens", "widen",
             "spread", "let")


def derive_gripper_only(subgoal: str, criterion: str = "") -> bool:
    """Is this step the jaws alone, with the arm already where it needs to be?"""
    said, checked = _tokens(subgoal), _tokens(criterion)
    if wants_holding(criterion):
        return False                      # a grasp, however its sentence reads
    if not any(word in said for word in _JAW_NOUN):
        return False                      # "close the drawer" is not a step about the jaws
    if not any(word in said for word in _JAW_VERB):
        return False
    words = said + checked
    if any(token in _MOTION_WORD for token in words):
        return False
    return not any(token.startswith(stem) for token in words for stem in _MOTION)


#: Words for a PART of a thing.
_PART = ("face", "side", "rim", "edge", "handle", "body", "top", "bottom", "lid", "corner",
         "surface", "panel", "mouth", "neck", "base", "opening", "interior", "front", "back",
         "end", "wall", "spout", "brim", "lip", "half", "part", "underside")
#: What may stand between the article and the part word without changing that it is one.
_PART_MODIFIER = ("near", "far", "front", "back", "outer", "inner", "upper", "lower", "top",
                  "bottom", "near-side", "opposite", "closest", "nearest", "furthest")
#: A phrase with one of these in it is already saying where the thing is, not which part of
#: it is meant, so the trailing-word rule stays out of it.
_PREPOSITION = ("of", "with", "on", "in", "beside", "near", "under", "behind", "between",
                "above", "by", "against", "next", "atop", "inside")
_ARTICLE = ("the", "a", "an", "its", "this", "that")


def strip_part(target: str) -> str:
    """The whole-object phrase inside a phrase that named a part of it."""
    text = " ".join(str(target or "").split())
    for _ in range(4):                    # "the near side of the top of X" nests twice
        head, sep, rest = text.partition(" of ")
        if not sep or not _heads_with_part(head):
            break
        text = rest.strip() or text
    words = text.split()
    if len(words) >= 3 and words[-1].lower().strip(",.") in _PART             and not any(w.lower() in _PREPOSITION for w in words)             and words[-2].lower() not in _ARTICLE:
        text = " ".join(words[:-1])
        if text.lower().endswith("'s"):
            text = text[:-2].rstrip()
    return text


def _heads_with_part(head: str) -> bool:
    """Does this phrase START with a part word -- an article and one modifier allowed?"""
    words = [w.strip(",.").lower() for w in head.split() if w.strip(",.")]
    while words and words[0] in _ARTICLE:
        words.pop(0)
    if len(words) > 1 and words[0] in _PART_MODIFIER and words[1] in _PART:
        words.pop(0)
    return bool(words) and words[0].rstrip("s") in tuple(p.rstrip("s") for p in _PART)


#: Words that carry nothing about WHICH thing is meant.
_SMALL = ("the", "a", "an", "of", "with", "on", "in", "its", "it", "and", "to", "that", "this",
          "at", "by", "for", "is", "are", "from", "near", "beside", "next")


def object_words(target: str) -> set:
    """The content words of a phrase for an object, with the part-of wording taken off."""
    return content_words(strip_part(target))


def same_object(left: str, right: str) -> bool:
    """Do two phrases name the same thing, one of them in more words than the other?"""
    small, big = sorted((object_words(left), object_words(right)), key=len)
    if not small or not big or not small <= big:
        return False
    return len(small) * 2 >= len(big)


def head_words(target: str) -> set:
    """The content words for the thing ITSELF: what stands before the first preposition that
    says where it is, so "the black bowl next to the ramekin" gives {black, bowl} and not the
    ramekin. A phrase with no such preposition gives all of its words."""
    head = []
    for word in strip_part(target).split():
        if word.lower().strip(",.;:") in _PREPOSITION:
            break
        head.append(word)
    return object_words(" ".join(head)) or object_words(target)


# ------------------------------------------------------------- where the planner MARKED it
#
# The planner sees the pictures once, at plan time, with the whole task in front of it and no
# hurry. The executor sees them every cycle, under a subgoal sentence, and would otherwise
# re-choose the object out of the words each time. So the planner draws the box ONCE and hands
# it over, and the executor is given a place to look rather than a question to re-answer.
#
# The coordinates are the ones the model already answers grounding in: 0..1000 across and
# 0..1000 down, whatever the picture's pixel size (src/executor/geometry.NORMALISED_SPAN, and
# the executor's own locate prompt asks for them in exactly these words). Keeping the span
# normalised is what lets one box survive a camera that renders at 768 beside one at 1024.

#: The span of the normalised box coordinates, both axes.
BOX_SPAN = 1000
#: How much of a picture one object's box may cover before it is not a box round an object.
MAX_BOX_SHARE = 0.40


def _camera_key(name: Any) -> str:
    """``'scene (now)'`` -> ``'scene'``. The prompt names the views with a parenthesis."""
    text = " ".join(str(name or "").split()).lower()
    return text.split("(")[0].strip().strip(":,")


def read_box(value: Any) -> Optional[List[int]]:
    """Four whole numbers out of whatever shape the box was written in, or None."""
    if isinstance(value, dict):
        for key in ("bbox_2d", "bbox", "box_2d", "box", "rect"):
            if key in value:
                return read_box(value[key])
        if all(key in value for key in ("x1", "y1", "x2", "y2")):
            value = [value["x1"], value["y1"], value["x2"], value["y2"]]
        elif all(key in value for key in ("x0", "y0", "x1", "y1")):
            value = [value["x0"], value["y0"], value["x1"], value["y1"]]
        else:
            return None
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    out = []
    for item in list(value)[:4]:
        if isinstance(item, bool) or item is None:
            return None
        try:
            out.append(int(round(float(item))))
        except (TypeError, ValueError):
            return None
    return out


def read_boxes(value: Any) -> Dict[str, List[int]]:
    """``{camera: [x0, y0, x1, y1]}`` out of the several shapes a reply writes it in."""
    if isinstance(value, (list, tuple)):
        value = {entry.get("camera") or entry.get("name") or entry.get("view"): entry
                 for entry in value if isinstance(entry, dict)}
    if not isinstance(value, dict):
        return {}
    out: Dict[str, List[int]] = {}
    for name, raw in value.items():
        camera = _camera_key(name)
        box = read_box(raw)
        if camera and box is not None:
            out[camera] = box
    return out


def box_fault(box: Sequence[int]) -> str:
    """What is wrong with this box, or "" when it is usable."""
    numbers = read_box(box)
    if numbers is None:
        return "it is not four numbers"
    x0, y0, x1, y1 = numbers
    if min(numbers) < 0 or max(numbers) > BOX_SPAN:
        return ("its coordinates are outside 0-{}, so they are not the normalised span"
                .format(BOX_SPAN))
    if x1 <= x0 or y1 <= y0:
        return "its corners are inverted or it has no area"
    share = float(x1 - x0) * float(y1 - y0) / float(BOX_SPAN * BOX_SPAN)
    if share >= MAX_BOX_SHARE:
        return ("it covers {:.0%} of the picture, which is not a box round one object"
                .format(share))
    return ""


def box_centre(box: Sequence[int]) -> Optional[Tuple[int, int]]:
    """The middle of a box, in the same normalised span it was written in."""
    numbers = read_box(box)
    if numbers is None:
        return None
    return ((numbers[0] + numbers[2]) // 2, (numbers[1] + numbers[3]) // 2)


def clean_boxes(boxes: Any, cameras: Sequence[str] = ()) -> Tuple[Dict[str, List[int]],
                                                                  List[str]]:
    """The boxes that can be used, and one sentence for each that cannot."""
    known = {_camera_key(name) for name in cameras if _camera_key(name)}
    out, notes = {}, []
    for camera, box in sorted(read_boxes(boxes).items()):
        if known and camera not in known:
            notes.append("box for {!r} dropped: there is no such camera".format(camera))
            continue
        fault = box_fault(box)
        if fault:
            notes.append("box for {!r} dropped: {}".format(camera, fault))
            continue
        out[camera] = read_box(box)
    return out, notes


def marked_at(target_boxes: Any, target: str = "") -> str:
    """"the planner marked it at ..."""
    boxes, _ = clean_boxes(target_boxes)
    if not boxes:
        return ""
    said = []
    for camera, box in sorted(boxes.items()):
        centre = box_centre(box)
        if centre is None:
            continue
        # No pixels: the render's size is not the size the picture is sent at.
        said.append("{} at ({}, {}) of 1000".format(camera, *centre))
    if not said:
        return ""
    return ("THE PLANNER MARKED IT AT PLAN TIME\nLooking at the pictures when it wrote the "
            "plan, the planner put {} in the {} image(s), at: {}. That is WHERE IT LOOKED, "
            "not proof of what the thing is -- the scene may have moved since, and the plan's "
            "reading of a picture can be wrong. Use it to know where to look; judge what you "
            "see there."
            .format(repr(target.strip()) if target.strip() else "the step's object",
                    len(said), "; ".join(said)))


class Subgoal(BaseModel):
    """One step the executor is asked to finish on its own."""

    model_config = ConfigDict(extra="forbid")

    name: str
    subgoal: str
    criterion: str
    # How to tell the object of this subgoal from everything else on the table, in words the
    # planner can SEE at plan time -- its colour, what is written on it, what shape of thing it
    # is, where it sits among its neighbours. A step that acts on no object may send null.
    target: Annotated[str, BeforeValidator(lambda value: "" if value is None else value)] = ""
    # WHERE that object is in the pictures the plan was written from, one box per camera that
    # can see it: {"scene": [x0, y0, x1, y1], ...} in 0-1000 normalised coordinates.
    target_boxes: Dict[str, List[int]] = Field(default_factory=dict)
    # Which of the task's clauses this step serves -- the index into ``Plan.clauses``, or -1 for
    # a step that serves none of them (a retreat, a look from higher up).
    clause: int = -1
    max_cycles: int = 10
    # NOT the planner's to write: worked out from the wording by derive_gripper_only() and
    # overwritten on every plan and every replan.
    gripper_only: bool = False

    @field_validator("target_boxes", mode="before")
    @classmethod
    def _boxes(cls, value):
        """Read, never refused."""
        return read_boxes(value)

    @field_validator("clause", mode="before")
    @classmethod
    def _clause(cls, value):
        """An index written as anything but a number is no index: -1, not a refused plan."""
        if isinstance(value, bool) or value is None:
            return -1
        try:
            return int(value)
        except (TypeError, ValueError):
            return -1


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subgoals: List[Subgoal] = Field(default_factory=list)
    rationale: str = ""
    # What the task asks for, one entry per thing, in the planner's own words, worked out ONCE
    # when the first plan is written.
    clauses: List[str] = Field(default_factory=list)
    # WHICH PICTURES the boxes above were drawn on: the camera names, and the capture time of
    # the observation the plan was written from.
    boxed_on: List[str] = Field(default_factory=list)
    boxed_at: float = 0.0
    # NOT the model's to write: the steps whose box a replan KEPT from an earlier plan rather
    # than taking the one drawn now, each with the capture time of the picture that box was
    # drawn on -- so a sweep scoring a step's box against the truth tape reads the right frame.
    boxes_kept_from: Dict[str, float] = Field(default_factory=dict)
    # Whether there is anything left to try.
    recoverable: bool = True

    @field_validator("recoverable", mode="before")
    @classmethod
    def _recoverable(cls, value):
        """Only an unmistakable "no" gives up. Everything else carries on."""
        if isinstance(value, bool):
            return value
        if value is None:
            return True
        if isinstance(value, str):
            return _word(value) not in ("false", "no", "0", "none", "unrecoverable")
        try:
            return bool(value)
        except Exception:                 # pragma: no cover -- an exotic object
            return True


#: What makes giving up available on the FIRST replan, in the words the prompt asks for.
_HARD_TO_RECOVER = (
    "out of reach", "unreachable", "beyond the arm", "beyond its reach", "cannot reach",
    "off the table", "on the floor", "fallen off", "fell off", "fell to the floor",
    "outside the workspace", "out of the workspace", "broken", "shattered", "smashed",
    "spilled", "spilt", "cannot be regrasped", "cannot be re-grasped", "cannot be picked up",
    "permanently", "irreversible", "irreversibly", "cannot be undone", "no longer possible")


def hard_to_recover(reason: str) -> bool:
    """Does this reason name a fault the arm cannot undo, rather than an attempt that failed?"""
    return any(word in " ".join(str(reason or "").lower().split())
               for word in _HARD_TO_RECOVER)


def gave_up_reason(plan: "Plan", replans_so_far: int = 0) -> str:
    """Why the mission should stop now, or "" to carry on."""
    if plan is None or plan.recoverable:
        return ""
    reason = " ".join(str(plan.rationale or "").split()) or "the planner gave no reason"
    if int(replans_so_far) < 1 and not hard_to_recover(reason):
        return ""
    return reason


def outstanding_clauses(plan: "Plan", keep: int) -> List[int]:
    """The clauses the steps that have NOT been done are still serving, in order."""
    seen = []
    for step in list(plan.subgoals)[max(0, int(keep)):]:
        if 0 <= step.clause < len(plan.clauses) and step.clause not in seen:
            seen.append(step.clause)
    return seen


class Clause(BaseModel):
    """One thing the task asks for, and whether the scene shows it."""

    model_config = ConfigDict(extra="forbid")

    what: str = ""
    met: bool = False
    evidence: str = ""


class Verdict(BaseModel):
    """Judged once when a subgoal ends, whatever ended it."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["done", "not_done", "task_done", "abort"]
    because: str = ""
    evidence: str = ""
    # "That jug is white with a blue pattern, not black." Kept apart from the verdict on
    # purpose: the plan's name for a thing is the planner's reading of a picture, and a reader
    # asked to name what is wrong will name a mismatched name, because it is the cheapest thing
    # in front of it.
    identity: str = ""
    replan: bool = False
    # One row per thing the TASK asks for, filled whenever the task's own question is answered.
    clauses: List["Clause"] = Field(default_factory=list)

    @field_validator("status", mode="before")
    @classmethod
    def _status(cls, value):
        return _word(value)

    def printed_elsewhere(self, task: str) -> List[str]:
        """Printing the verdict says was READ off the thing ("labeled 'X'"), not the task's word."""
        said = " ".join((self.identity, self.because, self.evidence))
        return foreign_labels(" ".join(m.group(0) for m in _PRINTED.finditer(said)), task)


_FINDINGS = ("fine", "wrong_object", "dropped", "collision", "subgoal_done_early",
             "scene_changed", "unsure")


class Alert(BaseModel):
    """What the scene monitor may say while a subgoal is running. There is no action field."""

    model_config = ConfigDict(extra="forbid")

    level: Literal["fine", "attention", "stop"]
    finding: Literal[_FINDINGS] = "fine"
    because: str = ""
    # Somewhere to put "that does not look like the plan's description" that is not a stop.
    identity: str = ""

    @field_validator("level", "finding", mode="before")
    @classmethod
    def _enum(cls, value):
        return _word(value)

    @model_validator(mode="before")
    @classmethod
    def _unnamed(cls, data):
        """A finding outside the list (level copied in, r4full: 739 of 2591) is "unsure"."""
        if isinstance(data, dict) and _word(data.get("finding", "fine")) not in _FINDINGS:
            data = {**data, "finding": "unsure", "level": "attention"
                    if _word(data.get("level")) == "stop" else data.get("level")}
        return data


# ------------------------------------------------------------- what is WRITTEN on a thing
#
# The planner describes an object by the words printed on it, and a quoted label read off the
# wrong tin (a distractor's) sends the locator to that distractor. A label is only a
# description of the thing the TASK means when the task's own noun is in it.

#: A quoted run of text.
_QUOTED = re.compile(r"['\"‘“]([^'\"‘’“”]{2,}?)"
                     r"['\"’”](?![A-Za-z])")
_PRINTED = re.compile(r"(?:printed|label\w*|reads?|reading|says)\W{{1,3}}{0}|{0}\W{{1,3}}(?:printed|"
                      r"written|printing)".format(_QUOTED.pattern))

#: The words that introduce a quotation and mean nothing once it is gone.
_LEAD = ("labeled", "labelled", "label", "labels", "marked", "reading", "reads", "read",
         "says", "saying", "printed", "titled", "named", "written", "word", "words",
         "text", "lettering", "legible", "that", "which", "with", "and", "a", "an", "the",
         "in", "of", "on", "is", "are", "it")


def content_words(text: str) -> set:
    """The words of a sentence that carry which thing is meant."""
    return {word for word in _tokens(text) if word not in _SMALL}


def quoted_labels(target: str) -> List[str]:
    """Every run of quoted text in a phrase, in the order it was written."""
    return [m.group(1).strip() for m in _QUOTED.finditer(str(target or ""))
            if m.group(1).strip()]


def foreign_labels(target: str, task: str) -> List[str]:
    """The quoted labels in a target that are NOT the task's own word for the thing."""
    said = content_words(task)
    if not said:
        return []                          # no task to check against: the rule is off
    foreign = []
    for label in quoted_labels(target):
        words = [word for word in _tokens(label) if word not in _SMALL]
        if words and words[-1] not in said:
            foreign.append(label)
    return foreign


def without_labels(target: str, labels: Sequence[str]) -> str:
    """The phrase with those quotations taken out, and the words that introduced them."""
    wanted = {str(label).strip() for label in labels}
    text = str(target or "")
    out, cut = [], 0
    for match in _QUOTED.finditer(text):
        if match.group(1).strip() not in wanted:
            continue
        head = text[cut:match.start()]
        words = head.split()
        while words and words[-1].strip(",.:;").lower() in _LEAD:
            words.pop()
        out.append(" ".join(words))
        cut = match.end()
    if not out:
        return " ".join(text.split())
    out.append(text[cut:])
    words = " ".join(" ".join(out).split()).strip(" ,.;:").split()
    while words and words[-1].strip(",.:;").lower() in _LEAD:
        words.pop()
    return " ".join(words).strip(" ,.;:")


# ------------------------------------------------------------- what a step DOES to a thing

#: The beginnings of the words for a step that ACTS on something -- the jaws close on it, it is
#: carried, put down, let go, or a control is worked.
_ACT = ("clos", "grasp", "take", "took", "pick", "lift", "carr", "plac", "put", "releas",
        "drop", "push", "shov", "pull", "turn", "rotat", "slid", "press", "squeez",
        "insert", "deposit", "lower", "shut")
#: ...and the ones that have to match WHOLE, because "gripper" is the name of the tool and
#: every sentence in the plan has it: a stem test on it made a look from higher up an action.
_ACT_WORD = ("grip", "grips", "gripped", "gripping")


def _acting(said) -> set:
    return ({stem for stem in _ACT if any(word.startswith(stem) for word in said)}
            | {word for word in _ACT_WORD if word in said})


def acts(*text: str) -> bool:
    """Does this step DO something to the object, rather than look at it or go near it?"""
    return bool(_acting(_tokens(" ".join(str(part or "") for part in text))))


#: The directions a step can come at a thing from, as words a sentence uses for them.
_SIDE = ("left", "right", "front", "back", "behind", "above", "below", "underneath", "side",
         "horizontally", "vertically", "sideways", "diagonally")


def ways(*text: str) -> set:
    """The side and the mode a step names -- how it went about the thing."""
    said = _tokens(" ".join(str(part or "") for part in text))
    return {"side: " + w for w in _SIDE if w in said} | {"mode: " + s for s in _acting(said)}


# ------------------------------------------------------------- where a thing is being PUT

#: How a criterion says where something ends up.
_RECEIVES = ("resting on the", "resting in the", "sitting on the", "sitting in the",
             "placed on the", "placed in the", "on top of the", "inside the", "into the",
             "onto the", "in the", "on the")


def place_words(criterion: str) -> set:
    """The words naming where a criterion says the thing is put down."""
    text = " ".join(str(criterion or "").lower().split())
    found = set()
    for phrase in _RECEIVES:
        start = 0
        while True:
            at = text.find(phrase, start)
            if at < 0:
                break
            start = at + len(phrase)
            tail = text[start:start + 60].split()[:4]
            found |= {word.strip(",.;:") for word in tail
                      if word.strip(",.;:") and word.strip(",.;:") not in _SMALL}
    return found


def names_a_place(criterion: str, task: str) -> bool:
    """Does this criterion say where the thing goes, in the task's own words for a place?"""
    said = content_words(task)
    return bool(said and (place_words(criterion) & said))


#: The words for a step whose whole point is to PUT the thing somewhere, as opposed to one
#: that only opens the jaws.
_PLACES = ("put", "place", "lower", "deposit", "set", "drop", "insert", "into")


def places(*text: str) -> bool:
    """Is this step a placement -- the thing is being set down somewhere on purpose?"""
    said = _tokens(" ".join(str(part or "") for part in text))
    return any(word.startswith(stem) for word in said for stem in _PLACES)
