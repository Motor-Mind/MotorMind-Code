"""What a reply from the model is allowed to contain."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, field_validator


class Proposal(BaseModel):
    """One short motion towards the subgoal, or a claim that it is already met."""

    model_config = ConfigDict(extra="forbid")

    assessment: str = ""
    done: bool = False
    command: Optional[Dict[str, Any]] = None
    expect: str = ""
    confidence: float = 0.0

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))


FINDINGS = ("on_course", "wrong_direction", "overshoot", "obstacle", "scene_changed", "unsure")


class Supervision(BaseModel):
    """Continue or stop."""

    model_config = ConfigDict(extra="forbid")

    decision: str
    because: str = ""
    evidence: str = ""
    finding: str = "on_course"

    @field_validator("finding")
    @classmethod
    def _finding(cls, value: str) -> str:
        cleaned = str(value).strip().lower()
        if cleaned not in FINDINGS:
            raise ValueError("finding must be one of {}, got {!r}".format(", ".join(FINDINGS), value))
        return cleaned

    @field_validator("decision")
    @classmethod
    def _known(cls, value: str) -> str:
        cleaned = str(value).strip().lower()
        if cleaned not in ("continue", "stop"):
            raise ValueError('decision must be "continue" or "stop", got {!r}'.format(value))
        return cleaned


# ------------------------------------------------------- what a criterion is ABOUT

#: Words that make a criterion one about the PAYLOAD: it cannot be true with empty jaws.
_HOLDING = ("holding", "hold", "holds", "held", "grasp", "grasped", "grasping",
            "carry", "carrying", "carried", "gripping")
#: ...and the words that make it a RELEASE, which cannot be true while something is held.
_RELEASE = ("release", "released", "releasing", "let go", "lets go", "jaws open",
            "jaws as open", "nothing held", "nothing is held", "no longer holding",
            "not holding", "resting on", "resting in", "sitting in", "sitting on")


def wants_holding(criterion: str) -> bool:
    """True when the criterion cannot be true with empty jaws -- and is not a release."""
    text = " ".join(str(criterion or "").lower().split())
    if not text or any(word in text for word in _RELEASE):
        return False
    return any(word in text for word in _HOLDING)


def wants_release(criterion: str) -> bool:
    """True when the step is about letting the thing GO -- a release, a put-down, a place."""
    text = " ".join(str(criterion or "").lower().split())
    return bool(text) and any(word in text for word in _RELEASE)


#: How a step says where the thing it is carrying ends up, longest phrase first so the one that
#: matches is the one that was written.
_PUT_IN = ("resting in the", "sitting in the", "placed in the", "dropped in the", "inside the",
           "into the", "in the")
_PUT_ON = ("resting on the", "sitting on the", "placed on the", "set down on the",
           "on top of the", "onto the", "touches the", "touching the", "on the")


def placed_in(*text: str) -> Optional[bool]:
    """True for IN something, False for ON something, None when nothing says."""
    said = " ".join(" ".join(str(part or "").lower().split()) for part in text)
    best: Optional[bool] = None
    at = len(said) + 1
    for phrases, inside in ((_PUT_IN, True), (_PUT_ON, False)):
        for phrase in phrases:
            start = 0
            while True:
                found = said.find(phrase, start)
                if found < 0:
                    break
                start = found + len(phrase)
                tail = said[start:start + 20].split()
                if tail and tail[0].strip(",.;:") in _NOT_A_PLACE:
                    continue          # "in the wrist view" is a camera, not a placement
                if found < at:
                    at, best = found, inside
                break
    return best


#: What can follow "in the" or "on the" and not be somewhere a thing is put down.
_NOT_A_PLACE = ("view", "picture", "frame", "image", "air", "way", "same", "wrist", "scene",
                "camera", "middle", "meantime", "process")


#: ...and how a step says where the thing is going while it is still being CARRIED there, which
#: is not a placement yet: "until it is over the basket".
_CARRY_TO = ("directly over the", "directly above the", "over the", "above the", "toward the",
             "towards the")
#: Where a receiving phrase stops. Everything after one of these belongs to another clause.
_ANOTHER_CLAUSE = ("and", "then", "until", "while", "with", "so", "seen", "which", "that",
                   "but", "before", "after", "still", "without")
#: Words that describe a thing and never name one: a comma after one sits between two
#: describers, "the open, dark patterned bowl" (gpt6sol-full-fix6b 106: the receiver was "open").
_DESCRIBES = ("open", "empty", "closed", "dark", "light", "pale", "bright", "small", "large",
              "big", "little", "flat", "round", "shallow", "deep", "tall", "short", "wide",
              "narrow", "black", "white", "gray", "grey", "red", "green", "blue", "yellow",
              "brown", "orange", "pink", "purple", "silver")


_OWN = r"(?:(?!\b(?:is|are|and)\b)[^,.;:])"            # a clause's own words
#: "the plate is centred under the gripper" is "over the plate" said from underneath...
_UNDER = re.compile(r"(?:^|(?<=[,.;:])|(?<= and )) *(?:(?:the|a|an) )?({0}+?) (?:is|are) "
                    r"(?:{0}*? )?(?:under|beneath|below) the ".format(_OWN))
#: ...and what comes after the word for holding a thing describes IT, not where it goes -- up to
#: the next clause or a carry's own words ("the held bowl until its bottom touches the plate",
#: "the held bowl a short way toward the plate" both name the plate).
_HELD = re.compile(r"\b(?:holding|holds|held|gripping)\b.*?(?= (?:and|then|until|while|so|"
                   r"before|after)\b| (?:{}) |[,.;:]|$)".format("|".join(_CARRY_TO)))
#: "over the middle of the plate" is over the plate (gpt6sol-debug-1 task 0: no receiver).
_MIDDLE = re.compile(r"\bthe (?:middle|centre|center) of (?=the )")


def receiving_words(*text: str, held: str = "") -> str:
    """What a step says the thing in the jaws is going TO, in the step's own words. Where
    ``held`` -- the thing's own name -- says where it stood, that is its name, not where it
    goes (gpt6sol-full-fix5 task 3: "the black bowl on the cookie box touches the plate")."""
    said = " ".join(" ".join(str(part or "").lower().split()) for part in text)
    for name in str(held or "").lower().split(","):
        name = re.sub(r"^(?:the|a|an) ", "", " ".join(name.split()))
        where = [name.find(" " + phrase) for phrase in _PUT_IN + _PUT_ON if " " + phrase in name]
        if where:
            said = said.replace(name, name[:min(where)])
    said = _UNDER.sub(r"over the \1, ", _HELD.sub(",", _MIDDLE.sub("", said)))
    best, at = "", len(said) + 1
    for phrase in _CARRY_TO + _PUT_IN + _PUT_ON:
        start = 0
        while True:
            found = said.find(phrase, start)
            if found < 0:
                break
            start = found + len(phrase)
            words = []
            for word in said[start:].split():
                bare = word.strip(",.;:")
                if not words and bare in _NOT_A_PLACE:
                    break             # "in the wrist view" is a camera, not a destination
                if bare in _ANOTHER_CLAUSE or not bare:
                    break
                words.append(bare)
                if word != bare and not (word.endswith(",") and bare in _DESCRIBES):
                    break             # a comma ends it, unless it follows a describer
            if {"is", "are"} & set(said[start:].split()[:len(words) + 1]):
                continue              # "the ramekin it is resting on" says where it IS
            if words and found < at:
                at, best = found, " ".join(words)
            break
    return best
