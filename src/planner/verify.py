"""Did that subgoal actually happen?"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.executor.context import describe_cameras
from src.executor.loop import HELD_MARKER
from src.executor.models import wants_holding, wants_release
from src.executor.proposal import Attempt

from .models import (Alert, Subgoal, Verdict, head_words, marked_at, object_words, strip_part,
                     wants_turn)
from .planner import NOTHING_SUMMARISED, ask_model, attempts_as_dicts


@dataclass
class VerdictResult:
    verdict: Optional[Verdict] = None
    error: str = ""
    attempts: List[Attempt] = field(default_factory=list)
    elapsed_s: float = 0.0
    # Set when the answer came from the gate below rather than from the model, so a sweep can
    # count how often the reading settled it. An empty string means the model was asked.
    gate: str = ""
    # The whole-task answer a step's verdict bought (``also_ask_the_task``), when it did not
    # override that verdict.
    asked: Optional[Verdict] = None

    @property
    def ok(self) -> bool:
        return self.verdict is not None and not self.error

    def as_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "error": self.error, "elapsed_s": round(self.elapsed_s, 2),
                "gate": self.gate,
                "verdict": None if self.verdict is None else self.verdict.model_dump(),
                "asked": None if self.asked is None else self.asked.model_dump(),
                "attempts": attempts_as_dicts(self.attempts)}


#: The name the mission files the whole-task question under, and the one subgoal name whose
#: verdict must never trigger another one.
TASK = "task"


def task_question(task: str) -> Subgoal:
    """The whole task, written as the one subgoal a verifier can be asked about."""
    task = task.strip()
    return Subgoal(name=TASK, subgoal=task,
                   criterion="The WHOLE task is complete: {}. Judge the scene as it "
                             "is now against the task, not against any one step: "
                             "answer task_done if everything the task asked for is "
                             "where it asked for it, and not_done if anything is "
                             "still missing -- say what. \"done\" is not an answer to "
                             "this question. Break the task into its clauses -- one "
                             "per thing it asks for -- and fill `clauses` with all of "
                             "them, each with what it asks and whether the scene shows "
                             "it: a task_done with any clause unmet, or with none "
                             "listed, is not accepted.".format(task),
                   max_cycles=1)


class Verifier:
    def __init__(self, client, prompts: Dict[str, Any], max_attempts: int = 3):
        self.client = client
        self.prompts = prompts
        self.max_attempts = max_attempts
        # The latest FINISHED memory note, read when a prompt is built and never waited for.
        self.note: Callable[[], str] = lambda: ""

    def verify(self, task: str, subgoal: Subgoal, before, now,
               executor_claim: Dict[str, Any], alert: Optional[Alert]) -> VerdictResult:
        started = time.monotonic()
        # Never for the whole-task question: its criterion is the TASK's own sentence, so a
        # task that says "carry", "holding" or "resting on" -- most of them -- would be
        # answered by the gripper's reading about one step and never put to anybody.
        gated = (None if subgoal.name == TASK
                 else grasp_gate(subgoal.criterion, getattr(now, "grasp", None)))
        if gated is not None:
            # Nothing to ask: the criterion is about the payload and the robot has just said
            # whether it has one.
            verdict, why = gated
            return self.also_ask_the_task(
                VerdictResult(verdict=verdict, gate=why,
                              elapsed_s=time.monotonic() - started),
                task, subgoal, before, now, executor_claim)
        images, names = _now_and_before(now, before)
        base = self._base(
            task, subgoal.subgoal.strip(),
            subgoal.criterion.strip() or "No criterion was written for this subgoal: "
                                         "judge it from the subgoal itself.",
            _claim_text(executor_claim), _alert_text(alert), now, before, names)
        # Where the planner drew this object at plan time.
        hint = marked_at(subgoal.target_boxes, subgoal.target)
        if hint:
            base = base + "\n\n" + hint
        system = self.prompts["verify"]["system"]
        grasp = getattr(now, "grasp", None)
        held = held_numbers(executor_claim)
        if held is not None and subgoal.name != TASK:
            # The step ended because the robot said it was holding something.
            gap, width = held
            verdict, attempts, error = ask_model(
                self.client, base + _IDENTITY.format(
                    target=repr(subgoal.target.strip() or subgoal.subgoal.strip()),
                    number="The pads settled {:.0f} mm apart, and the cameras measure the thing "
                           "this step names at about {:.0f} mm across the line they close "
                           "along. A close on a body settles at about that body's width; a "
                           "close on a rim or a handle settles far narrower."
                           .format(gap, width)),
                system, images, names, Verdict, None, self.max_attempts)
            if verdict is None:
                verdict = Verdict(status="done", because="the robot reports it is holding "
                                                         "something and nothing could be read "
                                                         "against that")
            off = ((executor_claim or {}).get("reached") or {}).get("from_mark_mm")
            if verdict.status == "not_done" and (grasp or {}).get("holding") is True \
                    and off is not None and off <= _REACHED_MM \
                    and not verdict.printed_elsewhere(task):
                # Closed on what the planner marked, and holding it: a look-alike named in the
                # picture is a doubt, not a verdict. Printing read as another product's is one.
                verdict = verdict.model_copy(update={
                    "status": "done", "replan": False, "identity": "doubted: " + verdict.because,
                    "because": "closed {:.0f} mm from the planner's mark, and held".format(off)})
            verdict, mismatch = located_identity_gate(subgoal, executor_claim, verdict)
            return VerdictResult(verdict=verdict, error=error, attempts=attempts,
                                 gate=mismatch or ("identity" if verdict.status != "done"
                                                   else ""),
                                 elapsed_s=time.monotonic() - started)
        if (grasp or {}).get("holding") is True and wants_holding(subgoal.criterion):
            base = base + _HOLDING_IS_SETTLED.format(
                reason=(grasp or {}).get("reason") or "the gripper's own reading")
        verdict, attempts, error = ask_model(self.client, base, system, images, names,
                                             Verdict, None, self.max_attempts)
        gate = ""
        if holding_contradiction(subgoal.criterion, grasp, verdict):
            # The reading and the picture disagree about whether anything is in the jaws, and
            # the reading is the one that cannot be fooled by a finger in front of a handle.
            again, more, _ = ask_model(self.client, base + _CONTRADICTION.format(
                reason=(grasp or {}).get("reason") or "the gripper's own reading"),
                system, images, names, Verdict, None, self.max_attempts)
            attempts = list(attempts) + list(more)
            if again is not None and not holding_contradiction(subgoal.criterion, grasp, again):
                verdict, error = again, ""       # it answered on other grounds: that stands
            else:
                # Still "empty", or no valid answer at all.
                verdict, error, gate = _held_anyway(again or verdict, grasp), "", "holding_true"
        else:
            # The other disagreement worth one more question, and the same shape: the reader has
            # named an object under the tool that is not the one this step is about, and the
            # cameras measured the tool onto the one it IS about.
            named = reach_contradiction(subgoal, executor_claim, verdict)
            if named:
                reached = (executor_claim or {}).get("reached") or {}
                again, more, _ = ask_model(self.client, base + _GROUNDED.format(
                    named=named, target=repr(subgoal.target.strip() or "the step's object"),
                    label=repr(str(reached.get("label") or "")),
                    residual=float(reached.get("residual_mm") or 0.0)),
                    system, images, names, Verdict, None, 1)
                attempts = list(attempts) + list(more)
                if again is not None:
                    # Whatever it answers the second time stands: it was shown the number and
                    # asked to argue with it, which is the whole of what the re-ask buys.
                    verdict, error, gate = again, "", "regrounded"
        # Settled in code: a done for a take or a lift
        # whose own reach picked out something else by name is not one.
        verdict, mismatch = located_identity_gate(subgoal, executor_claim, verdict)
        gate = mismatch or gate
        verdict, undone = control_gate(subgoal, executor_claim, verdict)
        gate = undone or gate
        verdict, downgraded = check_clauses(verdict, subgoal.name == TASK)
        gate = downgraded or gate
        return VerdictResult(verdict=verdict, error=error, attempts=attempts, gate=gate,
                             elapsed_s=time.monotonic() - started)

    def _base(self, task: str, subgoal: str, criterion: str, claim: str, alert: str,
              now, before, names) -> str:
        """The verify prompt, filled: every question this verifier asks starts from it."""
        return self.prompts["verify"]["text"].format(
            task=task.strip(), subgoal=subgoal, criterion=criterion, claim=claim,
            alert=alert,
            note=self.note().strip() or NOTHING_SUMMARISED,
            robot=(getattr(now, "robot_text", "") or "").strip()
            or "the robot reported nothing about itself",
            cameras=_camera_text(names, now, before))

    # ------------------------------------------------------------- the question a gate skips

    def also_ask_the_task(self, gated: VerdictResult, task: str, subgoal: Subgoal, before, now,
                          executor_claim) -> VerdictResult:
        """A task question after a settled verdict: it can finish a done, never undo a not_done."""
        if subgoal.name == TASK or gated.verdict is None:
            return gated                  # never from the whole-task check itself
        # The code's own reading, not the executor's proposal from before the step's last moves.
        claim = dict(executor_claim or {}, done=gated.verdict.status == "done",
                     assessment="{} ({})".format(gated.verdict.because, gated.verdict.evidence))
        asked = self.ask_task(task, before, now, claim)
        gated.attempts = list(gated.attempts) + list(asked.attempts)
        if asked.verdict is not None and asked.verdict.status == "task_done" \
                and gated.verdict.status == "done":
            return VerdictResult(verdict=asked.verdict, error="", attempts=gated.attempts,
                                 gate=gated.gate, elapsed_s=gated.elapsed_s)
        gated.asked = asked.verdict          # not the answer, but its working is evidence
        return gated

    def ask_task(self, task: str, before, now, executor_claim) -> VerdictResult:
        """Ask only "is the whole task finished now?", from the pictures in hand."""
        started = time.monotonic()
        question = task_question(task)
        images, names = _now_and_before(now, before)
        base = self._base(task, question.subgoal, question.criterion,
                          _claim_text(executor_claim), _alert_text(None), now, before, names)
        # ONE attempt, not the usual three: this is the extra question a gate bought, and its
        # bound is what makes it affordable.
        verdict, attempts, error = ask_model(self.client, base, self.prompts["verify"]["system"],
                                             images, names, Verdict, None, 1)
        verdict, gate = check_clauses(verdict, True)
        return VerdictResult(verdict=verdict, error=error, attempts=attempts, gate=gate,
                             elapsed_s=time.monotonic() - started)


def _now_and_before(now, before):
    """Now first, the pre-subgoal frame last -- the ordering ``proposal._pair`` uses."""
    images, names = [], []
    if now is not None and getattr(now, "images", None):
        images += list(now.images)
        names += ["{} (now)".format(n) for n in now.names]
    if before is not None and getattr(before, "images", None):
        images += list(before.images)
        names += ["{} (before this subgoal started)".format(n) for n in before.names]
    if not images:
        return [], []
    return images, names


#: Words that say a step is about the thing coming UP -- off the table, clear of what it was
#: standing on.
_A_LIFT = ("lift", "lifted", "lifting", "risen", "rise", "rises",
           "off the table", "clear of the table", "into the air")


def about_a_lift(text: str) -> bool:
    said = " ".join(str(text or "").lower().split())
    return any(word in said for word in _A_LIFT)


def _camera_text(names, now, before) -> str:
    """Say what is missing as well as what is there: an absent picture is a fact, not a gap."""
    missing = []
    if not (now is not None and getattr(now, "images", None)):
        missing.append("There is no picture from now, so judge from the robot's numbers and "
                       "the executor's claim alone, and prefer not_done when you cannot tell.")
    if not (before is not None and getattr(before, "images", None)):
        missing.append("There is no earlier picture -- none was captured before this subgoal "
                       "started -- so a criterion about a CHANGE cannot be read off the images; "
                       "use the robot's numbers for it.")
    return " ".join([describe_cameras(names)] + missing)


def _claim_text(claim: Optional[Dict[str, Any]]) -> str:
    claim = claim or {}
    lines = ["it says the subgoal IS met" if claim.get("done")
             else "it did NOT say the subgoal was met"]
    if claim.get("assessment"):
        lines.append("its last assessment: {}".format(str(claim["assessment"]).strip()))
    lines.append("it used {} motion(s)".format(claim.get("cycles", "an unrecorded number of")))
    if claim.get("ending"):
        # An attempt that answered "done" on its first cycle and never moved the robot reads,
        # from the count alone, exactly like one that worked.
        lines.append("how it ended: {}".format(claim["ending"]))
    if claim.get("reached"):
        lines.append("measured by the cameras, not claimed: asked for {target!r}, they picked out "
                     "{label!r}, and at the last look the tool -- or what it carries -- was "
                     "{residual_mm:.0f} mm from it".format(**claim["reached"]))
    net = _net_motion_text(claim)
    if net:
        # What a control is judged by.
        lines.append(net)
    if claim.get("stopped_by"):
        lines.append("what ended it: {}".format(claim["stopped_by"]))
    if claim.get("error"):
        lines.append("it reported an error: {}".format(claim["error"]))
    return "\n".join("- " + line for line in lines)


def _net_motion_text(claim: Dict[str, Any]) -> str:
    """The net turn and the net travel since the last grasp, where there is any to report."""
    parts = []
    turn_total = abs(float(claim.get("turn_total_deg") or 0.0))
    move_total = abs(float(claim.get("move_total_mm") or 0.0))
    if turn_total >= 1.0:
        parts.append("turned a NET {:.0f} deg about the vertical ({:.0f} deg of turning in "
                     "all)".format(float(claim.get("turn_net_deg") or 0.0), turn_total))
    if move_total >= 1.0:
        parts.append("moved a NET {:.0f} mm from where it started ({:.0f} mm travelled in "
                     "all)".format(float(claim.get("move_net_mm") or 0.0), move_total))
    if not parts:
        return ""
    return ("measured by the robot, not claimed -- since it last took hold of something the "
            "tool has " + " and ".join(parts) +
            ". A net far smaller than the travel means it went and came back.")


def _alert_text(alert: Optional[Alert]) -> str:
    if alert is None:
        return "The scene monitor raised nothing while this subgoal ran."
    return ("The scene monitor said {} ({}): {}".format(
        alert.level, alert.finding, alert.because.strip() or "no reason given"))


# --------------------------------------------------------------------------- the gate

#: The words a reader uses when it thinks the jaws are shut on air.
_EMPTY_JAWS = ("empty", "nothing held", "nothing is held", "nothing in the jaws",
               "holding nothing", "nothing between the fingers", "not holding anything",
               "no object is held", "nothing in its jaws", "did not grasp", "failed to grasp",
               "not grasped", "no object in the", "grasped nothing", "closed on nothing",
               "closed on air")
#: ...and the same claim made about the OBJECT instead of about the jaws.
_EMPTY_OBJECT = ("still on the table", "still sitting on the table", "remains on the table",
                 "not picked up", "never picked up", "was not lifted")
_EMPTY = _EMPTY_JAWS + _EMPTY_OBJECT

#: How a reader says the jaws are in the wrong PLACE on a thing they are already holding -- not
#: that they are empty, which is what ``_EMPTY`` is for.
_LEVEL = ("level with", "not level", "must be level", "above the top", "too high",
          "above the body", "higher than the", "not low enough")


#: The other clauses a criterion can carry.
_OTHER_CLAUSE = ("risen", "rise", "rises", "height", "above", "below", "under", "over",
                 "inside", "resting", "wrist view", " mm", "centred", "centered", "aligned",
                 "directly", "level")


def only_about_holding(criterion: str) -> bool:
    """Is the WHOLE of this criterion "the robot has the thing in its jaws"?"""
    text = " ".join(str(criterion or "").lower().split())
    if not wants_holding(text) or any(ch.isdigit() for ch in text):
        return False
    return not any(word in text for word in _OTHER_CLAUSE)


#: Appended to the verify prompt BEFORE the first question, whenever the robot says it is
#: holding something and the step is about the payload.
_HOLDING_IS_SETTLED = """

WHAT THE GRIPPER HAS ALREADY SETTLED
The robot's own gripper reports that it IS holding something ({reason}). For a criterion
about holding, that reading is the answer, not the picture: the fingers hide what they hold,
and a thing taken by a handle, a rim or a stem shows almost nothing from above. Do not
withhold `done` because the jaws look high on the object, because they are not level with its
body, or because the tool sits some millimetres above its top -- none of those is what the
criterion asks, and the step's own claim that they should be lower is not evidence about what
is held. Say not_done only if printing read on the held thing names another product than the
task's, or another clause of the criterion -- a height, a place, a state -- is unmet; say which.
"""


#: Appended to the verify prompt for the ONE re-ask a contradiction earns.
_IDENTITY = """

ONE QUESTION, AND IT IS NOT WHETHER THE GRASP HAPPENED
The robot reports the jaws are holding something and that settles the grasp: do not answer on
whether it is held, how level the jaws are, or how far down they went. {number}

Answer ONLY this: is the thing in the jaws the thing this step names, {target}? Look at the
two pictures, before and now, and say `done` if it is, or if you cannot tell. Say `not_done`,
with "it is holding a different object" or "it is holding nothing" in your reason, ONLY if they
show the named thing still standing where it was, or something else in the jaws -- printing
you can read on it naming another product than the task's does (quote it: printed "...").
"""

def held_numbers(claim: Optional[Dict[str, Any]]) -> Optional[Tuple[float, float]]:
    """(the gap the pads settled at, the width measured for the named thing), or None."""
    text = " ".join(str((claim or {}).get(key) or "")
                    for key in ("stopped_by", "ending", "assessment"))
    if HELD_MARKER not in text:
        return None
    gap = re.search(r"settled ([0-9.]+) mm apart", text)
    width = re.search(r"about ([0-9.]+) mm across", text)
    if not gap or not width:
        return None
    return float(gap.group(1)), float(width.group(1))


_CONTRADICTION = """

ONE THING TO SETTLE BEFORE YOU ANSWER AGAIN
The robot's own gripper says it IS holding something ({reason}). That reading, and not the
picture, settles whether anything is in the jaws: the fingers hide what they hold, and a thing
taken by a handle, a rim or a stem shows almost nothing from above. Your answer said the jaws
were empty, and the reading contradicts it. Answer once more. You may still say not_done -- but
only because some OTHER clause of the criterion (a height, a place, a state) is not met, and
then say which clause. Not because nothing is held.
"""


# ------------------------------------------------- a name against a measurement, asked once

#: How close the tool has to have ended to the thing the cameras picked out before its label
#: is worth putting against a reader's naming of what is under the tool, or the planner's mark.
_REACHED_MM = 30.0

#: How a verdict says what the tool is over.
_OVER = ("is positioned directly over the ", "is positioned over the ", "positioned over the ",
         "is directly over the ", "is directly above the ", "is hovering over the ",
         "is centred over the ", "is centered over the ", "is sitting over the ",
         "is over the ", "is above the ", "over the ", "above the ", "on top of the ",
         "over a ", "above a ")                   # "above a different container" (debug-4 t50)
#: ...and where that phrase stops.
_ENDS = (",", ".", ";", ":", " not ", " rather ", " instead ", " and the ", " while ",
         " but ", " which ", " whereas ", " at a ", " with the ")


def names_another_object(text: str, target: str) -> str:
    """The thing a verdict says is under the tool, when it is not the step's own target."""
    said = " ".join(str(text or "").lower().split())
    hit = next(((said.index(word) + len(word)) for word in _OVER if word in said), None)
    if hit is None:
        return ""
    rest = said[hit:]
    for end in _ENDS:
        index = rest.find(end)
        if index > 0:
            rest = rest[:index]
    rest = " ".join(rest.split()[:6]).strip()
    if not object_words(rest) or agrees(rest, target):
        return ""
    return rest


def agrees(left: str, right: str) -> bool:
    """Are these two phrases about the same thing, one of them in fewer words than the other?"""
    small, big = sorted((object_words(left), object_words(right)), key=len)
    return bool(small) and bool(big) and small <= big


def reach_contradiction(subgoal: Subgoal, claim: Optional[Dict[str, Any]],
                        verdict: Optional[Verdict]) -> str:
    """A not_done that names another object under the tool, against a reach that measured
    one."""
    if verdict is None or verdict.status != "not_done" or subgoal.name == TASK:
        return ""
    reached = (claim or {}).get("reached") or {}
    residual = reached.get("residual_mm")
    if not isinstance(residual, (int, float)) or residual > _REACHED_MM:
        return ""
    if not agrees(str(reached.get("label") or ""), subgoal.target):
        return ""
    return names_another_object("{} {}".format(verdict.because, verdict.evidence),
                                subgoal.target)


#: Appended to the verify prompt for the ONE re-ask a naming against a measurement earns.
_GROUNDED = """

ONE THING TO SETTLE BEFORE YOU ANSWER AGAIN
You have said the thing under the tool is {named}. During this same step the cameras were
asked for {target}, picked out {label} and measured the tool {residual:.0f} mm from it. That
is a measurement of where the tool ended, not a description of it, and it says the tool is
over the object this step names. Answer once more, with that in front of you. You may still
say not_done -- but then say what IN THE PICTURE shows a different object there: what is
written on it, its colour, what kind of thing it is, or where it stands. Or name the clause of
the criterion that is not met. A different name on its own is not enough to throw away a step
that was measured onto its target, and the plan that follows a not_done here drops every
remaining step for this object.
"""


# ------------------------------------------- the cameras' own name for what was reached


def names_something_else(label: str, asked: str, target: str) -> bool:
    """Did a look ASKED FOR the step's own object come back with a name that has nothing of
    that object's own head phrase in it? A look whose words miss that head phrase -- for the
    receiver, the landmark named after "on", a PART of it -- says nothing about the object; a
    label sharing the head noun is a naming of it, one sharing nothing is not."""
    label, asked, target = label.strip(), asked.strip(), target.strip()
    head = head_words(target)
    if not label or not target or not (object_words(asked) & head) or strip_part(asked) != asked:
        return False
    return not (object_words(label) & head)


def located_identity_gate(subgoal: Subgoal, claim: Optional[Dict[str, Any]],
                          verdict: Optional[Verdict]):
    """A take or a lift judged ``done`` while the cameras, asked for the step's own object,
    picked out something else by name: folded to ``not_done`` with the mismatch as the
    reason.

    The label the locate writes is free text and cannot tell two look-alikes apart, so this
    gate does not catch a look-alike (the box kept across replans does); it catches a reach
    that named a different thing outright. Two guards: the reach must have been FOR the
    step's own target -- a lift's last look often senses the landmark the thing stood on,
    which is not a naming of what is in the jaws -- and the label must share no word with the
    target's own head phrase, since a landmark named in the target ("...on the ramekin")
    shares a word with the whole phrase."""
    if verdict is None or verdict.status != "done" or subgoal.name == TASK:
        return verdict, ""
    if not (wants_holding(subgoal.criterion)
            or about_a_lift("{} {}".format(subgoal.subgoal, subgoal.criterion))):
        return verdict, ""
    reached = (claim or {}).get("reached") or {}
    label, target = str(reached.get("label") or "").strip(), subgoal.target.strip()
    if not names_something_else(label, str(reached.get("target") or ""), target):
        return verdict, ""
    return Verdict(status="not_done", replan=True,
                   because="the thing this step was measured onto is not the thing it names: "
                           "asked for {!r}, the cameras picked out {!r} and the tool was "
                           "taken to that, so what was taken or lifted is the wrong "
                           "object".format(target, label),
                   evidence="measured by the cameras, not claimed: the reach for {!r} "
                            "located {!r}, {:.0f} mm from the tool at the end".format(
                                target, label, float(reached.get("residual_mm") or 0.0)),
                   identity=label, clauses=list(verdict.clauses)), "located_identity"


# --------------------------------------------------------------- every clause, or not done

def check_clauses(verdict: Optional[Verdict], whole_task: bool):
    """``task_done`` is accepted only when every clause of the task is met."""
    if verdict is None or verdict.status != "task_done":
        return verdict, ""
    unmet = [c for c in verdict.clauses if not c.met]
    if verdict.clauses and not unmet:
        return verdict, ""
    if not verdict.clauses:
        why = ("task_done was answered without listing what the task asks for, clause by "
               "clause, so it is not accepted")
    else:
        why = "task_done cannot stand while {} of the task's {} clause(s) are unmet: {}".format(
            len(unmet), len(verdict.clauses),
            "; ".join((c.what.strip() or "an unnamed clause") +
                      ((" -- " + c.evidence.strip()) if c.evidence.strip() else "")
                      for c in unmet[:3]))
    # The whole-task question has no other answer: nothing is left to be "done" there.
    said = verdict.because.strip()
    return verdict.model_copy(update={
        "status": "not_done" if whole_task else "done",
        "because": why + ((" (it said: " + said + ")") if said else "")}), "clauses"


# ----------------------------------------------------------- a control that was turned back

#: Below this much turning in all, there is nothing to have undone.
_TURNED_AT_ALL_DEG = 30.0
#: A net turn smaller than this share of the turning travelled is a turn that came back.
_UNDONE_SHARE = 0.2


def control_gate(subgoal: Subgoal, claim: Optional[Dict[str, Any]],
                 verdict: Optional[Verdict]):
    """A turn of a control that was wound one way and then wound back is not done -- nor one
    that went one way only for under half the turn its step asks for. Met a stop that soon, it
    is the end of the travel the control started at; met none, it may be the grip's give
    alone (gpt6sol 107: 23-27 deg of wrist, no stop yet, the knob still at its off stop)."""
    if verdict is None or verdict.status != "done" or subgoal.name == TASK:
        return verdict, ""
    if not wants_turn("{} {}".format(subgoal.subgoal, subgoal.criterion)):
        return verdict, ""
    claim = claim or {}
    total = abs(float(claim.get("turn_total_deg") or 0.0))
    net = float(claim.get("turn_net_deg") or 0.0)
    asked = re.search(r"(\d+(?:\.\d+)?)\s*deg", subgoal.subgoal)
    if total - abs(net) < 1.0 and asked and abs(net) < 0.5 * float(asked.group(1)):
        why = "the control was turned one way only, {:.0f} of the {} deg asked, and {}".format(
            abs(net), asked.group(1), "met a stop: the end of the travel it started at -- turn "
            "it the other way" if claim.get("turn_stop") else "no stop: that can be the grip's "
            "give alone -- keep turning it the same way until that far or a stop")
    elif total < _TURNED_AT_ALL_DEG or abs(net) > _UNDONE_SHARE * total:
        return verdict, ""
    else:
        why = ("the control was turned and then turned back: since it was taken the tool has "
               "turned {:.0f} deg in all and a net {:.0f} deg, so the control is where it "
               "started".format(total, net))
    return Verdict(status="not_done", because=why,
                   evidence="measured by the robot, not claimed: net turn {:.0f} deg of "
                            "{:.0f} deg turned".format(net, total),
                   identity=verdict.identity, clauses=list(verdict.clauses),
                   replan=False), "net_turn"


def holding_contradiction(criterion: str, grasp: Optional[Dict[str, Any]],
                          verdict: Optional[Verdict]) -> bool:
    """A not_done that says the jaws are empty while the robot says they are not."""
    if verdict is None or verdict.status != "not_done":
        return False
    if (grasp or {}).get("holding") is not True or not wants_holding(criterion):
        return False
    if verdict.identity.strip() and verdict.replan:
        return False
    text = " ".join("{} {}".format(verdict.because, verdict.evidence).lower().split())
    if any(word in text for word in _EMPTY):
        return True
    # ...and the same refusal made about WHERE the jaws are on a thing they are holding, where
    # the criterion asks for nothing but the holding.
    return only_about_holding(criterion) and any(word in text for word in _LEVEL)


def _held_anyway(verdict: Optional[Verdict], grasp: Optional[Dict[str, Any]]) -> Verdict:
    """The verdict the reading settles when the model will not stop arguing with it."""
    reason = (grasp or {}).get("reason") or "the gripper's own reading"
    return Verdict(status="done",
                   because="the only ground given for not_done was that nothing is held, and "
                           "the robot's own gripper reports that it IS holding something",
                   evidence="the gripper reports holding something: {}".format(reason),
                   identity=(verdict.identity if verdict is not None else ""),
                   replan=False)


def grasp_gate(criterion: str, grasp: Optional[Dict[str, Any]]):
    """The verdict the gripper's own reading settles, or None to ask the model."""
    holding = (grasp or {}).get("holding")
    if holding is None:
        return None                       # "cannot tell" is not "no": ask
    text = " ".join(str(criterion or "").lower().split())
    if not text:
        return None
    reason = (grasp or {}).get("reason") or "the gripper's own reading"
    if wants_release(text):
        if holding is True:
            return (Verdict(status="not_done",
                            because="the robot still reports holding something, so it has not "
                                    "let go",
                            evidence="the gripper reports holding something: {}".format(reason),
                            replan=False),
                    "still holding")
        return None
    if wants_holding(text) and holding is False:
        return (Verdict(status="not_done",
                        because="the criterion is about the object the gripper should have, "
                                "and the robot reports nothing held",
                        evidence="the gripper reports nothing held: {}".format(reason),
                        replan=False),
                "nothing held")
    return None
