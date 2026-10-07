"""Task language in, a list of subgoals out -- and the same list again, mended, after a
failure."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from pydantic import ValidationError

from src.executor.context import describe_cameras
from src.executor.gates import CANNOT_FROM_THE_SIDE, Gates
from src.executor.models import wants_release
from src.executor.proposal import Attempt, _problems

from .models import BOX_SPAN, Plan, Subgoal, Verdict, acts, clean_boxes, \
    derive_gripper_only, foreign_labels, names_a_place, places, same_object, strip_part, \
    without_labels


#: A step asking for the jaws turned level in its own words, not the vocabulary's: "tilt the
#: jaws level", "level them", "until they are level", "with level, open jaws".
_TILTS_THE_JAWS = re.compile(r"\btilt(?:s|ing)?\s+(?:the\s+)?(?:jaws|them|gripper|tool|hand)\b|"
                             r"\blevel\s+them\b|\b(?:jaws|they)\s+are\s+level\b(?!\s+with)|"
                             r"\blevel,?\s+(?:open\s+)?jaw")


def side_take_unsaid(step: Subgoal) -> bool:
    """A step that tilts the jaws, or closes them on the handle of a thing that slides, and does
    not say "from the side" -- the words that alone let the robot tilt them and aim at the part
    (gpt6sol-goal-sidegrasp 100/103: every tilt refused, in every step, from the first plan)."""
    text = " ".join(step.subgoal.lower().split())
    if Gates.SIDE_WORDS in text:
        return False
    if _TILTS_THE_JAWS.search(text):
        return "vertical" not in text                     # a yaw that squares the jaws
    sliding = "|".join(Gates.SLIDING_PARTS)
    return bool(re.search(r"\bhandle of (?:the |its )?(?:[a-z]+ ){{0,3}}(?:{0})\b|\b(?:{0})(?:'s)? "
                          r"handle\b".format(sliding), text)) \
        and bool(re.search(r"\b(?:close (?:on|them|the (?:gripper )?jaws)|grasp|grip)\b", text))


@dataclass
class PlanResult:
    plan: Optional[Plan] = None
    error: str = ""
    attempts: List[Attempt] = field(default_factory=list)
    elapsed_s: float = 0.0
    # "above_can: max_cycles 30 -> 10".
    clamped: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.plan is not None and not self.error

    def as_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "error": self.error, "elapsed_s": round(self.elapsed_s, 2),
                "clamped": list(self.clamped),
                # How many rules this plan was mended by rather than asked about.
                "mended": len(self.clamped),
                "plan": None if self.plan is None else self.plan.model_dump(),
                "attempts": attempts_as_dicts(self.attempts)}


def attempts_as_dicts(attempts: List[Attempt]) -> List[Dict[str, Any]]:
    """Every round trip, with the complaint LAST."""
    return [{**a.reply, "error": a.error or a.reply.get("error", ""),
             "prompt": a.prompt, "images": a.images} for a in attempts]


def _retry_note(error: str) -> str:
    """The executor's retry shape, with its wording: the complaint, then ask again."""
    return ("\n\nYOUR PREVIOUS ANSWER WAS REJECTED\n{}\n"
            "Answer again, as one JSON object in the shape given above.".format(error))


#: How many tokens a PLAN may be.
PLAN_OUTPUT_TOKENS = 3000


def ask_model(client, base: str, system: str, images, names, model_cls, validate,
              max_attempts: int, hint=None, normalise=None, reasks: int = 2,
              max_tokens: Optional[int] = None):
    """Ask until the reply is a valid ``model_cls``, handing the complaint back each time."""
    attempts: List[Attempt] = []
    prompt, parsed = base, None
    left, spent = max(1, int(max_attempts)), {}
    # Passed only when the caller asked for one, so a client that does not take the keyword
    # is left exactly as it was: the verdicts and the alerts are short and never needed it.
    room = {} if max_tokens is None else {"max_tokens": int(max_tokens)}
    while left > 0:
        reply = client.ask_json(prompt, images=images, system=system, **room)
        attempt = Attempt(reply=reply.as_dict(), prompt_chars=len(prompt), prompt=prompt,
                          images=list(names))
        attempts.append(attempt)
        if not reply.ok:
            attempt.error = reply.error or "no JSON in the reply"
            left -= 1
        else:
            try:
                candidate = model_cls(**(normalise(reply.data) if normalise else reply.data))
            except ValidationError as exc:
                extra = hint(reply.data) if hint else None
                attempt.error = _problems(exc) + ("\n" + extra if extra else "")
                left -= 1
            else:
                complaint = validate(candidate) if validate else None
                key = ""
                if isinstance(complaint, tuple):
                    complaint, key = (complaint + ("", ""))[:2]
                if not complaint:
                    parsed = candidate
                    break
                attempt.error = complaint
                if not key:
                    left -= 1
                elif spent.get(key, 0) >= max(0, int(reasks)):
                    # Its own budget is gone.
                    parsed = candidate
                    break
                else:
                    spent[key] = spent.get(key, 0) + 1
        prompt = base + _retry_note(attempt.error)
    error = "" if parsed is not None else (attempts[-1].error if attempts
                                           else "the model was never reached")
    return parsed, attempts, error


def describe_plan(plan: Plan, verdicts: Optional[Dict[str, Verdict]] = None) -> str:
    """The plan as a numbered list with each subgoal's verdict, for prompts and for logs."""
    lines = []
    for index, step in enumerate(plan.subgoals, 1):
        verdict = (verdicts or {}).get(step.name)
        if verdict is None:
            mark = "not attempted"
        else:
            mark = verdict.status + (": " + verdict.because.strip() if verdict.because.strip()
                                     else "")
        lines.append("{}. {} [{}] -- {} (done when: {}; up to {} motions{}{}){}".format(
            index, step.name, mark, step.subgoal.strip(), step.criterion.strip(),
            step.max_cycles, "; gripper only" if step.gripper_only else "",
            # which of the task's clauses this step is for: a replan that cannot see it
            # cannot tell a chain it reordered from a chain it deleted
            "; for clause {}".format(step.clause) if step.clause >= 0 else "",
            # carried so a replan can keep a description that worked, or correct the one that
            # sent the tool to the neighbour
            "\n   target: " + step.target.strip() if step.target.strip() else ""))
    return "\n".join(lines) if lines else "the plan has no subgoals"


def _shape(data) -> Optional[str]:
    """The measured wrong shape: one subgoal where a list of them was asked for."""
    if isinstance(data, dict) and "subgoals" not in data and "subgoal" in data:
        return ('You returned one subgoal; return {"subgoals": [...], "rationale": ...}.')
    return None


def _envelope(data):
    """One subgoal, or a bare list of them, read as the plan it plainly is."""
    if isinstance(data, list):
        return {"subgoals": data, "rationale": ""}
    if isinstance(data, dict) and "subgoals" not in data \
            and ("subgoal" in data or "criterion" in data):
        # a rationale written beside the subgoal belongs to the plan, not inside the step,
        # where Subgoal's extra="forbid" would refuse it
        step = {k: v for k, v in data.items() if k != "rationale"}
        return {"subgoals": [step], "rationale": str(data.get("rationale") or "")}
    return data


#: Added to a replan prompt when this object has already cost one.
_NO_MORE_PREPARATION = """

THIS OBJECT HAS ALREADY COST A REPLAN
The step that just failed is the second failure on the same object, and the steps that were
waiting behind it have still never been run once. Another step in front of them means they
will not run this time either. So do not write a preparatory step for this object again.
Choose one of these two and say in the rationale which you chose:

  * change HOW that one step is done -- a different direction to come from, a different
    order of the same motions -- and write it ONCE, with
    the steps that were already waiting behind it following unchanged; or
  * leave this object where it stands and give the steps for the REST of the task first, with
    this object's own steps after them -- what the task asks for does not stop being asked
    for because a step failed, so those steps stay in the plan, at the end of it.

Anything else is the third preparation for a thing that has not moved yet."""


#: What a replan is told when the same object has failed twice and the new plan goes about it
#: exactly as the failed attempts did.
_CHANGE_SOMETHING = """

YOU HAVE ALREADY TRIED THIS OBJECT THIS WAY
Everything the failed attempts on this object did is listed here -- the side they came from,
and what they did when they got there:
{ways}
A step worded the way a step that failed was worded fails the same way, so change at least one
of those two: come at it from a DIFFERENT SIDE, or do a DIFFERENT THING to it (push it rather
than lift it, slide it rather than take it). Say in the rationale which one you changed. If
instead you mean to leave this object until the rest of the task is done, put the OTHER
object's steps first and this one's after them, which is a change of order and counts."""


#: ...and what it is told when it plans to open the jaws with something still in them.
_RELEASE_WHILE_HOLDING = """

THE ROBOT IS HOLDING SOMETHING RIGHT NOW
The gripper reports that it has an object in its jaws. Opening them where the arm happens to
be is dropping the thing, not putting it down, and nothing later can undo it: {name}
Either write the step that carries it to where the task wants it and put it down THERE -- with
the criterion naming that place -- or leave the jaws shut and plan the rest around what is
already held. A step that only says the jaws end up open is a drop, and it is taken out of the
plan before the plan is run."""


def _clause_lines(clauses: Sequence[str], indexes: Sequence[int]) -> str:
    """"  2 -- the drawer is shut", one per line: how a clause is named to the model."""
    return "\n".join("  {} -- {}".format(i, str(clauses[i]).strip() or "not stated")
                      for i in indexes if 0 <= i < len(clauses)) or "  (none)"


def _about(steps: Sequence[Subgoal], target: str) -> List[int]:
    """Where in a list of steps the ones about this object are, by the planner's own words."""
    return [i for i, step in enumerate(steps) if same_object(step.target, target)]


def _bare_release(plan: Plan, task: str) -> Optional[Subgoal]:
    """The first step that opens the jaws with nowhere named to put what is in them."""
    for index, step in enumerate(plan.subgoals):
        if not wants_release(step.criterion):
            continue
        if places(step.name, step.subgoal):
            continue                       # the step IS the putting down, whatever its
                                           # criterion calls the place it puts it in
        if names_a_place(step.criterion, task):
            continue
        if step.target.strip() and any(
                places(earlier.name, earlier.subgoal)
                and same_object(earlier.target, step.target)
                for earlier in plan.subgoals[:index]):
            continue
        return step
    return None


def _unboxed(step: Subgoal) -> Subgoal:
    """A step put back from an earlier plan, with its boxes taken off -- ``keep_earlier_boxes``
    then puts the earlier plan's box back on it where that box still stands."""
    return step if not step.target_boxes else step.model_copy(update={"target_boxes": {}})


def keep_earlier_boxes(steps: List[Subgoal], earlier: Optional[Plan],
                       rebox: Sequence[str] = (), since: float = 0.0):
    """The steps with the EARLIER plan's box on every object it marked from ``since`` on (the
    first picture after the jaws last took hold, lost hold or let go), in place of the one
    drawn now; the box drawn now stands for an object nothing kept marks, or one named in
    ``rebox``. Returns the steps, ``{name: capture time the kept box was drawn at}`` and one
    note per step whose box was kept."""
    if earlier is None:
        return list(steps), {}, []
    marked: List[tuple] = []          # (target, boxes, drawn at), first-marked first
    for step in earlier.subgoals:
        at = float(earlier.boxes_kept_from.get(step.name, earlier.boxed_at))
        if step.target.strip() and step.target_boxes and at >= since \
                and not any(same_object(step.target, seen) for seen, _, _ in marked):
            marked.append((step.target, dict(step.target_boxes), at))
    out, kept_from, notes = [], {}, []
    for step in steps:
        if not step.target.strip() or any(same_object(step.target, r) for r in rebox):
            out.append(step)
            continue
        found = next(((boxes, at) for seen, boxes, at in marked
                      if same_object(step.target, seen)), None)
        if found is None:
            out.append(step)
            continue
        boxes, at = found
        kept_from[step.name] = at
        if boxes != dict(step.target_boxes):
            notes.append("{}: target_boxes kept from the earlier plan ({}), the object is "
                         "the same one".format(
                             step.name, "the replan drew {}".format(dict(step.target_boxes))
                             if step.target_boxes else "the replan drew none"))
        out.append(step.model_copy(update={"target_boxes": boxes}))
    return out, kept_from, notes


def _unique_names(steps: List[Subgoal], taken=(), notes: Optional[List[str]] = None):
    """Every step under a name of its own, and none of them a name already finished."""
    seen = {str(name).strip().lower() for name in (taken or ())}
    out = []
    for step in steps:
        name = step.name.strip()
        if name.lower() not in seen:
            seen.add(name.lower())
            out.append(step)
            continue
        suffix = 2
        while "{}_{}".format(name, suffix).lower() in seen:
            suffix += 1
        fresh = "{}_{}".format(name, suffix)
        seen.add(fresh.lower())
        if notes is not None:
            notes.append("{}: name -> {!r}, that one is taken".format(name, fresh))
        out.append(step.model_copy(update={"name": fresh}))
    return out


def _camera_sizes(observation) -> Dict[str, Any]:
    """``{name: payload}`` for every camera this observation actually carries."""
    payloads = getattr(observation, "cameras", None) or {}
    out: Dict[str, Any] = {}
    for name in getattr(observation, "names", []) or []:
        out[str(name)] = payloads.get(name) or {}
    return out


def _pixels(observation, name: str, side: float = 0) -> str:
    """", 768x768 px" -- one picture's size as SENT (at most ``side``), not as rendered."""
    payload = (getattr(observation, "cameras", None) or {}).get(name) or {}
    width, height = payload.get("width"), payload.get("height")
    if not width or not height:
        return ""
    shrink = min(1.0, float(side or 1e9) / max(width, height))
    return ", {}x{} px".format(round(width * shrink), round(height * shrink))


def fixed_cameras(cameras: Sequence[str]) -> List[str]:
    """The views that do not move with the arm."""
    return [str(name) for name in cameras if "wrist" not in str(name).lower()]


#: What a prompt says when no memory note has been written yet (or there is no summariser).
NOTHING_SUMMARISED = "nothing has been summarised yet"


class Planner:
    def __init__(self, client, prompts: Dict[str, Any], budget: int = 10,
                 max_attempts: int = 3, max_subgoals: int = 24):
        self.client = client
        self.prompts = prompts
        self.budget = max(1, int(budget))
        self.max_attempts = max_attempts
        self.max_subgoals = max_subgoals
        # The latest FINISHED memory note, read when a replan is written and never waited for.
        self.note: Callable[[], str] = lambda: ""
        # replan() is not given the capability text (the mission has it once, at plan time),
        # so it is kept from the first plan rather than made up again.
        self._capability_text = ""
        # ...and the task itself, for the one mending a plan with no clauses at all needs.
        self._task = ""
        # Whether the clause list this mission is tracking was MADE here, out of the task
        # sentence, because the first plan listed none.
        self._made_clause = False
        self._cannot: set = set()

    def plan(self, task: str, observation, capability_text: str) -> PlanResult:
        self._capability_text = capability_text or self._capability_text
        self._task = task.strip() or self._task
        base = self.prompts["plan"]["text"].format(
            task=task.strip(), **self._common(observation))
        return self._run(base, self.prompts["plan"]["system"], observation, taken=(),
                         first=True)

    def replan(self, task: str, plan: Plan, verdicts: Dict[str, Verdict], observation,
               trigger: str, tail: Sequence[Subgoal] = (),
               strict_target: str = "", outstanding: Sequence[int] = (),
               remaining: Sequence[Subgoal] = (), tried_ways: Sequence[str] = (),
               holding: bool = False, rebox: Sequence[str] = (),
               since: float = 0.0) -> PlanResult:
        """The REMAINING subgoals only. The earlier plan's boxes drawn from ``since`` on are
        kept, except on the objects named in ``rebox`` (see ``keep_earlier_boxes``)."""
        self._task = task.strip() or self._task
        # The steps the robot measured it cannot do, word for word: not asked for again.
        self._cannot = {step.subgoal.strip().lower() for step in plan.subgoals
                        if CANNOT_FROM_THE_SIDE in getattr((verdicts or {}).get(step.name),
                                                           "because", "")}
        finished = {name for name, verdict in (verdicts or {}).items()
                    if verdict is not None and verdict.status in ("done", "task_done")}
        outstanding = [i for i in outstanding if 0 <= int(i) < len(plan.clauses)]
        base = self.prompts["replan"]["text"].format(
            task=task.strip(),
            plan_so_far=describe_plan(plan, verdicts),
            trigger=trigger.strip() or "no trigger was recorded",
            note=self.note().strip() or NOTHING_SUMMARISED,
            outstanding=_clause_lines(plan.clauses, outstanding)
            if outstanding else "  (nothing is outstanding: judge the task as a whole)",
            **self._common(observation))
        # The second failure on one object may not buy it a third run-up.
        if strict_target:
            base = base + _NO_MORE_PREPARATION
        if tried_ways:
            base = base + _CHANGE_SOMETHING.format(
                ways="\n".join("  * " + str(way) for way in tried_ways))
        if holding:
            base = base + _RELEASE_WHILE_HOLDING.format(
                name="any step whose criterion is only that the jaws end up open")
        # A name the caller is keeping must not come back on a new subgoal: the mission files
        # verdicts and evidence by name, and a collision would overwrite a finished step.
        return self._run(base, self.prompts["replan"]["system"], observation, taken=finished,
                         tail=tail, strict_target=strict_target, clauses=plan.clauses,
                         outstanding=outstanding, remaining=remaining,
                         holding=bool(holding), earlier=plan, rebox=tuple(rebox),
                         since=since)

    # ------------------------------------------------------------------ the shared middle

    def _common(self, observation) -> Dict[str, Any]:
        names = list(getattr(observation, "names", []) or [])
        side = getattr(self.client, "image_side", 0)
        return {"budget": self.budget,
                "soft": max(1, self.budget // 2),
                "max_subgoals": self.max_subgoals,
                "vocabulary": self.prompts.get("vocabulary", ""),
                "capabilities": self._capability_text
                or "The executor's capabilities were not supplied; assume only that the arm "
                   "moves a few centimetres per motion and has a two-finger gripper.",
                "robot": (getattr(observation, "robot_text", "") or "").strip()
                or "the robot reported nothing about itself",
                "cameras": describe_cameras(
                    ["{} (now{})".format(n, _pixels(observation, n, side)) for n in names],
                    "A box you draw is in 0-{} across and 0-{} down whatever those pixel "
                    "sizes are.".format(BOX_SPAN, BOX_SPAN)),
                "legend": (getattr(observation, "legend", "") or "").strip()
                or "No camera legend is available, so do not name a robot direction yourself "
                   "-- the executor resolves directions from its own legend."}

    def _run(self, base: str, system: str, observation, taken, tail: Sequence[Subgoal] = (),
             strict_target: str = "", first: bool = False,
             clauses: Sequence[str] = (), outstanding: Sequence[int] = (),
             remaining: Sequence[Subgoal] = (),
             holding: bool = False, earlier: Optional[Plan] = None,
             rebox: Sequence[str] = (), since: float = 0.0) -> PlanResult:
        started = time.monotonic()
        images = list(getattr(observation, "images", []) or [])
        names = ["{} (now)".format(n) for n in getattr(observation, "names", []) or []]
        # Which pictures the boxes in this plan will be about, and when they were taken.
        cameras = _camera_sizes(observation)
        capture = float(getattr(observation, "capture_time", 0.0) or 0.0)
        # Two complaints that are asked ONCE and then mended in code.
        once: Dict[str, int] = {}
        plan, attempts, error = ask_model(
            self.client, base, system, images, names, Plan,
            lambda candidate: self._complain(candidate, first, clauses, outstanding,
                                             once, cameras),
            self.max_attempts, _shape, normalise=_envelope,
            max_tokens=PLAN_OUTPUT_TOKENS)
        result = PlanResult(error=error, attempts=attempts)
        if plan is not None:
            result.plan, result.clamped = self._settle(plan, tail, strict_target, first,
                                                       clauses, outstanding, remaining,
                                                       holding, taken, cameras, capture,
                                                       earlier, rebox, since)
            if not result.plan.subgoals and first:
                # The one rule nothing can mend: a FIRST plan with no steps in it.
                result.plan, result.error = None, (
                    (attempts[-1].error if attempts else "")
                    or "the plan has no subgoals and there is nothing to put back")
        result.elapsed_s = time.monotonic() - started
        return result

    def _complain(self, plan: Plan, first: bool = False, clauses: Sequence[str] = (),
                  outstanding: Sequence[int] = (), once: Optional[Dict[str, int]] = None,
                  cameras: Optional[Dict[str, Any]] = None):
        """The complaints that are worth a round trip -- and only those."""
        if not plan.subgoals and first:
            return ("The plan has no subgoals. Give at least one, and enough of them that each "
                    "is a single reach, descent, gripper action or lift.", "empty")
        if not plan.subgoals:
            # A REPLAN with no steps is an answer, not a shape: it says nothing is left to do.
            # Complaining only buys filler steps; the task's own final check settles it.
            return None
        if len(plan.subgoals) > self.max_subgoals:
            return ("The plan has {} subgoals and at most {} are allowed. Merge the steps that "
                    "are really one motion, or plan only as far as the first object."
                    .format(len(plan.subgoals), self.max_subgoals), "size")
        for step in plan.subgoals:
            if not step.name.strip() or not step.subgoal.strip() or not step.criterion.strip():
                return ("Every subgoal needs a non-empty name, subgoal and criterion; {!r} is "
                        "missing one of them.".format(step.name), "empty")
        unmarked = self._unmarked(plan, cameras)
        if unmarked and not (once or {}).get("boxes"):
            # The steps that name an object and marked it nowhere.
            if once is not None:
                once["boxes"] = 1
            return ("These steps name an object and do not say where it is in any of the "
                    "fixed pictures: {}. For each of them put a `target_boxes` entry under "
                    "every fixed camera ({}) that can SEE that object -- "
                    "[x0, y0, x1, y1], 0-{} across and 0-{} down, tight round the whole "
                    "object including where it meets the table -- and leave out any camera "
                    "that cannot see it. All the steps about one object carry the same "
                    "boxes, as they carry the same `target`."
                    .format(", ".join(repr(name) for name in unmarked[:6]),
                            ", ".join(repr(n) for n in fixed_cameras(cameras or {})) or
                            "the pictures named above", BOX_SPAN, BOX_SPAN),
                    "boxes")
        unsaid = [s.name for s in plan.subgoals if side_take_unsaid(s)]
        if unsaid:
            return ("{} take hold of the handle of something that slides, or tilt the jaws, "
                    "without saying it is taken from the side -- the only words that let the "
                    "robot tilt them. Write each as the vocabulary's own step, \"take the handle "
                    "of <the drawer or door> from the side and close on it\": one step that "
                    "makes its own approach, with no step before it that tilts the jaws."
                    .format(", ".join(repr(name) for name in unsaid)), "side")
        if first and not plan.clauses:
            # Nothing at all was listed, so every rule about which step serves which part of the
            # task would be off for the whole mission.
            if not (once or {}).get("clauses"):
                if once is not None:
                    once["clauses"] = 1
                return ("The plan lists nothing under `clauses`. That list is what the task "
                        "ASKS FOR, one entry per thing that has to be true at the end, and "
                        "every step carries the index of the one it serves -- it is how a "
                        "later plan is stopped from quietly dropping half the task. Read the "
                        "task sentence again, write one entry for each thing it asks for, and "
                        "give the same plan back with every step's `clause` filled in.",
                        "clauses")
        # A clause with no step is asked about by name -- on a replan too, where a step
        # labelled with a finished clause's index is dropped before it runs. A clause whose
        # steps never act is asked about on the FIRST plan only: a replan that acts on nothing
        # is its answer that nothing is left to do, and the task's own question settles it.
        missing = self._uncovered(plan, first, clauses, outstanding)
        if missing:
            return ("These parts of the task have no subgoal at all:\n{}\nEvery one needs at "
                    "least one step that serves it, and every step about one carries its index "
                    "in `clause`. Give the plan again with those steps in it."
                    .format(_clause_lines(plan.clauses if first else clauses, missing)),
                    "clause")
        mine = [[s for s in plan.subgoals if s.clause == index]
                for index in range(len(plan.clauses) if first else 0)]
        idle = [index for index, steps in enumerate(mine)
                if steps and not any(acts(s.name, s.subgoal) for s in steps)]
        if idle:
            return ("These parts of the task have steps that only look at the thing or go "
                    "near it, and nothing that ever touches it:\n{}\nAt least one step "
                    "for each has to DO something -- close the jaws on it, lift it, carry "
                    "it, put it down, let it go, push it or turn it -- and carry that "
                    "clause's index.".format(_clause_lines(plan.clauses, idle)), "act")
        # No complaint about going at the same object the same way: the ways already tried
        # are listed to the replanner in its prompt -- unless the robot measured it cannot be.
        again = [s.name for s in plan.subgoals if s.subgoal.strip().lower() in self._cannot]
        if again and once is not None and not once.get("cannot"):
            once["cannot"] = 1
            return ("{} asks word for word for what the robot measured it cannot do (see the "
                    "plan so far): write what it is for another way, or leave it out."
                    .format(", ".join(repr(name) for name in again)), "cannot")
        return None

    def _unmarked(self, plan: Plan, cameras: Optional[Dict[str, Any]] = None) -> List[str]:
        """The steps that name an object and drew no box for it in any FIXED view."""
        fixed = set(fixed_cameras(cameras or {}))
        out = []
        for step in plan.subgoals:
            if not step.target.strip():
                continue
            boxes, _ = clean_boxes(step.target_boxes, fixed)
            if not boxes:
                out.append(step.name)
        return out

    def _uncovered(self, plan: Plan, first: bool, clauses: Sequence[str],
                   outstanding: Sequence[int]) -> List[int]:
        """Which clauses this plan leaves with no step at all -- the whole of the rule."""
        served = {step.clause for step in plan.subgoals}
        if first:
            return [i for i in range(len(plan.clauses)) if i not in served]
        return [i for i in outstanding
                if 0 <= int(i) < len(clauses) and int(i) not in served]

    def _settle(self, plan: Plan, tail: Sequence[Subgoal] = (), strict_target: str = "",
                first: bool = False, clauses: Sequence[str] = (),
                outstanding: Sequence[int] = (), remaining: Sequence[Subgoal] = (),
                holding: bool = False, taken=(), cameras: Optional[Dict[str, Any]] = None,
                capture: float = 0.0, earlier: Optional[Plan] = None,
                rebox: Sequence[str] = (), since: float = 0.0):
        """Everything the plan is mended for rather than asked about, and the record of it."""
        notes, subgoals = [], []
        known = list(plan.clauses) if first else list(clauses)
        if first:
            self._made_clause = False
        made_a_clause = self._made_clause
        if first and not known:
            # A first plan that listed nothing the task asks for.
            known = [self._task.strip() or "what the task asks for"]
            made_a_clause, self._made_clause = True, True
            notes.append("clauses: none were listed, so the task itself is the one clause and "
                         "every step is for it")
        for step in plan.subgoals:
            update: Dict[str, Any] = {}
            capped = max(1, min(self.budget, int(step.max_cycles)))
            if capped != step.max_cycles:
                notes.append("{}: max_cycles {} -> {}".format(step.name, step.max_cycles, capped))
                update["max_cycles"] = capped
            quoted = foreign_labels(step.target, self._task)
            whole = strip_part(without_labels(step.target, quoted) if quoted
                               else step.target)
            if quoted:
                notes.append("{}: target dropped the quoted {} -- it is not this object's "
                             "writing".format(step.name,
                                              ", ".join(repr(q) for q in quoted)))
            if whole.strip().lower() != step.target.strip().lower():
                notes.append("{}: target {!r} -> {!r}".format(step.name, step.target, whole))
                update["target"] = whole
            # ...and the boxes that go with that phrase.
            wanted = list(cameras or {})
            if not (update.get("target", step.target) or "").strip():
                if step.target_boxes:
                    notes.append("{}: target_boxes dropped, the step names no object"
                                 .format(step.name))
                    update["target_boxes"] = {}
            else:
                boxes, dropped = clean_boxes(step.target_boxes, wanted)
                for said in dropped:
                    notes.append("{}: {}".format(step.name, said))
                if boxes != dict(step.target_boxes):
                    update["target_boxes"] = boxes
            if made_a_clause:
                update["clause"] = 0
            elif step.clause >= len(known) or step.clause < -1:
                # An index into a list that is not there points at nothing.
                notes.append("{}: clause {} -> -1, there is no such clause".format(
                    step.name, step.clause))
                update["clause"] = -1
            jaws = derive_gripper_only(step.subgoal, step.criterion)
            if jaws != bool(step.gripper_only):
                notes.append("{}: gripper_only {} -> {}".format(step.name, bool(step.gripper_only),
                                                                jaws))
                update["gripper_only"] = jaws
            subgoals.append(step.model_copy(update=update) if update else step)
        settled = plan.model_copy(update={"subgoals": subgoals,
                                          "clauses": known,
                                          # not the model's to write: which pictures the
                                          # boxes above are about, and when they were taken
                                          "boxed_on": list(cameras or {}),
                                          "boxed_at": float(capture),
                                          # a FIRST plan is written before the robot has
                                          # moved and cannot know that nothing will work;
                                          # only a replan may give up
                                          "recoverable": True if first else plan.recoverable})
        # When the one clause was made here (the plan listed none), the coverage rules stay
        # off: a made-up clause covering everything knows nothing about which step serves what.
        subgoals = list(settled.subgoals)
        asked = set(range(len(known)) if first else outstanding)
        for index in ([] if made_a_clause
                      else self._uncovered(settled, first, known, outstanding)):
            # The chain for something the task still asks for is not in this plan. A step that
            # ACTS on that clause's object under another index -- none, or one already
            # finished -- is its chain mislabelled; an approach or a retreat never is.
            earlier_steps = [step for step in (earlier.subgoals if earlier else ())
                             if step.clause == index]
            adopted = False
            for position, kept in enumerate(subgoals):
                if kept.clause in asked or not acts(kept.name, kept.subgoal):
                    continue
                if any(kept.name == step.name or (kept.target.strip() and step.target.strip()
                                                  and same_object(kept.target, step.target))
                       for step in earlier_steps):
                    notes.append("{}: clause {} -> {}, it is the step for something the task "
                                 "still asks for".format(kept.name, kept.clause, index))
                    subgoals[position] = kept.model_copy(update={"clause": index})
                    adopted = True
            if adopted:
                continue
            for step in [step for step in remaining if step.clause == index]:
                if any(step.name == kept.name for kept in subgoals):
                    continue
                notes.append("{}: put back (without its boxes, which were drawn on an "
                             "earlier picture), the plan dropped every step for something "
                             "the task still asks for".format(step.name))
                subgoals.append(_unboxed(step))
        settled = settled.model_copy(update={"subgoals": subgoals})
        # A step that opens the jaws with something in them and nowhere named to put it is a
        # drop.
        if holding:
            while True:
                bare = _bare_release(settled, self._task)
                if bare is None:
                    break
                notes.append("{}: dropped, it opens the jaws on something that is being held "
                             "with no place named to put it".format(bare.name))
                subgoals = [step for step in settled.subgoals if step.name != bare.name]
                settled = settled.model_copy(update={"subgoals": subgoals})
        # A mending that leaves nothing at all is not a mending.
        if not subgoals and remaining:
            for step in remaining:
                notes.append("{}: put back (without its boxes), the mended plan had no steps "
                             "left in it".format(step.name))
            subgoals = [_unboxed(step) for step in remaining]
            settled = settled.model_copy(update={"subgoals": subgoals})
        plan, subgoals = settled, list(settled.subgoals)
        # A name is how a verdict, an evidence row and a replan's idea of what is finished
        # are filed, so two alike lose a step -- but which of the two keeps the name is not
        # something only the model can decide, and a plan refused for a name is a plan lost.
        subgoals = _unique_names(subgoals, taken, notes)
        if strict_target:
            here = _about(subgoals, strict_target)
            extra = max(0, len(here) - len(_about(tail, strict_target)) - 1)
            # The preparation is written in front of what it prepares, so the steps to drop
            # are the first ones that do not act: what is left is the changed step and the tail.
            for index in reversed([i for i in here if not acts(subgoals[i].name,
                                                                subgoals[i].subgoal)][:extra]):
                notes.append("{}: dropped, a second preparation for the same object"
                             .format(subgoals[index].name))
                subgoals.pop(index)
        # A replan for an object the earlier plan already marked keeps that mark: a re-drawn
        # box tends to land on a look-alike neighbour. A mark drawn before the jaws last took
        # hold, lost hold or let go is not kept, nor one for an object named in ``rebox``.
        kept_from: Dict[str, float] = {}
        if not first:
            subgoals, kept_from, kept_notes = keep_earlier_boxes(subgoals, earlier, rebox,
                                                                 since)
            notes.extend(kept_notes)
        return plan.model_copy(update={"subgoals": subgoals,
                                       "boxes_kept_from": kept_from}), notes

