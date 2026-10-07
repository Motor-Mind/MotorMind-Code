"""The whole loop: plan, run one subgoal, judge it, summarise it in the background, replan.

Almost everything here is a bound rather than a behaviour: the models decide what to do, the
mission decides when to stop asking them. Four bounds end one whatever they say -- the
per-subgoal motion budget, one retry before a replan, ``max_replans``, and a cycle ceiling of
``spent + 2 x sum(the remaining subgoals)`` recomputed at every replan -- and under them all
the mission clock. The monitor is attached after construction, because it reads
``Mission.latest`` and calls ``Mission.on_alert``.
"""

from __future__ import annotations

import inspect
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.executor.gates import CANNOT_FROM_THE_SIDE, jaw_resolution_mm
from src.executor.loop import FROZEN_ENDING, ExecutorLoop
from src.executor.models import wants_holding, wants_release

from .models import _PRINTED, Alert, Plan, Subgoal, Verdict, acts, gave_up_reason, \
    object_words, outstanding_clauses, same_object, ways, strip_part, \
    wants_turn
from .evidence import EvidenceLog
from .monitor import SceneMonitor
from .planner import Planner, _unboxed, describe_plan
from .verify import TASK, VerdictResult, Verifier, names_something_else, task_question

def accepted_kwargs(call, **candidates) -> Dict[str, Any]:
    """Only the keyword arguments ``call`` NAMES -- and ``**kwargs`` is not a name."""
    try:
        parameters = inspect.signature(call).parameters
    except (TypeError, ValueError):        # a builtin or a C callable: offer nothing
        return {}
    allowed = {name for name, p in parameters.items()
               if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                             inspect.Parameter.KEYWORD_ONLY)}
    return {name: value for name, value in candidates.items() if name in allowed}


#: How many times a plan may be rewritten before the mission is over -- a RAIL, not a budget.
DEFAULT_MAX_REPLANS = 10

#: The mission's whole wall clock, in seconds, from the first plan call.
MISSION_BUDGET_S = 360.0

#: How long a finished mission waits for the note still being written, off its own clock.
NOTE_JOIN_S = 30.0

#: What a replan has to have time to be worth starting: one plan call and one subgoal.
REPLAN_NEEDS_S = 40.0

#: A subgoal about to run, priced at no more than the median of those that have run, and what
#: the verdict ending it costs on top (the note is written off the clock): LIBERO's, the
#: fallback where the robot declares no Capabilities.subgoal_median_s / subgoal_verdict_s.
SUBGOAL_MEDIAN_S = 21.0
SUBGOAL_TAX_S = 4.6

#: How the source of a clause row says it came from a STEP's own task question rather than
#: from the check at the end of the plan (see ``run``), which is what makes it a stale reading
#: once that check has answered.
_PER_STEP = "(the task's own question)"
#: ...and that it came from what the jaws measured, which no reader's answer replaces.
_MEASURED = "(the jaws' own reading)"


def what_the_outstanding_clauses_need(remaining: Sequence[Any], rows: Sequence[Any],
                                      outstanding: Sequence[int], median_s: float = SUBGOAL_MEDIAN_S,
                                      tax_s: float = SUBGOAL_TAX_S) -> Dict[int, Dict[str, Any]]:
    """What every clause still outstanding would cost to finish, one entry each."""
    first_cost, spent_on, started = {}, {}, set()
    for name, clause, elapsed in rows:
        clause, name = int(clause), str(name)
        started.add(clause)
        if name not in first_cost:
            first_cost[name] = float(elapsed or 0.0)
            spent_on[clause] = spent_on.get(clause, 0.0) + first_cost[name]
    per_clause = {}
    for index in sorted({int(i) for i in outstanding}):
        left = [str(name) for name, clause in remaining if int(clause) == index]
        if index not in started:
            # one median subgoal and its verdict: the median, not the cycle cap -- the worst
            # case refused replans that had time left
            needs, state = median_s + tax_s, "never started"
        elif left:
            needs = sum(min(first_cost.get(n, median_s), median_s) for n in left)
            state = "{} step(s) left".format(len(left))
        else:
            needs, state = spent_on.get(index, 0.0), "started, no step left to run"
        per_clause[index] = {"clause": index, "state": state, "needs_s": round(needs, 1)}
    return per_clause


@dataclass
class MissionResult:
    task: str = ""
    status: str = "error"
    error: str = ""
    budget: int = 10
    #: the wall clock this mission was given, and what was left of it when it stopped
    budget_s: float = MISSION_BUDGET_S
    left_s: float = 0.0
    plans: List[Dict[str, Any]] = field(default_factory=list)
    subgoals: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[Dict[str, Any]] = field(default_factory=list)
    total_cycles: int = 0
    elapsed_s: float = 0.0
    log: Optional[EvidenceLog] = None
    # The planner said there was nothing left to try, and the mission stopped rather than spend
    # its remaining replans on a scene it had been told could not be recovered.
    gave_up: bool = False
    gave_up_because: str = ""
    #: The mission's per-clause state as it ended: which clauses have been seen met, the LATEST
    #: answer about each with its evidence, every clause an answer took back, the EVENT the code
    #: settled behind each that has one, and every mark refused for having none.
    clauses: Dict[str, Any] = field(default_factory=dict)
    #: Set when the clock, not the plan, is what ended the mission: a replan was outstanding and
    #: there was not enough of the run left to pay for it.
    replan_refused_for_price: Dict[str, Any] = field(default_factory=dict)
    #: WHICH ending this was, for the two that both say ``timeout``: a replan refused because the
    #: clock cannot pay for the clauses outstanding, or the clock running out inside a subgoal.
    ended_on: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"task": self.task, "status": self.status, "error": self.error,
                "gave_up": bool(self.gave_up), "gave_up_because": self.gave_up_because,
                "budget": self.budget, "budget_s": round(self.budget_s, 1),
                "left_s": round(self.left_s, 1), "plans": list(self.plans),
                "subgoals": list(self.subgoals), "notes": list(self.notes),
                "clauses": dict(self.clauses),
                "replan_refused_for_price": dict(self.replan_refused_for_price),
                "ended_on": self.ended_on,
                "total_cycles": self.total_cycles, "elapsed_s": round(self.elapsed_s, 1),
                "log": {"rows": []} if self.log is None else self.log.as_dict()}


class Mission:
    def __init__(self, planner: Planner, verifier: Verifier,
                 monitor: Optional[SceneMonitor], log: EvidenceLog, loop_factory: Callable[[Subgoal], Any],
                 observe: Callable[[], Any], capability_text: str, budget: int = 10,
                 max_replans: int = DEFAULT_MAX_REPLANS, ground_truth: Optional[Callable[[], Dict[str, Any]]] = None,
                 budget_s: float = MISSION_BUDGET_S, summarizer=None,
                 subgoal_s: Optional[Sequence[Optional[float]]] = None):
        self.planner = planner
        self.verifier = verifier
        self.monitor = monitor              # public: the caller usually attaches it after build
        # Writes the memory note on its own thread: nobody waits for it, and a replan or a
        # verdict reads whichever note last FINISHED (see ``_request_note``).
        self.summarizer = summarizer
        self._note_lock = threading.Lock()
        self._note_thread: Optional[threading.Thread] = None
        self._note_asked: Optional[tuple] = None    # (task, plan text) a finished note owes
        self._note, self._note_text, self._note_version, self._note_since = None, "", 0, 0
        self._notes: List[Dict[str, Any]] = []
        self.log = log
        self.loop_factory = loop_factory
        self.observe = observe
        self.capability_text = capability_text
        self.budget = max(1, int(budget))
        self.max_replans = max(0, int(max_replans))
        # The mission's wall clock, and the only budget there is: a subgoal runs until the
        # executor says done, the stall rule fires or this passes.
        self.budget_s = float(budget_s)
        # ...and what a subgoal and its verdict take on this robot, which price a replan.
        median, verdict = subgoal_s or (None, None)
        self._subgoal_s = (float(median or SUBGOAL_MEDIAN_S), float(verdict or SUBGOAL_TAX_S))
        self._deadline: Optional[float] = None
        # Read by the eval after every subgoal and never shown to any model: the disagreement
        # between the verifier and MuJoCo is the finding, and it is lost if only one is kept.
        self.ground_truth = ground_truth

        self._on_event: Optional[Callable[[Dict[str, Any]], None]] = None
        self._latest: Optional[Any] = None          # the tee's slot, read by the monitor
        self._cancelled = threading.Event()
        self._current: Dict[str, Any] = {}          # the subgoal now running, for on_alert
        self._last_now: Optional[Any] = None        # the mission's own last picture
        # What the last executor measured about its object, offered to the next subgoal.
        self._carry: Dict[str, Any] = {}
        # What the last attempt at each subgoal ended as, so the next one can be told.
        self._tried: Dict[str, Dict[str, Any]] = {}
        # How many replans each object has already cost, by the planner's own phrase for it.
        self._replans_for: Dict[str, int] = {}
        # Which of the task's clauses the verifier has said are MET.
        self._met: set = set()
        # ...the LATEST answer about each clause and what it read, and what was taken back.
        self._clause_said: Dict[int, Dict[str, Any]] = {}
        self._retractions: List[Dict[str, Any]] = []
        # Which objects have already been gone back for once after the jaws lost them.
        self._regrasped: Dict[str, int] = {}
        # The net turn and the net travel of the tool since it last took hold of something --
        # what a control it is holding is judged by. Reset by a grasp and by a loss.
        self._since_grasp: Dict[str, Any] = no_motion_yet()
        # Where each object's FIRST plan-time box was measured (a re-drawn one can sit on a
        # look-alike); the capture time of the first picture after the jaws last took hold,
        # lost hold or let go -- no plan box drawn before it is kept by a replan.
        self._marks, self._hold_at = {}, 0.0
        # What the CODE settled about each clause.
        self._settled: Dict[int, Dict[str, Any]] = {}
        # ...and every mark refused for having no event, so a sweep can count them.
        self._refused_marks: List[Dict[str, Any]] = []
        # Which clauses an answer has TAKEN BACK, and whether the last question was re-asked.
        self._retracted: set = set()
        self._reasked = False

    # ------------------------------------------------------------------ what the monitor uses

    def latest(self) -> Optional[Any]:
        """The most recent observation the EXECUTOR took. The monitor's only window."""
        return self._latest

    def on_alert(self, alert: Alert, meta: Optional[Dict[str, Any]] = None) -> None:
        """Called from the monitor's thread for every valid alert, ``fine`` included."""
        current = self._current
        if not current or alert.level == "fine":
            return
        # The prompt is thousands of characters and the pictures are gone: neither is logged.
        meta = {k: v for k, v in (meta or {}).items() if k != "prompt"}
        alert = _not_a_stop(alert, current, (getattr(self._latest, "grasp", None) or {})
                            .get("holding") is True)
        name = current["name"]
        current["alerts"].append(alert)
        text = "the monitor said {} ({}): {}".format(
            alert.level, alert.finding, alert.because.strip() or "no reason given")
        self.log.record("alert", name, text,
                        {"level": alert.level, "finding": alert.finding, **meta})
        self._emit({"event": "alert", "subgoal": name, "level": alert.level,
                    "finding": alert.finding, "because": alert.because, **meta})
        if alert.level == "stop":
            current["stop"] = alert
            loop = current.get("loop")
            if loop is not None:
                loop.cancel()

    def cancel(self) -> None:
        """Stop the mission from outside: the page's STOP button, or a test."""
        self._cancelled.set()
        loop = (self._current or {}).get("loop")
        if loop is not None:
            loop.cancel()
        if self.monitor is not None:
            self.monitor.stop()

    # ------------------------------------------------------------------ the loop

    def run(self, task: str, on_event: Optional[Callable[[Dict[str, Any]], None]] = None):
        started = time.monotonic()
        self._deadline = started + self.budget_s
        self._on_event = on_event
        # NOT cleared: a mission that was cancelled stays cancelled, so a stop that lands
        # between building the mission and starting it is honoured rather than forgotten.
        self._tried, self._replans_for, self._regrasped = {}, {}, {}
        self._met, self._clause_said, self._retractions = set(), {}, []
        self._since_grasp = no_motion_yet()
        self._marks, self._hold_at = {}, 0.0
        self._settled, self._refused_marks, self._retracted = {}, [], set()
        self._reasked = False
        self._note, self._note_text, self._note_version = None, "", 0
        self._note_since, self._notes = len(self.log.rows), []
        if self.summarizer is not None:
            self.planner.note = self.verifier.note = lambda: self._note_text
        result = MissionResult(task=task, budget=self.budget, budget_s=self.budget_s,
                               log=self.log)
        verdicts: Dict[str, Verdict] = {}

        if self._cancelled.is_set():
            return self._finish(result, "cancelled", started)

        picture = self.observe()
        plan_result = self.planner.plan(task, picture, self.capability_text)
        result.plans.append({"version": 1, **plan_result.as_dict()})
        if not plan_result.ok or plan_result.plan is None:
            return self._finish(result, "error", started,
                                error=plan_result.error or "the planner never returned a plan")
        plan, version, replans = plan_result.plan, 1, 0
        # How many times a replan has answered that there is nothing left to do.
        said_done = 0
        self._record_plan(version, plan, plan_result.clamped, "the first plan")
        self._emit({"event": "planned", "version": version,
                    "plan": plan.model_dump(), "clamped": list(plan_result.clamped),
                    "elapsed_s": round(plan_result.elapsed_s, 2)})
        ceiling = self._ceiling(0, plan.subgoals)

        index, attempt = 0, 1
        while True:
            if self._cancelled.is_set():
                return self._finish(result, "cancelled", started)

            if index >= len(plan.subgoals):
                # The plan ran out with every subgoal judged done and nobody saying the task
                # was -- a run that SUCCEEDED, or one whose plan was missing a step.
                final = self._final_check(task, plan, picture, result)
                if self._cancelled.is_set():    # a STOP that landed during that model call
                    return self._finish(result, "cancelled", started)
                verdict = final.verdict
                if verdict is None:
                    return self._finish(result, "exhausted", started,
                                        error="every subgoal was judged done and the final "
                                              "task-level check never answered: {}".format(
                                                  final.error or "no reason given"))
                self._fold(plan, verdict, "the task-level check", retract=True)
                missing = [] if verdict.status != "task_done" \
                    else self._unmet_clauses(plan, verdict)
                if verdict.status == "task_done" and not missing:
                    return self._finish(result, "task_done", started)
                if verdict.status == "abort":
                    return self._finish(result, "aborted", started, error=verdict.because)
                if not self._unmet_clauses(plan, verdict):
                    # It says unfinished, and every clause has been seen met with its evidence.
                    return self._finish(result, "exhausted", started,
                                        error="the task-level check says the task is not "
                                              "complete ({}) while every clause of it has "
                                              "been seen met, so there is nothing left to "
                                              "plan".format(verdict.because.strip()
                                                            or "no reason given"))
                # done and not_done answer THIS question alike: the plan missed a step. Mend it.
                trigger = ("every planned subgoal was verified done but the task is not "
                           "complete: {}".format(self._partial(result, missing)
                                                 or verdict.because.strip()
                                                 or "no reason given"))
                keep, exhausted = len(plan.subgoals), trigger
                tail, strict, tried_ways, rebox = (), "", [], []
                # Nothing is outstanding by construction: every step was judged done.
            else:
                if result.total_cycles >= ceiling:
                    return self._finish(result, "exhausted", started,
                                        error="the plan's whole motion budget ({} cycles) was "
                                              "spent".format(ceiling))

                subgoal = plan.subgoals[index]
                row = self._attempt(task, subgoal, attempt, plan, verdicts)
                result.subgoals.append(row)
                result.total_cycles += len(row["cycles"])
                if self._cancelled.is_set():
                    return self._finish(result, "cancelled", started)
                if row["error"]:
                    if not row.get("episode_over"):
                        return self._finish(result, "error", started, error=row["error"])
                    # The robot cannot move again, so there is nothing to plan and nothing to
                    # retry -- but on this backend an episode also ends the moment it SUCCEEDS.
                    final = self._final_check(task, plan, picture, result,
                                              stopped=row["error"])
                    if self._cancelled.is_set():
                        return self._finish(result, "cancelled", started)
                    if final.verdict is not None:
                        self._fold(plan, final.verdict, "the task-level check", retract=True)
                    over = final.verdict is not None \
                        and final.verdict.status == "task_done"
                    missing = self._unmet_clauses(plan, final.verdict) if over else []
                    if over and not missing:
                        return self._finish(result, "task_done", started)
                    said = self._partial(result, missing) \
                        or (final.verdict.because if final.verdict is not None
                            else final.error) or "it gave no reason"
                    return self._finish(result, "exhausted", started,
                                        error="{} -- judged as it stands: {}".format(
                                            row["error"], said.strip()))

                verdict = row["verdict"].get("verdict") if row["verdict"] else None
                status = (verdict or {}).get("status") or "not_done"
                wants_replan = bool((verdict or {}).get("replan"))
                verdicts[subgoal.name] = Verdict(**verdict) if verdict else \
                    Verdict(status="not_done", because="the verifier never answered validly")
                # What the ROBOT settled here, written before any answer about it is folded.
                self._note_settled(plan, subgoal, row)
                # What the verifier has SEEN done.
                self._fold(plan, verdicts[subgoal.name], subgoal.name,
                           retract=verdicts[subgoal.name].status == "task_done")
                asked = (row.get("verdict") or {}).get("asked")
                if asked:
                    self._fold(plan, Verdict(**asked),
                               "{} {}".format(subgoal.name, _PER_STEP), retract=True)

                if status == "task_done":
                    # A step's picture (often the release moment) ends nothing: the step is done
                    # and the plan runs on to its final check, on a scene that has settled.
                    self._partial(result, self._unmet_clauses(plan, verdicts[subgoal.name]))
                    status = "done"
                if status == "abort":
                    return self._finish(result, "aborted", started,
                                        error=(verdict or {}).get("because", ""))
                if self._left() <= 0.0:
                    # The clock, not a cap: the last verdict, if the step got one in time.
                    return self._finish(result, "timeout", started,
                                        error="the {:.0f} s mission clock ran out during {}: "
                                              "{}".format(self.budget_s, subgoal.name,
                                                          (verdict or {}).get("because", "")
                                                          or "not judged, no time was left"))
                if status == "done":
                    index, attempt = index + 1, 1
                    continue
                if not wants_replan and attempt == 1 and not row.get("never_moved"):
                    # One retry of the same subgoal, note in hand -- unless the attempt ended
                    # with the tool where it was, which a second go from the same pose does
                    # not change: the replan is told instead.
                    attempt = 2
                    continue

                if row.get("lost_grasp"):
                    # The jaws lost the thing they set out with.
                    back = self._regrasp_index(plan, index, subgoal)
                    if back is not None:
                        # ...without its box or mark: both are from before it was taken.
                        take = plan.subgoals[back] = _unboxed(plan.subgoals[back])
                        self._marks.pop(self._object_key(take.target), None)
                        self._lost_after(take, subgoal, row)
                        for later in plan.subgoals[back:index + 1]:
                            verdicts.pop(later.name, None)
                        self.log.record("verdict", take.name,
                                        "the object was lost in {}: going back to the step "
                                        "that took it rather than replanning".format(
                                            subgoal.name),
                                        {"lost_grasp": True, "from": subgoal.name})
                        self._emit({"event": "regrasp", "subgoal": take.name,
                                    "after": subgoal.name})
                        index, attempt = back, self._next_attempt(result, take.name)
                        continue

                # not_done with nothing left to try here: mend the rest of the plan.
                trigger = "{} was judged not_done after {} attempt(s): {}{}{}".format(
                    subgoal.name, attempt,
                    (verdict or {}).get("because", "") or "no reason given",
                    # A step that never moved the arm failed differently from one that misaimed.
                    ("\nThe tool ended that attempt where it was: {}".format(
                        str(row.get("never_moved") or row.get("frozen"))[:200])
                     if row.get("never_moved") or row.get("frozen") else ""),
                    # ...and what the earlier steps about the same object were judged.
                    _earlier_failures(result, subgoal.target, subgoal.name))
                keep, exhausted = index, ("{} replans were allowed and all were used"
                                          .format(self.max_replans))
                # Everything after the failed step was never reached.
                tail = tuple(plan.subgoals[index + 1:])
                object_key = self._object_key(subgoal.target)
                strict = object_key if self._replans_for.get(object_key, 0) else ""
                # ...and what it may not write again: every side and mode already tried -- at
                # once, where the robot measured that a step on it cannot be done.
                tried_ways = ways_already_tried(result, subgoal.target) \
                    if strict or (row["verdict"] or {}).get("gate") == "cannot" else []
                if object_key:
                    self._replans_for[object_key] = self._replans_for.get(object_key, 0) + 1
                # ...and whether the replan may re-draw this object's box: the look lost it, or
                # the verdict read printing naming another thing on it. A description quibble in
                # `identity` is not that: 128 of the 138 such re-draws in r4full were in failures.
                said = verdicts[subgoal.name]
                let_go = box_released(row, subgoal.target) or (
                    "it reads " + ", ".join(said.printed_elsewhere(task))
                    if said.printed_elsewhere(task) else "")
                rebox = [subgoal.target] if let_go else []
                if let_go:
                    self.log.record("plan", "mission",
                                    "the box for {!r} is not kept across this replan: {}"
                                    .format(subgoal.target, let_go),
                                    {"rebox": subgoal.target, "because": let_go})

            # Both roads lead here: mend the plan from `keep` onwards, under the same bounds.
            if keep >= len(plan.subgoals):
                # A plan that has run out has no unrun step to read a clause off.
                outstanding = [i for i in range(len(plan.clauses)) if i not in self._met]
            else:
                outstanding = [i for i in outstanding_clauses(plan, keep) if i not in self._met]
            price = self._price_of_a_replan(result, plan, outstanding, trigger, keep)
            if self._left() < max(REPLAN_NEEDS_S, price["would_need_s"]):
                # A replan no subgoal can follow is a plan call spent on nothing, and one that
                # cannot fund every outstanding clause robs a clause nobody has tried.
                result.replan_refused_for_price = price
                self.log.record("plan", "mission",
                                "a replan was outstanding and refused for price: it needed "
                                "about {:.0f} s ({}) and {:.0f} s were left".format(
                                    price["would_need_s"],
                                    "; ".join("clause {}: {} needs {:.0f} s".format(
                                        entry["clause"], entry["state"], entry["needs_s"])
                                        for entry in price["clauses"]) or "no clause named",
                                    price["left_s"]),
                                dict(price))
                self._emit({"event": "replan_refused_for_price", **price})
                return self._finish(result, "timeout", started,
                                    error="{:.0f} s of the {:.0f} s mission were left when "
                                          "{} -- the clause(s) still outstanding need about "
                                          "{:.0f} s and a plan call {:.0f} s"
                                          .format(max(0.0, self._left()), self.budget_s,
                                                  trigger.split(":")[0].strip(),
                                                  price["clauses_need_s"],
                                                  price["plan_call_s"]))
            if replans >= self.max_replans:
                return self._finish(result, "exhausted", started, error=exhausted)
            replanned = self.planner.replan(task, plan, verdicts, self._last_now or picture,
                                            trigger, tail=tail,
                                            strict_target=strict, outstanding=outstanding,
                                            remaining=tuple(plan.subgoals[keep:]),
                                            tried_ways=tried_ways,
                                            holding=self._holding_now(picture),
                                            rebox=rebox, since=self._hold_at)
            version += 1
            result.plans.append({"version": version, "trigger": trigger,
                                 "outstanding": list(outstanding),
                                 "met_clauses": sorted(self._met),
                                 # ...and what the clock said when this one was ALLOWED, so
                                 # a sweep can read the decisions that went the other way
                                 "reserve": dict(price),
                                 **replanned.as_dict()})
            if not replanned.ok or replanned.plan is None:
                return self._finish(result, "error", started,
                                    error=replanned.error or "the replan returned nothing")
            # ...and what it may not write at all: a step for a clause already SEEN met.
            fresh = [sg for sg in replanned.plan.subgoals if sg.clause not in self._met]
            dropped = [sg.name for sg in replanned.plan.subgoals if sg.clause in self._met]
            if dropped:
                self.log.record("plan", "mission",
                                "the replan wrote {} step(s) for clause(s) already seen met; "
                                "they were dropped: {}".format(len(dropped),
                                                               ", ".join(dropped)),
                                {"dropped": dropped, "met_clauses": sorted(self._met)})
                self._emit({"event": "dropped", "version": version, "subgoals": dropped,
                            "met_clauses": sorted(self._met)})
                result.plans[-1]["dropped"] = list(dropped)
            if not any(acts(sg.name, sg.subgoal) for sg in fresh):
                # The planner answered that there is nothing left to do: no step, or none that
                # acts on anything a clause still waits for. That is an answer, not a broken
                # shape, and the thing that settles it is the task's own question: the plan is
                # cut back to the steps that ran, and the top of the loop takes the final check
                # on it. Twice over is a planner talking past the check, and the mission stops.
                said_done += 1
                self.log.record("plan", "mission",
                                "the replan says there is nothing left to do: {}".format(
                                    " ".join(str(replanned.plan.rationale or "").split())
                                    or "no reason given"),
                                {"said_done": said_done, "version": version})
                self._emit({"event": "nothing_left_to_do", "version": version,
                            "because": replanned.plan.rationale, "said_done": said_done})
                if said_done > 1:
                    last = next(r["verdict"].get("verdict") or {} for r in
                                reversed(result.subgoals) if r["name"] == TASK)
                    return self._finish(result, "exhausted", started, error=(
                        "the planner twice answered that there is nothing left to do and the "
                        "task-level check last said {} with clause(s) {} held unmet".format(
                            last.get("status") or "nothing valid",
                            [i for i in range(len(plan.clauses)) if i not in self._met])))
                plan = Plan(subgoals=list(plan.subgoals[:keep]), rationale=plan.rationale,
                            clauses=list(plan.clauses), boxed_on=list(plan.boxed_on),
                            boxed_at=float(plan.boxed_at))
                verdicts = {name: v for name, v in verdicts.items()
                            if name in {s.name for s in plan.subgoals}}
                index, attempt, replans = len(plan.subgoals), 1, replans + 1
                continue
            # The planner may say there is nothing left to try: out of reach, broken, the same
            # step failed the same way twice, or something the task asks made permanently false.
            surrender = gave_up_reason(replanned.plan, replans)
            if surrender:
                result.gave_up = True
                result.gave_up_because = surrender
                self.log.record("plan", "mission", "the planner gave up: {}".format(surrender),
                                {"gave_up": True, "replans": replans})
                self._emit({"event": "gave_up", "version": version, "because": surrender,
                            "replans": replans})
                return self._finish(result, "gave_up", started,
                                    error="planner gave up: {}".format(surrender))
            replans += 1
            # Everything before the failed subgoal is done by construction (after the final
            # check, the whole plan); the planner refuses to reuse a finished name.
            kept = list(plan.subgoals[:keep])
            verdicts = {name: v for name, v in verdicts.items()
                        if name in {s.name for s in kept}}
            # The clause list is the FIRST plan's and is never rewritten: one written again at
            # every replan can lose an entry, the failure the clause index exists to stop.
            plan = Plan(subgoals=kept + fresh,
                        rationale=replanned.plan.rationale, clauses=list(plan.clauses),
                        # the boxes on the new steps were drawn on the pictures the REPLAN was
                        # shown, so which pictures those were comes with them -- and the steps
                        # kept from before, or whose box was kept, say when THEIR box was drawn
                        boxed_on=list(replanned.plan.boxed_on),
                        boxed_at=float(replanned.plan.boxed_at),
                        boxes_kept_from={
                            **{step.name: float(plan.boxes_kept_from.get(step.name,
                                                                        plan.boxed_at))
                               for step in kept if step.target_boxes},
                            **dict(replanned.plan.boxes_kept_from)})
            # The replanner returns a TAIL; what runs is the tail merged onto the kept steps.
            result.plans[-1].update(plan=plan.model_dump(),
                                    replan=replanned.plan.model_dump())
            self._record_plan(version, plan, replanned.clamped, trigger)
            self._emit({"event": "replanned", "version": version, "trigger": trigger,
                        "plan": plan.model_dump(), "clamped": list(replanned.clamped),
                        "elapsed_s": round(replanned.elapsed_s, 2)})
            ceiling = self._ceiling(result.total_cycles, fresh)
            index, attempt = len(kept), 1

    # ------------------------------------------------------------ is the TASK finished

    def _fold(self, plan: Plan, verdict, source: str, retract: bool = False) -> None:
        """Fold one answer's clause rows into the mission's per-clause state."""
        for index, row in clause_marks(plan.clauses, verdict).items():
            met = bool(getattr(row, "met", False))
            if not met and not retract:
                continue
            said = {"clause": index, "met": met, "source": source,
                    "what": str(getattr(row, "what", "") or "")[:120],
                    "evidence": str(getattr(row, "evidence", "") or "").strip()[:240]}
            event, standing = self._settled.get(index), self._clause_said.get(index) or {}
            if not met and standing.get("met") is False \
                    and str(standing.get("source")).endswith(_MEASURED):
                continue                # a measured miss stands; a reader's denial adds nothing
            if met and index in self._retracted and not event:
                self._refused_marks.append(said)
                self.log.record("verdict", source,
                                "clause {} is not marked met again on this answer alone: an "
                                "answer took it back and nothing the executor measures has "
                                "happened to that clause's object since".format(index),
                                dict(said))
                self._emit({"event": "refused_mark", **said})
                continue
            if not met:
                # Denied: no later answer may mark this clause met on its own, met before or not,
                # without something measured in between.
                self._retracted.add(index)
            if not met and index in self._met:
                self._retractions.append(said)
                self.log.record("verdict", source,
                                "clause {} had been seen met and this answer to the task's "
                                "own question says it is not: it is outstanding again "
                                "({})".format(index, said["evidence"] or "no evidence given"),
                                dict(said))
                self._emit({"event": "retracted", **said})
            self._met = (self._met | {index}) if met else (self._met - {index})
            self._clause_said[index] = said

    def _unmet_clauses(self, plan: Plan, verdict) -> List[str]:
        """The clauses of the PLAN that nothing has yet seen met."""
        # Read through the same evidence rule as :meth:`_fold`, the only reason this union is
        # here: an answer's rows may not let a clause in by the door the fold refused it at.
        met = self._met | {index for index in met_clauses(plan.clauses, verdict)
                           if index not in self._retracted or index in self._settled}
        return [clause for index, clause in enumerate(plan.clauses) if index not in met]

    def _price_of_a_replan(self, result: MissionResult, plan: Plan,
                           outstanding: Sequence[int], trigger: str,
                           keep: int = 0) -> Dict[str, Any]:
        """What another plan and the work it would have to buy would cost, against what is
        left of the clock."""
        clause_of = {step.name: step.clause for step in plan.subgoals}
        rows = [(row.get("name"), clause_of.get(row.get("name"), -1),
                 float(row.get("elapsed_s") or 0.0)) for row in result.subgoals]
        remaining = [(step.name, step.clause)
                     for step in plan.subgoals[max(0, int(keep)):]]
        per_clause = what_the_outstanding_clauses_need(remaining, rows, outstanding,
                                                       *self._subgoal_s)
        spent = sum(cost for _, clause, cost in rows if clause in set(outstanding))
        clauses_need = sum(entry["needs_s"] for entry in per_clause.values())
        plan_call = float((result.plans[-1] if result.plans else {}).get("elapsed_s") or 0.0)
        return {"left_s": round(max(0.0, self._left()), 1),
                "rail_s": REPLAN_NEEDS_S,
                "plan_call_s": round(plan_call, 1),
                "steps_already_cost_s": round(spent, 1),
                # ...and what each clause still outstanding needs, which is the gate
                "clauses": [per_clause[index] for index in sorted(per_clause)],
                "clauses_need_s": round(clauses_need, 1),
                "would_need_s": round(plan_call + clauses_need, 1),
                "outstanding": [int(i) for i in outstanding],
                "because": trigger.split(":")[0].strip()}

    def _partial(self, result: MissionResult, missing: Sequence[str]) -> str:
        """File a ``task_done`` that does not cover the plan as partial, and say what is
        left."""
        if not missing:
            return ""
        why = "the task's other clause(s) are not met: " + "; ".join(missing[:3])
        row = result.subgoals[-1] if result.subgoals else None
        if isinstance((row or {}).get("verdict"), dict):
            row["verdict"]["partial"] = why
        self._emit({"event": "partial", "because": why})
        return why

    # ------------------------------------------------------------------ the last question

    def _final_check(self, task: str, plan: Plan, before, result: MissionResult,
                     stopped: str = ""):
        """Ask the verifier about the WHOLE task, once, when a plan runs out all-done."""
        started = time.monotonic()
        question = task_question(task)
        claim = {"done": True, "cycles": result.total_cycles,
                 "assessment": "every one of the {} planned subgoal(s) -- {} -- was verified "
                               "done, one at a time".format(
                                   len(plan.subgoals),
                                   ", ".join(s.name for s in plan.subgoals) or "none")}
        if stopped:
            # The other caller: the robot stopped mid-plan and will not move again.
            claim = {"done": False, "cycles": result.total_cycles,
                     "assessment": "the robot can make NO FURTHER MOTION: {}. Judge the "
                                   "scene exactly as it is now -- nothing can change it "
                                   "from here, and what the plan had left to do is not an "
                                   "argument either way.".format(stopped.strip()),
                     "ending": "the run ended part way through a step"}
        now = self.observe()
        self._last_now = now              # what the replan below is shown, if there is one
        final = self.verifier.verify(task, question, before, now, claim, None)
        self._record_verdict("task", final, now, 1)
        again = self._ask_once_more(task, plan, before, claim, final)
        if again is not None:
            now, final = self._last_now, again
        result.subgoals.append(self._task_row(question, claim, final,
                                              1 if again is None else 2, started))
        return final

    def _task_row(self, question: Subgoal, claim: Dict[str, Any], final, attempt: int,
                  started: float) -> Dict[str, Any]:
        """The subgoal row the whole-task question is filed under."""
        row: Dict[str, Any] = {"name": TASK, "attempt": attempt,
                               "subgoal": question.subgoal,
                               "criterion": question.criterion, "max_cycles": 0,
                               "gripper_only": False, "cycles": [], "executor_claim": claim,
                               "verdict": final.as_dict(), "alerts": [],
                               "stopped_by_monitor": False, "error": "",
                               "elapsed_s": round(time.monotonic() - started, 1)}
        self._add_ground_truth(row)
        return row

    def _add_ground_truth(self, row: Dict[str, Any]) -> None:
        """The simulator's own answer, filed on the row; a record, never a dependency."""
        if self.ground_truth is not None:
            try:
                row["ground_truth"] = self.ground_truth()
            except Exception as exc:
                row["ground_truth"] = {"error": str(exc)[:120]}

    def _ask_once_more(self, task: str, plan: Plan, before, claim, final):
        """The one re-ask a mission may buy: every clause has an event behind it and the check
        still says the task is not complete, so it is asked again on a fresh picture."""
        if self._reasked or final.verdict is None or final.verdict.status == "task_done":
            return None
        if not all(index in self._settled for index in range(len(plan.clauses))):
            return None
        self._reasked = True
        self._last_now = self.observe()
        again = self.verifier.ask_task(task, before, self._last_now, claim)
        self._record_verdict("task", again, self._last_now, 2)
        self._emit({"event": "asked_again", "verdict": again.as_dict()})
        return again if again.verdict is not None else None

    # ------------------------------------------------------------------ one attempt

    def _attempt(self, task: str, subgoal: Subgoal, attempt: int, plan: Plan,
                 verdicts: Dict[str, Verdict]) -> Dict[str, Any]:
        started = time.monotonic()
        self._emit({"event": "subgoal_start", "name": subgoal.name, "attempt": attempt,
                    "max_cycles": subgoal.max_cycles, "subgoal": subgoal.subgoal,
                    "criterion": subgoal.criterion, "target": subgoal.target})
        row: Dict[str, Any] = {"name": subgoal.name, "attempt": attempt,
                               "subgoal": subgoal.subgoal, "criterion": subgoal.criterion,
                               "target": subgoal.target,
                               # where the planner marked that object, and in which pictures
                               # -- so a sweep can score the marking against the truth tape
                               "target_boxes": dict(subgoal.target_boxes),
                               "boxed_on": list(plan.boxed_on),
                               "boxed_at": float(plan.boxes_kept_from.get(subgoal.name,
                                                                           plan.boxed_at)),
                               "boxes_kept": subgoal.name in plan.boxes_kept_from,
                               "max_cycles": subgoal.max_cycles, "gripper_only":
                               subgoal.gripper_only, "cycles": [], "executor_claim": {},
                               "verdict": {}, "alerts": [], "error": "", "elapsed_s": 0.0}

        loop = self.loop_factory(subgoal)
        first: List[Any] = []                   # the pre-subgoal picture, for verify
        self._latest = None                     # never judge the last subgoal's frame under
        inner = loop.observe                    # this subgoal's question
        # A carry, a lift or a place (until it opens the jaws) STARTS holding something.
        watch: Dict[str, Any] = {"wants": wants_holding(subgoal.criterion),
                                 "release": wants_release(subgoal.criterion)}

        def tee():
            observation = inner()
            self._latest = observation
            if not first:
                first.append(observation)
            self._watch_grasp(observation, watch, loop, subgoal.name)
            return observation

        loop.observe = tee
        self._current = {"name": subgoal.name, "loop": loop, "alerts": [], "stop": None,
                         # for _not_a_stop: is this step ABOUT letting the thing go, and have
                         # the jaws already been opened on purpose in it?
                         "release": wants_release(subgoal.criterion), "opened": False}
        start_row = len(self.log.rows)

        # What the executor has done in this attempt, rebuilt from its own events.
        shadow: List[Dict[str, Any]] = []

        def on_executor_event(event: Dict[str, Any]) -> None:
            kind = event.get("event")
            if kind == "episode_over":
                # The loop says so itself before it raises.
                row["episode_over"] = True
            if kind == "held":
                watch["took"] = True
            if kind == "proposed":
                shadow.append({"index": int(event.get("cycle") or len(shadow)),
                               "proposal": {k: v for k, v in event.items()
                                            if k not in ("event", "cycle")},
                               "steps": [], "supervision": [], "stopped_by": "", "finding": "",
                               "reach": {}, "wrist_check": {}, "error": ""})
            elif kind == "reached" and shadow:
                shadow[-1]["reach"] = {k: v for k, v in event.items()
                                       if k not in ("event", "cycle")}
            elif kind == "step_end" and shadow:
                shadow[-1]["steps"].append({k: v for k, v in event.items() if k != "event"})
            if kind == "step_end" and (event.get("kind") or "") == "gripper" \
                    and "open" in str(event.get("label") or "").lower():
                # The jaws have been opened on purpose.
                (self._current or {})["opened"] = True
                # How wide they went, which is what says whether the payload is out of them.
                gap = (event.get("detail") or {}).get("opening_mm")
                if gap is not None:
                    watch["opened_to_mm"] = float(gap)
                self._watch_release(watch, loop, subgoal.name)
            if kind == "proposed":
                # What the executor said this motion should change.
                said = event.get("proposal") or {}
                (self._current or {})["expect"] = str(said.get("expect") or "").strip()
            self.log.from_executor_event(subgoal.name, event)
            if kind == "step_end" and self.monitor is not None:
                self.monitor.notify_step_end()
            self._emit({**event, "subgoal": subgoal.name})

        cycles: List[Any] = []
        if self.monitor is not None:
            self.monitor.start(task, describe_plan(plan, verdicts), subgoal,
                               lambda: self._recent(subgoal.name, start_row))
        try:
            # The planner marked this object in the plan-time pictures; the executor is given
            # the marks so it does not re-choose the object out of the words on every cycle.
            handover = accepted_kwargs(loop.run, target_boxes=dict(subgoal.target_boxes))
            row["boxes_handed_over"] = sorted(handover)
            # The mission's wall clock, armed over this subgoal: when it passes the executor is
            # stopped the way the page's STOP stops it, between two actions and never inside one.
            clock = None if self._deadline is None else \
                threading.Timer(max(0.0, self._left()), loop.cancel)
            if clock is not None:
                clock.daemon = True
                clock.start()
            try:
                cycles = loop.run(subgoal.subgoal, on_event=on_executor_event,
                                  history_preface=self._tried_text(subgoal.name, attempt),
                                  criterion=subgoal.criterion, target=subgoal.target,
                                  carry=self._carry, **handover)
            finally:
                if clock is not None:
                    clock.cancel()
            row["out_of_time"] = self._left() <= 0.0
        except Exception as exc:                # a fault in the executor ends the mission
            row["error"] = "the executor loop raised: {}".format(exc)
            row["traceback"] = traceback.format_exc()[-2000:]
            # ...unless it is the episode itself ending, which is not a fault: nothing can move
            # again, and the scene as it stands is the result.
            row["episode_over"] = bool(row.get("episode_over")) or cannot_move_again(exc)
        finally:
            # Stopped BEFORE the alerts are read: once stop() returns no alert of this subgoal
            # can land, and it never waits for a look that is still out.
            if self.monitor is not None:
                self.monitor.stop()
            current, self._current = self._current, {}
            try:
                self._carry = loop.carry_over()
            except Exception:               # a loop that raised owes the next one nothing
                self._carry = {}

        row["cycles"] = [c.as_dict() for c in cycles or []]
        if not row["cycles"] and shadow:
            # The loop raised and took its own list with it.
            row["cycles"], row["cycles_recovered"] = shadow, True
        self.log.record_cycles(subgoal.name, row["cycles"])
        # A grasp or a loss changes what is held: the net motion before it means nothing.
        took_hold = bool(watch.get("wants")) and not watch.get("held")
        if took_hold or watch.get("lost"):
            self._since_grasp = no_motion_yet()
        else:
            self._since_grasp = net_motion(row["cycles"], self._since_grasp)
        for cycle in row["cycles"]:
            reach = cycle.get("reach") or {}
            if reach.get("from_plan_boxes") and (reach.get("located") or {}).get("point_mm"):
                self._marks.setdefault(self._object_key(reach.get("target") or ""),
                                       reach["located"]["point_mm"])
        row["executor_claim"] = _claim(row["cycles"], self._since_grasp,
                                       self._marks.get(self._object_key(subgoal.target)))
        # The executor ends a step that commanded motion and moved nothing with a frozen-pose
        # outcome; the row keeps it, since it is the next attempt's most useful fact.
        row["frozen"] = frozen_reason(row)
        # ...and, more narrowly, whether the attempt ENDED with the tool where it was -- what
        # says a second attempt from the same pose is not bought.
        row["never_moved"] = ended_without_motion(row)
        row["alerts"] = [a.model_dump() for a in current.get("alerts", [])]
        alert = current.get("stop") or (current["alerts"][-1] if current.get("alerts") else None)
        row["stopped_by_monitor"] = current.get("stop") is not None
        if row["error"] or self._cancelled.is_set() or self._left() <= 0.0:
            # A fault, a cancel or the mission clock ends the mission: nothing more is asked.
            row["elapsed_s"] = round(time.monotonic() - started, 1)
            return row

        now = self.observe()
        self._last_now = now              # what a replan is shown, if this subgoal triggers one
        if watch.get("took") or watch.get("lost") or watch.get("released"):
            self._hold_at = float(getattr(now, "capture_time", 0.0) or 0.0)
        measured = self._released_for_good(row, watch, now, loop)
        if watch.get("lost"):
            # Nothing to ask.
            verdict_result = VerdictResult(
                verdict=Verdict(status="not_done", replan=True,
                                because="the object was lost during this step: the robot was "
                                        "holding something when it began and reports nothing "
                                        "held now",
                                evidence=watch.get("reason") or "the gripper reports nothing "
                                                                "held"),
                gate="lost_grasp")
            # Settled in code, like the verifier's own gates -- so, like them, it buys the one
            # question nobody on this path has asked: is the task itself finished?
            verdict_result = self.verifier.also_ask_the_task(
                verdict_result, task, subgoal, first[0] if first else None, now,
                row["executor_claim"])
            row["lost_grasp"] = True
        elif CANNOT_FROM_THE_SIDE in str(row["executor_claim"].get("stopped_by") or ""):
            # Measured before anything moved: the same step again ends the same way.
            verdict_result = VerdictResult(verdict=Verdict(
                status="not_done", replan=True, because=row["executor_claim"]["stopped_by"],
                evidence="measured by the robot before it moved"), gate="cannot")
        elif measured:
            # The jaws measured the payload out of them; where, settles whether it is put down.
            row["release_confirmed"] = measured
            miss = measured_release(row).get("miss")
            verdict_result = VerdictResult(
                verdict=Verdict(status="not_done" if miss else "done", replan=bool(miss),
                                because=miss or "the jaws opened wider than the close measured "
                                "the thing between them to be and the robot no longer reports "
                                "holding it, so it is out of them and this step has put it down",
                                evidence="the jaws went to {opened_to_mm} mm around a "
                                         "payload the close measured at {payload_width_mm} "
                                         "mm, which is more than the {jaw_resolution_mm} mm "
                                         "these jaws can resolve".format(**measured)),
                gate="release_missed" if miss else "released")
            verdict_result = self.verifier.also_ask_the_task(
                verdict_result, task, subgoal, first[0] if first else None, now,
                row["executor_claim"])
        else:
            verdict_result = self.verifier.verify(task, subgoal, first[0] if first else None,
                                                  now, row["executor_claim"], alert)
        # Written whatever the verdict turned out to be: a sweep counting how often the
        # release cut a step short cannot read it off a verdict that a gate settled.
        row["released"] = bool(watch.get("released"))
        placed = watch.get("released") or (
            watch.get("release") and verdict_result.verdict is not None
            and verdict_result.verdict.status == "done")
        if placed and not verdict_result.gate:
            # The step ended on the release itself, or the verifier says the step that lets the
            # thing go is met -- the same moment by the other road, which the watch cannot always
            # see (the grasp reading may have been unknown).
            verdict_result = self.verifier.also_ask_the_task(
                verdict_result, task, subgoal, first[0] if first else None, now,
                row["executor_claim"])
        row["verdict"] = verdict_result.as_dict()
        self._record_verdict(subgoal.name, verdict_result, now, attempt)
        row["note_version"] = self._note_version        # the note this step's verdict read
        self._request_note(task, plan, verdicts)
        self._remember_attempt(subgoal.name, attempt, verdict_result, row)
        self._add_ground_truth(row)
        row["elapsed_s"] = round(time.monotonic() - started, 1)
        return row

    def _regrasp_index(self, plan: Plan, index: int, subgoal: Subgoal) -> Optional[int]:
        """Which step to go back to after a lost grasp, or ``None`` to replan as before."""
        key = self._object_key(subgoal.target)
        if not key or self._regrasped.get(key):
            return None
        back = index
        while back - 1 >= 0:
            earlier = plan.subgoals[back - 1]
            if not wants_holding(earlier.criterion):
                break
            if earlier.target and subgoal.target and \
                    not same_object(earlier.target, subgoal.target):
                break
            back -= 1
        if back == index:
            return None                   # the step that lost it IS the grasp: replan
        self._regrasped[key] = self._regrasped.get(key, 0) + 1
        return back

    def _lost_after(self, take: Subgoal, lost_in: Subgoal, row: Dict[str, Any]) -> None:
        """Tell the grasp, on its next attempt, that what it took did not stay taken."""
        self._tried[take.name] = {
            "attempt": self._tried.get(take.name, {}).get("attempt", 1),
            "status": "not_done",
            "because": "what was taken here did not stay held: the robot was holding it when "
                       "the later step \"{}\" began and reported nothing held part way "
                       "through it, so this grasp took the object somewhere it could slip "
                       "out of".format(lost_in.name),
            "evidence": ((row.get("verdict") or {}).get("verdict") or {}).get("evidence", ""),
            "ending": "it ended with the object in the jaws, and a later step found them "
                      "empty -- so take it somewhere it cannot slide out of, or take it by "
                      "a narrower part"}

    def _next_attempt(self, result: MissionResult, name: str) -> int:
        """One more than the highest attempt already filed under this name."""
        seen = [int(r.get("attempt") or 1) for r in result.subgoals if r.get("name") == name]
        return (max(seen) + 1) if seen else 1

    def _holding_now(self, fallback=None) -> bool:
        """Does the robot say it has something in its jaws at this moment?"""
        now = self._last_now or fallback
        return (getattr(now, "grasp", None) or {}).get("holding") is True

    def _object_key(self, target: str) -> str:
        """What this object has been counted under before, or the phrase as it stands now."""
        words = strip_part(target).strip().lower()
        if not words:
            return ""
        # Both counts, because one phrase keys both: an object reworded by a replan would be a
        # new object to the lost-grasp count and could spend its one re-entry twice.
        for known in list(self._replans_for) + list(self._regrasped):
            if same_object(known, words):
                return known
        return words

    # ------------------------------------------------------------------ the pieces

    def _watch_grasp(self, observation, watch: Dict[str, Any], loop, name: str) -> None:
        """Stop a carry the moment the robot stops holding what it set out with."""
        holding = (getattr(observation, "grasp", None) or {}).get("holding")
        if holding is not None:
            # The reading is None whenever the jaws were not asked to close, are still moving or
            # sit near their open endpoint (src/controller/clearance.py:: grasp_state), so a step
            # can begin with one; the first DEFINITE reading says whether it began holding.
            watch["holding"] = holding
            if not watch.get("seen"):
                watch["seen"], watch["held"] = True, holding is True
        if not (watch.get("wants") or watch.get("release")) or watch.get("lost") \
                or watch.get("released"):
            return
        if not watch.get("held") or holding is not False:
            return
        watch["lost"] = True
        watch["reason"] = (getattr(observation, "grasp", None) or {}).get("reason") \
            or "the gripper reports nothing held"
        text = ("the object was LOST: the robot was holding something when this step began "
                "and now reports nothing held ({})".format(watch["reason"]))
        # Filed as an alert: it is the same kind of fact the monitor's stops are, it ends the
        # step the same way, and ``Evidence.kind`` is a closed set (src/planner/evidence.py).
        self.log.record("alert", name, text,
                        {"level": "stop", "finding": "dropped", "lost": True})
        self._emit({"event": "lost_grasp", "subgoal": name, "because": text})
        if loop is not None:
            loop.cancel()

    def _watch_release(self, watch: Dict[str, Any], loop, name: str) -> None:
        """End a place or a release at the step that puts the thing down."""
        if not watch.get("release") or watch.get("released"):
            return
        if not (watch.get("held") or watch.get("holding") is True):
            return                        # the jaws opened with nothing in them: an ordinary
        watch["released"] = True          # step, and a grasp usually starts with one
        text = ("the object was LET GO: this step is about putting it down, the robot was "
                "holding it, and the jaws have now been opened on purpose -- the step is over "
                "and the task's own question is asked on the spot")
        self.log.record("verdict", name, text, {"released": True})
        self._emit({"event": "released", "subgoal": name, "because": text})
        if loop is not None:
            loop.cancel()

    def _released_for_good(self, row: Dict[str, Any], watch: Dict[str, Any],
                           now, loop) -> Dict[str, Any]:
        """What the jaws themselves say about the open this step made, or ``{}``."""
        if not watch.get("released"):
            return {}
        gap, width = watch.get("opened_to_mm"), measured_release(row).get("width_mm")
        if gap is None or width is None:
            return {}
        if (getattr(now, "grasp", None) or {}).get("holding") is True:
            return {}                   # the robot says it still has it: not a release
        try:
            resolution = float(jaw_resolution_mm(loop))
        except Exception:               # a loop that cannot be asked settles nothing here
            return {}
        if float(gap) <= float(width) + resolution:
            return {}
        return {"opened_to_mm": round(float(gap), 1),
                "payload_width_mm": round(float(width), 1),
                "jaw_resolution_mm": round(resolution, 1),
                "holding": (getattr(now, "grasp", None) or {}).get("holding")}

    def _note_settled(self, plan: Plan, subgoal: Subgoal, row: Dict[str, Any]) -> None:
        """Write down the event this attempt settled in code, against the subgoal's clause."""
        index = int(subgoal.clause)
        if not (0 <= index < len(plan.clauses)):
            return                      # a retreat serves no clause and settles none
        source = "{} {}".format(subgoal.name, _MEASURED)
        miss = measured_release(row).get("miss") if row.get("release_confirmed") else ""
        if miss:
            # Let go of away from where it goes: denied until something measured happens to it.
            self._settled.pop(index, None)
            self._fold(plan, Verdict(status="not_done", clauses=[{
                "what": plan.clauses[index], "met": False, "evidence": miss}]), source, True)
            return
        event = what_the_code_settled(row, plan.clauses[index])
        if not event:
            return
        self._settled[index] = {"clause": index, **event}
        # An event re-opens a clause an answer took back: something happened to its object.
        self._retracted.discard(index)
        self.log.record("verdict", subgoal.name,
                        "clause {} has an event behind it now: {}".format(index,
                                                                         event["said"]),
                        dict(self._settled[index]))
        self._emit({"event": "settled", **self._settled[index]})

    def _remember_attempt(self, name: str, attempt: int, verdict_result, row) -> None:
        """Keep what this attempt was judged to be, for the next attempt at the same subgoal."""
        verdict = verdict_result.verdict
        if verdict is None or verdict.status in ("done", "task_done"):
            self._tried.pop(name, None)
            return
        ending = _how_it_ended(row["cycles"])
        if row.get("frozen"):
            ending = ("{} -- and the tool ended where it began: {}. Going the same way again "
                      "will not move it.".format(ending, str(row["frozen"])[:160]))
        self._tried[name] = {"attempt": attempt, "status": verdict.status,
                             "because": verdict.because.strip() or "no reason was given",
                             "evidence": verdict.evidence.strip(),
                             "ending": ending}

    def _tried_text(self, name: str, attempt: int) -> str:
        """What is appended to the subgoal sentence on a second attempt at the same step."""
        last = self._tried.get(name)
        if attempt <= 1 or not last:
            return ""
        self.log.record("verdict", name,
                        "attempt {}: the previous attempt was judged {} ({}); {}".format(
                            attempt, last["status"], last["because"], last["ending"]),
                        {"attempt": attempt, "because": last["because"][:200],
                         "evidence": last["evidence"][:200], "ending": last["ending"]})
        return _TRIED.format(attempt=attempt, because=last["because"],
                             evidence=last["evidence"] or "nothing further was named",
                             ending=last["ending"])

    def _record_verdict(self, name: str, verdict_result, now, attempt: int) -> None:
        """One log row and one event per verdict -- a subgoal's, and the task-level one."""
        verdict = verdict_result.verdict
        self.log.record("verdict", name,
                        "the verifier said {}: {}".format(
                            verdict.status if verdict else "nothing valid",
                            (verdict.because if verdict else verdict_result.error)
                            or "no reason given"),
                        {"status": verdict.status if verdict else "",
                         "replan": bool(verdict.replan) if verdict else False,
                         "evidence": (verdict.evidence if verdict else "")[:200],
                         "attempt": attempt,
                         # No executor event carries the clearance, so it is written here from
                         # the mission's own picture: the number a failed descent is argued about.
                         "clearance_mm": getattr(now, "clearance_mm", None)})
        self._emit({"event": "verdict", "name": name,
                    "status": verdict.status if verdict else "",
                    "because": verdict.because if verdict else verdict_result.error,
                    "evidence": verdict.evidence if verdict else "",
                    "replan": bool(verdict.replan) if verdict else False,
                    "elapsed_s": round(verdict_result.elapsed_s, 2)})


    def _recent(self, name: str, start_row: int) -> str:
        """The last three things that happened in this attempt, for the monitor's prompt."""
        rows = [r for r in self.log.rows[start_row:]
                if r.subgoal == name and r.kind in ("cycle", "supervision")]
        text = self.log.render(rows[-3:])
        # What the executor SAID the motion now running should change.
        expect = (self._current or {}).get("expect") or ""
        if expect:
            text = (text + "\n" if text else "") + \
                "[proposal] what the executor expects this motion to change: " + expect
        return text

    def _record_plan(self, version: int, plan: Plan, clamped: List[str], why: str) -> None:
        self.log.record("plan", "", "plan v{} ({} subgoals): {}".format(
            version, len(plan.subgoals), why),
            {"version": version, "subgoals": [s.name for s in plan.subgoals],
             "clamped": list(clamped), "rationale": plan.rationale[:200]})

    def _left(self) -> float:
        """Seconds of the mission's wall clock still unspent."""
        return 1e9 if self._deadline is None else self._deadline - time.monotonic()

    def _ceiling(self, spent: int, subgoals: List[Subgoal]) -> int:
        """One retry's worth of every remaining subgoal, measured from here (see the docstring)."""
        return int(spent) + 2 * sum(max(1, int(s.max_cycles)) for s in subgoals)

    def _finish(self, result: MissionResult, status: str, started: float,
                error: str = "") -> MissionResult:
        result.status = status
        result.error = result.error or error
        if status == "timeout" and not result.ended_on:
            # The two ways a mission can run out are different findings and both are counted:
            # the seconds left could not fund the work outstanding, or there were none left.
            result.ended_on = "replan_refused_for_price" \
                if result.replan_refused_for_price else "clock_ran_out"
        result.clauses = {"met": sorted(self._met),
                          "said": [self._clause_said[i] for i in sorted(self._clause_said)],
                          "retractions": list(self._retractions),
                          "settled": [self._settled[i] for i in sorted(self._settled)],
                          "refused_marks": list(self._refused_marks)}
        result.elapsed_s = time.monotonic() - started
        result.left_s = 0.0 if self._deadline is None else self._deadline - time.monotonic()
        if self.monitor is not None:
            self.monitor.stop()
        self._current = {}
        # Off the mission's clock: the note in flight is let finish, so notes.json is whole.
        thread = self._note_thread
        if thread is not None:
            thread.join(timeout=NOTE_JOIN_S)
        with self._note_lock:
            self._note_asked = None
            result.notes = list(self._notes)
        self._emit({"event": "finished", "status": status, "error": result.error,
                    "total_cycles": result.total_cycles,
                    "elapsed_s": round(result.elapsed_s, 1),
                    "subgoals": len(result.subgoals)})
        return result

    # ------------------------------------------------------------------ the memory note

    def _request_note(self, task: str, plan: Plan, verdicts: Dict[str, Verdict]) -> None:
        """Ask for a note covering every row since the last one; never wait for it. One call
        is in flight at a time: a request made meanwhile is written when it returns."""
        if self.summarizer is None:
            return
        asked = (task, describe_plan(plan, verdicts))
        with self._note_lock:
            self._note_asked = asked
            if self._note_thread is not None:
                return
            self._note_thread = threading.Thread(target=self._write_notes, name="memory-note",
                                                 daemon=True)
            self._note_thread.start()

    def _write_notes(self) -> None:
        # Here, not at the top: src.memory reads src.planner.evidence, whose package is this one.
        from src.memory.summarize import fallback_context, render_note
        while True:
            with self._note_lock:
                asked, self._note_asked = self._note_asked, None
                if asked is None or self._cancelled.is_set() or self._left() <= 0.0:
                    self._note_thread = None
                    return
                end, previous = len(self.log.rows), self._note
                rows = self.log.rows[self._note_since:end]
            note_result = self.summarizer.note(asked[0], asked[1], previous, rows)
            with self._note_lock:
                if note_result.ok:
                    self._note, self._note_since = note_result.note, end
                    self._note_text = render_note(note_result.note)
                    self._note_version += 1
                else:
                    self._note_text = fallback_context(previous, rows)
                self._notes.append(dict(note_result.as_dict(), version=self._note_version))
                version = self._note_version
            if note_result.ok:
                self.log.record("note", "", "note {}: {}".format(
                    version, note_result.note.summary.strip() or "nothing was summarised"),
                    {"version": version})
            self._emit({"event": "note", "version": version, "error": note_result.error,
                        "note": None if note_result.note is None
                        else note_result.note.model_dump(),
                        "elapsed_s": round(note_result.elapsed_s, 2)})

    def _emit(self, event: Dict[str, Any]) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(event)
        except Exception:
            # The caller's drawing is not worth the mission.
            pass


def _claim(cycles: List[Dict[str, Any]], since_grasp: Optional[Dict[str, Any]] = None,
           mark: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """What the executor says it did -- shown to the verifier so it can disagree with it."""
    done = any((c.get("proposal") or {}).get("done") for c in cycles)
    last = cycles[-1] if cycles else {}
    proposal = last.get("proposal") or {}
    said = proposal.get("proposal") or {}
    stopped = str(last.get("stopped_by") or "")
    return {"done": bool(done), "assessment": (said or {}).get("assessment", ""),
            "cycles": len(cycles),        # ...and what ended it, less the frozen ending's verdict
            "stopped_by": stopped.partition(":")[0] if stopped.startswith(_FROZEN_ENDING_SAYS)
            else stopped,
            # A count cannot tell work from a "done" on cycle 0: five subgoals ended that way.
            "ending": _how_it_ended(cycles),
            # WHICH object the cameras picked, and how far off it the step ended.
            **_reached(cycles, mark),
            **_since_grasp_numbers(since_grasp),
            "error": last.get("error") or proposal.get("error", "")}


def _reached(cycles: List[Dict[str, Any]],
             mark: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """What the last reach of this step MEASURED: the label the cameras gave the thing they
    aimed at, the gap the LAST cycle left to it, and the located point's distance to ``mark``."""
    latest = next((c["batch"]["residual_mm"] for c in reversed(list(cycles or []))
                   if isinstance((c.get("batch") or {}).get("residual_mm"), (int, float))), None)
    for cycle in reversed(list(cycles or [])):
        reach = cycle.get("reach") or {}
        located = reach.get("located") or {}
        label = str(located.get("label") or "").strip()
        residual = reach.get("residual_mm") if latest is None else latest
        if label and located.get("doubtful"):
            # A look that says its own position may be anywhere in a strip measures no gap
            # (gpt6sol-full-fix4 task 100: "151 mm from that handle", with the handle held).
            return {}
        if label and isinstance(residual, (int, float)):
            point = located.get("point_mm")
            off = {} if not (point and mark) else {"from_mark_mm": round(
                ((point[0] - mark[0]) ** 2 + (point[1] - mark[1]) ** 2) ** 0.5, 1)}
            return {"reached": {"label": label, "residual_mm": round(float(residual), 1),
                                "target": str(reach.get("target") or ""), **off}}
    return {}


#: Which way round the vertical each rotation word turns, from schema/directions.yaml.
_YAW_SIGN = {"yaw_left": 1.0, "anticlockwise": 1.0, "counterclockwise": 1.0,
             "counter_clockwise": 1.0, "ccw": 1.0,
             "yaw_right": -1.0, "clockwise": -1.0, "cw": -1.0}
#: ...and which way each direction word moves it, for the same reason.
_MOVE_AXIS = {"forward": (1.0, 0.0, 0.0), "backward": (-1.0, 0.0, 0.0), "back": (-1.0, 0.0, 0.0),
              "left": (0.0, 1.0, 0.0), "right": (0.0, -1.0, 0.0),
              "up": (0.0, 0.0, 1.0), "down": (0.0, 0.0, -1.0)}


def no_motion_yet() -> Dict[str, Any]:
    """A fresh accumulator: nothing turned and nothing travelled."""
    return {"turn_net_deg": 0.0, "turn_total_deg": 0.0, "turn_stop": 0.0,
            "move": [0.0, 0.0, 0.0], "move_total_mm": 0.0}


def net_motion(cycles: List[Dict[str, Any]], into: Optional[Dict[str, Any]] = None):
    """Add up what the robot MEASURED over these cycles: net turn and net displacement."""
    out = dict(into or no_motion_yet())
    out["move"] = list(out.get("move") or [0.0, 0.0, 0.0])
    for cycle in cycles or []:
        for step in cycle.get("steps") or []:
            words = [w.strip("()[]").lower() for w in str(step.get("label") or "").split()]
            kind = str(step.get("kind") or "")
            if kind == "rotate":
                sign = next((_YAW_SIGN[w] for w in words if w in _YAW_SIGN), None)
                turned = abs(float(step.get("turned_deg") or 0.0))
                if sign is None:
                    continue
                # the way the last turn was stopped short, pushed back on; 0 when it was not
                out["turn_stop"] = sign if step.get("outcome") == "limited" else 0.0
                if not turned:
                    continue
                out["turn_net_deg"] += sign * turned
                out["turn_total_deg"] += turned
            elif kind == "translate":
                axis = next((_MOVE_AXIS[w] for w in words if w in _MOVE_AXIS), None)
                moved = abs(float(step.get("along_axis_mm") or 0.0))
                if axis is None or not moved:
                    continue
                out["move"] = [a + b * moved for a, b in zip(out["move"], axis)]
                out["move_total_mm"] += moved
    return out


def _since_grasp_numbers(since_grasp: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The accumulator as the verifier reads it: two numbers for a turn, two for a travel."""
    if not since_grasp:
        return {}
    move = list(since_grasp.get("move") or [0.0, 0.0, 0.0])
    return {"turn_net_deg": round(float(since_grasp.get("turn_net_deg") or 0.0), 1),
            "turn_total_deg": round(float(since_grasp.get("turn_total_deg") or 0.0), 1),
            "turn_stop": float(since_grasp.get("turn_stop") or 0.0),
            "move_net_mm": round(sum(v * v for v in move) ** 0.5, 1),
            "move_total_mm": round(float(since_grasp.get("move_total_mm") or 0.0), 1)}


#: What an exception out of the executor says when the robot cannot be made to move again -- the
#: episode is over, and every motion after it is refused before it reaches the robot.
_OVER_PHRASES = ("stopped accepting motion", "episode has ended", "episode is over",
                 "episode step horizon", "no further motion", "will not accept any further",
                 "nothing commanded from here can move")


def cannot_move_again(exc: BaseException) -> bool:
    """Is this the end of the episode rather than a fault in the run?"""
    if getattr(exc, "episode_over", False) or getattr(exc, "no_more_motion", False):
        return True
    if type(exc).__name__ in ("EpisodeOver", "EpisodeFinished"):
        return True
    text = " ".join(str(exc).lower().split())
    return any(phrase in text for phrase in _OVER_PHRASES)


#: Handed to the executor ahead of its history on a retry of the SAME subgoal. See ``_tried_text``.
_TRIED = """

WHAT HAS ALREADY BEEN TRIED FOR THIS SAME STEP
This is attempt {attempt}. The last attempt at this same step was judged NOT DONE: {because}
What the judge read: {evidence}
How that attempt ended: {ending}
Proposing what was proposed last time will be judged the same way. Do something different:
look from somewhere else first, aim at a different part of the object, or change the order --
and if you believe the step is already met, say what in the picture shows it.
"""


def _not_a_stop(alert: Alert, current: Dict[str, Any], holding: bool) -> Alert:
    """Downgrade a STOP the robot's readings answer: `dropped` after an open on purpose, and
    `wrong_object` about what it holds with no printing read on it (the verifier's rule)."""
    if alert.level != "stop":
        return alert
    if alert.finding == "dropped" and (current.get("release") or current.get("opened")):
        why = ("this step opens the jaws on purpose, so the object leaving them is the step "
               "working rather than an accident")
    elif alert.finding == "wrong_object" and holding and not _PRINTED.search(
            "{} {}".format(alert.because, alert.identity)):
        why = ("the robot reports holding what this step's grasp took, and a look-alike seen "
               "elsewhere is not printing read on the thing in the jaws")
    else:
        return alert
    return Alert(level="attention", finding=alert.finding, identity=alert.identity,
                 because=(alert.because.strip() + " -- recorded, but NOT a stop: " + why))


def _how_it_ended(cycles: List[Dict[str, Any]]) -> str:
    """One phrase for the shape of a finished attempt -- above all, the empty one."""
    cycles = list(cycles or [])
    if not cycles:
        return "it ran no cycles at all"
    # Steps, and nothing else: a reach that resolved to nothing writes its `text` and moves
    # the robot not at all.
    moved = [c for c in cycles if c.get("steps")]
    if not moved and (cycles[-1].get("proposal") or {}).get("done"):
        return ("it answered on its first cycle that the step was already met, proposed no "
                "action at all and never moved the robot")
    if not moved:
        return "it ran {} cycle(s) and the robot never moved".format(len(cycles))
    return "it ran {} cycle(s), {} of which moved the robot".format(len(cycles), len(moved))


#: What the executor calls a step that commanded motion and moved nothing.
_FROZEN = "frozen"
#: ...and the sentence a stop reason uses for the same thing.
_NO_MOTION = ("no motion", "did not move", "never moved", "moved 0", "0 mm of travel",
              "has not moved", "have not moved")


def _frozen_bits(value) -> str:
    """Anything in this row, at any depth, that says the tool did not move."""
    if isinstance(value, str):
        text = value.lower()
        if _FROZEN in text or any(word in text for word in _NO_MOTION):
            return value.strip()
        return ""
    if isinstance(value, dict):
        for key in ("outcome", "reason", "stopped_by", "message", "ending", "frozen"):
            found = _frozen_bits(value.get(key))
            if found:
                return found
        for key in ("executor_claim", "cycles", "steps"):
            found = _frozen_bits(value.get(key))
            if found:
                return found
        return ""
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _frozen_bits(item)
            if found:
                return found
    return ""


def frozen_reason(row: Dict[str, Any]) -> str:
    """Did this attempt end with the tool where it started, and what was said about it?"""
    return _frozen_bits(row) if isinstance(row, dict) else ""


#: Under how much travel, summed over every motion step of an attempt, the attempt moved the
#: tool nowhere: above the 2 mm a move is cut back to at the reach wall. (The executor's 40 mm
#: still-CYCLE figure is per cycle and would call a 30 mm approach no motion.)
NEVER_MOVED_MM = 3.0


def ended_without_motion(row: Dict[str, Any]) -> str:
    """Did this attempt END with the tool where it was, in the executor's own words -- or ""?

    Narrower than :func:`frozen_reason`, which reads any mention of no motion at any depth
    and so is raised by one blocked step among moving ones. Three endings count: the executor's own
    frozen ending on the last cycle, an attempt with no motion step at all, and one whose
    every motion step moved and turned nothing."""
    cycles = list(row.get("cycles") or [])
    steps = [step for cycle in cycles for step in (cycle.get("steps") or [])]
    if not steps:
        return _how_it_ended(cycles)
    last = str((cycles[-1] or {}).get("stopped_by") or "")
    if _FROZEN_ENDING_SAYS in last:
        return last
    motion = [step for step in steps if step.get("kind") in ("translate", "rotate")]
    if not motion or any(step.get("kind") == "gripper" for step in steps):
        return ""                     # the jaws changing is a change a criterion can be about
    # ``moved_mm`` is the controller's whole displacement; a row written before it carried
    # one has only the travel along the commanded axis, which is what is read then.
    moved = sum(abs(float(step.get("along_axis_mm") or 0.0)) if step.get("moved_mm") is None
                else abs(float(step["moved_mm"])) for step in motion)
    turned = sum(abs(float(step.get("turned_deg") or 0.0)) for step in motion)
    if moved < NEVER_MOVED_MM and turned < ExecutorLoop.NO_TURN_DEG:
        return ("every motion it commanded came to nothing: {} step(s) moved the tool "
                "{:.0f} mm in all".format(len(motion), moved))
    return ""


#: The executor's own ending for a step that stopped changing anything (``FROZEN_ENDING``),
#: up to the cycle count it fills in.
_FROZEN_ENDING_SAYS = FROZEN_ENDING.split("{}")[0]


def box_released(row: Dict[str, Any], target: str) -> str:
    """Why a replan may re-draw this object's box rather than keep the earlier plan's -- the
    executor's last look for these words found nothing, or found something else by name --
    or "" to keep it. (A box drawn before the jaws last took hold, lost hold or let go is
    re-drawn anyway: see ``_hold_at``.)"""
    last = None
    for cycle in row.get("cycles") or []:
        reach = cycle.get("reach") or {}
        if reach.get("located") is not None \
                and same_object(str(reach.get("target") or ""), str(target or "")):
            last = reach
    if last is None:
        return ""
    located = last.get("located") or {}
    if located.get("error"):
        return "the last look for it could not find it: {}".format(
            str(located.get("error"))[:160])
    if located.get("point_mm") is None:
        return "the last look for it came back with no position"
    # ...or found something else by name at any step, not only the take the gate judges: a
    # box kept through that replan would send the next approach to the other thing again.
    if names_something_else(str(located.get("label") or ""), str(last.get("target") or ""),
                            str(target or "")):
        return "the last look for it picked out {!r} by name".format(located.get("label"))
    return ""


def ways_already_tried(result: MissionResult, target: str) -> List[str]:
    """Every side and mode the failed attempts on this object have already used."""
    if not (target or "").strip():
        return []
    found = set()
    for row in result.subgoals:
        if not same_object(row.get("target") or "", target):
            continue
        verdict = (row.get("verdict") or {}).get("verdict") or {}
        status = verdict.get("status") or "not_done"
        if status in ("done", "task_done") and not row.get("frozen"):
            continue
        found |= ways(row.get("name") or "", row.get("subgoal") or "")
    return sorted(found)


#: How much of a clause's words a verdict's clause must share to be read as the same one.
_SAME_CLAUSE_SHARE = 0.5


def clause_marks(clauses: Sequence[str], verdict) -> Dict[int, Any]:
    """Every clause of the plan this verdict wrote a row about, as ``{index: row}``."""
    rows = list(getattr(verdict, "clauses", None) or [])
    found: Dict[int, Any] = {}
    for position, row in enumerate(rows):
        said = set(object_words(getattr(row, "what", "") or ""))
        best, score = -1, 0.0
        for index, clause in enumerate(clauses):
            words = set(object_words(str(clause)))
            if not words or not said:
                continue
            share = len(words & said) / float(len(words))
            if share > score:
                best, score = index, share
        if score >= _SAME_CLAUSE_SHARE:
            found[best] = row
        elif len(rows) == len(clauses) and position < len(clauses):
            found.setdefault(position, row)
    return found


#: How far a contact push has to have carried the tool along the face it was aimed at before it
#: counts as a push that MOVED the thing rather than one that merely touched it.
PUSHED_ALONG_MM = 5.0

#: How far the adapter has to say the wrist actually turned before a rotation counts as one.
TURNED_DEG = 1.0


def what_the_code_settled(row: Dict[str, Any], clause: str = "") -> Dict[str, Any]:
    """The event in one attempt that the CODE settled -- a turn only for a turning clause."""
    for cycle in row.get("cycles") or []:
        steps = list(cycle.get("steps") or [])
        aimed = [str(delta.get("label") or "")
                 for delta in (cycle.get("proposal") or {}).get("deltas") or []
                 if delta.get("push") and delta.get("kind") == "translate"
                 and abs(float((delta.get("axis") or [0.0, 0.0, 1.0])[2])) < 0.9]
        for label in [one for one in aimed if one]:
            along = sum(abs(float(step.get("along_axis_mm") or 0.0)) for step in steps
                        if str(step.get("label") or "").startswith(label))
            if along >= PUSHED_ALONG_MM:
                return {"kind": "pushed", "subgoal": row.get("name") or "",
                        "said": "a push along the face carried the tool "
                                "{:.0f} mm ({})".format(along, label)}
        for step in steps:
            if step.get("kind") != "rotate" or not wants_turn(clause):
                continue
            turned = abs(float(step.get("turned_deg") or 0.0))
            if turned >= TURNED_DEG and str(step.get("outcome") or "") != "aborted":
                return {"kind": "turned", "subgoal": row.get("name") or "",
                        "said": "the adapter reported the wrist turned "
                                "{:.0f} deg ({})".format(turned, step.get("label") or "")}
    return {}


#: How far off the middle a release INTO something may be, as a share of its allowance -- a
#: container's walls forgive a miss. ON something the allowance itself is the bar.
SETTLES_WITHIN = 1.5


def measured_release(row: Dict[str, Any]) -> Dict[str, Any]:
    """What the last release this attempt wrote down measured, or ``{}``: the payload's width,
    and ``miss``, why it was let go of too far off the middle of what it goes on (measured
    finer than allowed)."""
    for cycle in reversed(list(row.get("cycles") or [])):
        release = cycle.get("release") or {}
        if release.get("payload_width_mm") is None:
            continue
        allowed, offset = release.get("allowed_mm"), release.get("offset_mm")
        allowed = None if allowed is None else float(allowed) * (
            SETTLES_WITHIN if release.get("placing_in") else 1.0)
        return {"width_mm": float(release["payload_width_mm"]),
                "miss": "" if allowed is None or offset is None or float(offset) <= allowed
                else "it was let go of {:.0f} mm from the middle of {}, and at most {:.0f} mm "
                     "is over it".format(float(offset), release.get("receiver") or
                                         "what it goes on", allowed)}
    return {}


def met_clauses(clauses: Sequence[str], verdict) -> List[int]:
    """Which of the task's clauses a verdict marked MET, as indexes into ``clauses``."""
    return sorted(index for index, row in clause_marks(clauses, verdict).items()
                  if getattr(row, "met", False))


def _earlier_failures(result: MissionResult, target: str, skip: str, keep: int = 3) -> str:
    """The steps about this same object that have already been judged not_done, for the
    replan."""
    words = (target or "").strip().lower()
    if not words:
        return ""
    seen = []
    for row in result.subgoals:
        if row.get("name") == skip or (row.get("target") or "").strip().lower() != words:
            continue
        verdict = (row.get("verdict") or {}).get("verdict") or {}
        if (verdict.get("status") or "not_done") == "not_done":
            seen.append("{} (attempt {}): {}".format(
                row.get("name"), row.get("attempt"),
                verdict.get("because") or "no reason given"))
    if not seen:
        return ""
    return "\nEarlier steps about this same object were judged not_done too -- {}.".format(
        "; ".join(seen[-keep:]))
