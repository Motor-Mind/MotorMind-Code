"""Turn the evidence log into the short note the planner replans from."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from src.planner.evidence import Evidence, EvidenceLog

from .models import MemoryNote

#: The shapes a sentence takes when it asserts that a GOAL CLAUSE has been achieved -- a thing
#: put where the task asked for it, a step of the task finished.
_ASSERTS = ("is in the", "are in the", "is inside", "are inside", "is now in", "now inside",
            "already in the", "was placed", "is placed", "has been placed", "were placed",
            "successfully placed", "placed in", "placed into", "placed on", "put in the",
            "put into", "deposited", "delivered into", "dropped into", "sits in the",
            "sitting in the", "rests in", "rests on", "resting in", "resting on",
            "is on the", "are on the", "is complete", "is completed", "is finished",
            "is done", "has been done", "was achieved", "is satisfied", "has been shut",
            "is shut", "is closed", "has been closed", "is open now")

#: ...and what a ROW has to say before such a sentence may be written: a verdict the code
#: filed, with the status it settled on. Nothing else in the log is an outcome.
_SETTLED = ("done", "task_done")

#: Words that carry nothing about WHICH thing a sentence is about.
_SMALL = frozenset((
    "the", "a", "an", "of", "with", "on", "in", "its", "it", "and", "to", "that", "this",
    "at", "by", "for", "is", "are", "was", "were", "from", "near", "beside", "next", "has",
    "have", "been", "now", "already", "into", "inside", "still", "not", "no", "but", "as",
    "after", "before", "which", "what", "when", "where", "there", "here", "then", "than",
    "cycle", "cycles", "robot", "gripper", "jaws", "tool", "arm", "step", "subgoal", "task",
    "confirmed", "measured", "reported", "shows", "showed", "seen", "monitor", "verifier",
    "placed", "put", "done", "complete", "completed", "finished", "successfully", "over"))


def _words(text: str) -> set:
    """The content words of a sentence -- what it is ABOUT, with the scaffolding taken off."""
    return {w for w in re.findall(r"[a-z]+", str(text or "").lower())
            if len(w) > 2 and w not in _SMALL}


def asserts_an_outcome(sentence: str) -> bool:
    """Does this sentence claim a thing is where the task asked for it, or a part of it done?"""
    return _subject(sentence) is not None


def _subject(sentence: str) -> Optional[str]:
    """What such a sentence is ABOUT: everything in front of the phrase that asserts it."""
    text = " ".join(str(sentence or "").lower().split())
    if not text:
        return None
    found = [text.index(phrase) for phrase in _ASSERTS if phrase in text]
    return text[:min(found)] if found else None


def _settled_rows(rows: List[Evidence]) -> List[Evidence]:
    """The rows in THIS window that actually settled something: a verdict, with its status."""
    return [r for r in rows or []
            if r.kind == "verdict" and str((r.data or {}).get("status") or "") in _SETTLED]


def supported_by(sentence: str, rows: List[Evidence]) -> bool:
    """Is there a row in this window that settled an outcome for the thing this names?"""
    about = _words(_subject(sentence) or sentence) or _words(sentence)
    if not about:
        return True                    # nothing named: nothing to check it against
    for row in _settled_rows(rows):
        said = _words(row.text) | _words(str(row.subgoal or "").replace("_", " "))
        if about & said:
            return True
    return False


def hearsay(sentence: str) -> bool:
    """Does this cite the monitor's one look -- a question for the verifier, not a fact?"""
    return bool(set(re.findall(r"[a-z]+", str(sentence or "").lower())) & {"monitor", "alert"})


def keep_facts(facts: List[str], rows: List[Evidence]):
    """The facts that may stand, and the ones dropped for asserting an outcome nothing
    showed or for resting on a monitor alert alone."""
    kept, dropped = [], []
    for fact in facts or []:
        if (hearsay(fact) or asserts_an_outcome(fact)) and not supported_by(fact, rows):
            dropped.append(str(fact))
        else:
            kept.append(fact)
    return kept, dropped

NOTHING_YET = ("Nothing has been learned yet: this is the first time the planner has been "
               "asked, so there is no evidence to draw on.")


@dataclass
class NoteAttempt:
    """One round trip. Kept whether it worked or not -- the refusals are the interesting part."""

    reply: Dict[str, Any]
    error: str = ""
    prompt: str = ""


@dataclass
class NoteResult:
    note: Optional[MemoryNote] = None
    error: str = ""
    attempts: List[NoteAttempt] = field(default_factory=list)
    elapsed_s: float = 0.0
    # The facts struck out for claiming an outcome no row in this window showed.
    dropped_facts: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True only for a note this call actually produced -- a carried-over one is not ok."""
        return self.note is not None and not self.error

    def as_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "error": self.error, "elapsed_s": round(self.elapsed_s, 2),
                "dropped_facts": list(self.dropped_facts),
                "note": None if self.note is None else self.note.model_dump(),
                "attempts": [{"error": a.error, "prompt": a.prompt, **a.reply}
                             for a in self.attempts]}


class Summarizer:
    def __init__(self, client, prompts: Dict[str, Any], max_attempts: int = 3):
        self.client = client            # injected: this module never builds one
        self.prompts = prompts
        self.max_attempts = max_attempts

    def note(self, task: str, plan_text: str, previous: Optional[MemoryNote],
             rows: List[Evidence]) -> NoteResult:
        started = time.monotonic()
        result = NoteResult()
        base = self.prompts["summarize"]["text"].format(
            task=(task or "").strip() or "not stated",
            plan=(plan_text or "").strip() or "no plan has been written yet",
            previous=render_note(previous),
            rows=EvidenceLog(rows).render() or "nothing has happened since the last note")
        system = self.prompts["summarize"]["system"]

        prompt = base
        for _ in range(self.max_attempts):
            reply = self.client.ask_json(prompt, images=(), system=system)
            attempt = NoteAttempt(reply=reply.as_dict(), prompt=prompt)
            result.attempts.append(attempt)
            if not reply.ok:
                attempt.error = reply.error or "no JSON in the reply"
            else:
                try:
                    note = MemoryNote(**reply.data)
                except ValidationError as exc:
                    attempt.error = _first_problem(exc)
                else:
                    # A note is not allowed to assert that the task, or a part of it, has been
                    # achieved unless a row in the window it was written from settled it.
                    kept, dropped = keep_facts(note.facts, rows)
                    result.dropped_facts = dropped
                    result.note = note.model_copy(update={"facts": kept}) if dropped else note
                    break
            prompt = base + _retry_note(attempt.error)

        if result.note is None:
            # The last note is better than no note, and saying so is better than a blank.
            result.note = previous
            result.error = (result.attempts[-1].error if result.attempts
                            else "the model was never reached")
        result.elapsed_s = time.monotonic() - started
        return result


def fallback_context(previous: Optional[MemoryNote], rows: List[Evidence],
                     limit_rows: int = 30) -> str:
    """What the planner is given when the summariser is down: the last note and the raw tail."""
    tail = list(rows or [])[-int(limit_rows):]
    return ("{}\n\nRAW EVIDENCE (the last {} rows, not summarised -- the summariser could not "
            "be reached)\n{}".format(render_note(previous), len(tail),
                                     EvidenceLog(tail).render() or "nothing recorded yet"))


def render_note(note: Optional[MemoryNote]) -> str:
    """The block a planner prompt embeds. Never empty: silence reads as "all is well"."""
    if note is None:
        return NOTHING_YET
    lines = ["WHERE THINGS STAND: " + (note.summary.strip() or "not stated")]
    for title, items in (("WHAT HAS GONE WRONG", note.failures),
                         ("WHAT IS KNOWN ABOUT THE SCENE", note.facts),
                         ("ADVICE FOR THE NEXT PLAN", note.advice)):
        if items:
            lines.append(title)
            lines.extend("  - " + str(item).strip() for item in items)
    return "\n".join(lines)


def _retry_note(error: str) -> str:
    return ("\n\nYOUR PREVIOUS ANSWER WAS REJECTED\n{}\n"
            "Answer again, with only the four fields asked for above.".format(error))


def _first_problem(exc: ValidationError) -> str:
    first = exc.errors()[0]
    where = ".".join(str(p) for p in first.get("loc", ())) or "the reply"
    return "{}: {}".format(where, first.get("msg", "invalid"))
