"""Ask the model for the next short motion, and refuse anything that will not run."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from pydantic import ValidationError

from src.controller.convert import ConversionError, to_deltas
from src.controller.types import Capabilities, TcpDelta
from src.schema.actions import (ActionError,
                               parse_command)

from .context import compact_json, describe_cameras
from .models import Proposal


@dataclass
class Attempt:
    """One round trip, kept whether it worked or not -- the retries are the interesting part."""

    reply: Dict[str, Any]
    error: str = ""
    prompt_chars: int = 0
    # What was actually sent.
    prompt: str = ""
    images: List[str] = field(default_factory=list)


@dataclass
class ProposalResult:
    proposal: Optional[Proposal] = None
    deltas: List[TcpDelta] = field(default_factory=list)
    attempts: List[Attempt] = field(default_factory=list)
    error: str = ""
    elapsed_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.proposal is not None and not self.error

    @property
    def done(self) -> bool:
        return bool(self.proposal and self.proposal.done)

    def as_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "done": self.done, "error": self.error,
                "elapsed_s": round(self.elapsed_s, 2),
                "proposal": None if self.proposal is None else self.proposal.model_dump(),
                "deltas": [d.as_dict() for d in self.deltas],
                # the complaint LAST: ``VlmReply.as_dict()`` carries an ``error`` of its own,
                # empty when the transport worked, which must not hide a refusal.
                "attempts": [{**a.reply, "error": a.error or a.reply.get("error", ""),
                              "prompt": a.prompt, "images": a.images}
                             for a in self.attempts]}


class Proposer:
    def __init__(self, client, prompts: Dict[str, Any], schema: Dict[str, Any],
                 directions: Dict[str, Any], max_attempts: int = 3,
                 gate: Optional[Callable[..., Optional[str]]] = None):
        self.client = client
        self.prompts = prompts
        self.schema = schema
        self.directions = directions
        self.max_attempts = max_attempts
        # A rule applied to a proposal that already parses: given (proposal, deltas,
        # observation) it returns why the motion is refused, or None.
        self.gate = gate

    def propose(self, subgoal: str, observation, robot_text: str,
                capabilities: Capabilities, capability_text: str,
                criterion: str = "", history: str = "", legend: str = "",
                forbid: str = "", target: str = "") -> ProposalResult:
        started = time.monotonic()
        result = ProposalResult()
        # A "done" whose final action was refused or would not parse.
        done_only = None
        images, names = _pair(observation)
        base = self.prompts["propose"]["text"].format(
            subgoal=subgoal.strip(),
            # The planner's description of the object, written while it could see the whole
            # table.
            target=target.strip() or "No description was given for this subgoal: either it is "
                                     "about no object in particular, or nobody wrote one. "
                                     "Identify the object from the subgoal's own words.",
            criterion=criterion.strip() or "No separate criterion was given: judge it from the "
                                           "subgoal itself.",
            history=history.strip() or "nothing yet -- this is the first attempt",
            robot=robot_text,
            capabilities=capability_text,
            schema=compact_json(self.schema),
            cameras=describe_cameras(names,
                                     "These are the pictures as they are NOW. What changed "
                                     "since the last look is measured for you under WHAT THE "
                                     "ROBOT REPORTS, not shown as a second set of frames."),
            legend=legend or "No legend is available for these cameras, so do not infer a "
                             "robot direction from a direction in the picture.")
        system = self.prompts["propose"]["system"]

        if forbid:
            base = base + "\n\nRESTRICTION FOR THIS MOTION\n" + forbid
        prompt = base
        for _ in range(self.max_attempts):
            reply = self.client.ask_json(prompt, images=images, system=system)
            attempt = Attempt(reply=reply.as_dict(), prompt_chars=len(prompt),
                              prompt=prompt, images=list(names))
            result.attempts.append(attempt)
            if not reply.ok:
                attempt.error = reply.error or "no JSON in the reply"
                prompt = base + _retry_note(attempt.error)
                continue
            try:
                proposal = Proposal(**reply.data)
            except ValidationError as exc:
                attempt.error = _problems(exc)
                prompt = base + _retry_note(attempt.error)
                continue

            if not _carries_an_action(proposal.command):
                # An empty action list is not a command.
                result.proposal = proposal
                break
            if proposal.done:
                # A reply may claim the subgoal met AND carry the action that meets it -- "open
                # the jaws, and that is the release done".
                done_only = proposal
            try:
                command = parse_command(proposal.command)
                deltas = to_deltas(command, capabilities, self.directions)
            except (ActionError, ConversionError) as exc:
                attempt.error = str(exc)
                prompt = base + _retry_note(str(exc))
                continue
            refused = self.gate(proposal, deltas, observation) if self.gate else None
            if refused:
                attempt.error = refused
                prompt = base + _retry_note(refused)
                continue

            result.proposal, result.deltas = proposal, deltas
            break

        if result.proposal is None and done_only is not None:
            result.proposal, result.deltas = done_only, []
        if result.proposal is None:
            result.error = (result.attempts[-1].error if result.attempts
                            else "the model was never reached")
        result.elapsed_s = time.monotonic() - started
        return result


def _carries_an_action(command: Any) -> bool:
    """Is there anything in this command for the robot to do?"""
    if not command:
        return False
    if isinstance(command, dict) and not command.get("actions"):
        return False
    return True


def _pair(observation):
    """The CURRENT images only, named so the model knows when they were taken."""
    return list(observation.images), ["{} (now)".format(n) for n in observation.names]


def _retry_note(error: str) -> str:
    return ("\n\nYOUR PREVIOUS ANSWER WAS REJECTED\n{}\n"
            "Answer again. Use only the command language given above.".format(error))


def _problems(exc: ValidationError, limit: int = 6) -> str:
    """EVERY distinct complaint, one per line -- not just the first."""
    lines = {}
    for problem in exc.errors():
        where = ".".join(str(part) for part in problem.get("loc", ())) or "the reply"
        lines.setdefault(where, "{}: {}".format(where, problem.get("msg", "invalid")))
    kept = list(lines.values())[:limit]
    if len(lines) > limit:
        kept.append("... and {} more".format(len(lines) - limit))
    return "\n".join(kept) or "the reply did not match the schema"
