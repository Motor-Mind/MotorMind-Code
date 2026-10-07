"""One after-the-fact judgement of a batch that was stopped or took the tool further away:
continue or stop, with a finding -- and nothing else."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from .context import describe_cameras
from .models import Supervision


@dataclass
class SupervisionStep:
    """One check. Carries a decision and its reasons -- never an action."""

    decision: str
    because: str = ""
    evidence: str = ""
    finding: str = "on_course"
    capture_time: float = 0.0
    elapsed_s: float = 0.0
    error: str = ""
    prompt: str = ""
    images: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {"decision": self.decision, "because": self.because, "evidence": self.evidence,
                "finding": self.finding, "capture_time": self.capture_time, "elapsed_s": round(self.elapsed_s, 2),
                "error": self.error, "prompt": self.prompt, "images": self.images}


class MotionSupervisor:
    def __init__(self, client, prompts: Dict[str, Any]):
        self.client = client
        self.prompts = prompts
        self._judged_capture = 0.0

    def begin(self, from_capture: float = 0.0) -> None:
        """Called once per batch: a picture from before it is not a picture of it."""
        self._judged_capture = float(from_capture or 0.0)

    def fresh_enough(self, capture_time: float) -> bool:
        """Is this a picture that has not been judged yet?"""
        return capture_time > self._judged_capture

    def check(self, subgoal: str, motion: str, expectation: str, progress: str,
              robot_text: str, observation, before=None,
              legend: str = "") -> Optional[SupervisionStep]:
        """Judge the motion against a picture taken now. ``None`` if there is nothing new."""
        capture = float(getattr(observation, "capture_time", 0.0) or 0.0)
        if not self.fresh_enough(capture):
            return None

        images = list(observation.images)
        names = ["{} (now)".format(n) for n in observation.names]
        if before is not None and before.images:
            images = list(before.images) + images
            names = ["{} (before the motion)".format(n) for n in before.names] + names

        prompt = self.prompts["supervise"]["text"].format(
            subgoal=subgoal.strip(), motion=motion, expectation=expectation or "not stated",
            progress=progress, robot=robot_text, cameras=describe_cameras(names),
            legend=legend or "No legend is available for these cameras.")

        started = time.monotonic()
        reply = self.client.ask_json(prompt, images=images,
                                     system=self.prompts["supervise"]["system"])
        self._judged_capture = capture
        elapsed = time.monotonic() - started

        if not reply.ok:
            step = SupervisionStep(decision="continue", capture_time=capture, elapsed_s=elapsed,
                                   prompt=prompt, images=list(names),
                                   error=reply.error or "no JSON in the reply",
                                   because="the model did not answer; a short bounded motion "
                                           "carries on and will be checked again")
            return step
        try:
            answer = Supervision(**reply.data)
        except ValidationError as exc:
            # A reply carrying an action is rejected whole.
            first = exc.errors()[0]
            where = ".".join(str(p) for p in first.get("loc", ())) or "the reply"
            step = SupervisionStep(decision="stop", capture_time=capture, elapsed_s=elapsed,
                                   prompt=prompt, images=list(names),
                                   error="{}: {}".format(where, first.get("msg", "invalid")),
                                   because="the supervision reply was not a valid decision, so "
                                           "the motion is stopped rather than trusted")
            return step

        step = SupervisionStep(decision=answer.decision, because=answer.because,
                               evidence=answer.evidence, finding=answer.finding,
                               capture_time=capture, elapsed_s=elapsed, prompt=prompt,
                               images=list(names))
        return step
