"""A second pair of eyes on a running subgoal -- and no hands at all."""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from pydantic import ValidationError

from src.executor.context import describe_cameras
from src.executor.proposal import _problems

from .models import Alert, Subgoal, marked_at


class SceneMonitor:
    def __init__(self, client, prompts: Dict[str, Any], latest: Callable[[], Optional[Any]],
                 on_alert: Callable[[Alert, Dict[str, Any]], None], period_s: float = 4.0):
        self.client = client
        self.prompts = prompts
        self.latest = latest
        self.on_alert = on_alert
        self.period_s = period_s
        self.history: List[Alert] = []
        self._wake = threading.Event()
        self._stopping = threading.Event()      # this watch's own: set once, never cleared
        self._raising = threading.Lock()        # an alert is handed over whole, or not at all
        self._judged: Optional[float] = None    # None, not 0.0: capture_time defaults to 0.0

    # ------------------------------------------------------------------ lifecycle

    def start(self, task: str, plan_text: str, subgoal: Subgoal,
              recent: Callable[[], str]) -> None:
        self.stop()                                   # a new subgoal is a new question
        self._stopping, self._wake = threading.Event(), threading.Event()
        self._judged = None
        threading.Thread(target=self._watch, name="scene-monitor", daemon=True,
                         args=(task, plan_text, subgoal, recent, self._stopping,
                               self._wake)).start()

    def notify_step_end(self) -> None:
        """A motion has just ended: look now rather than at the end of the period."""
        self._wake.set()

    def stop(self) -> None:
        """Never waits: a look still out when its subgoal ends is dropped when it comes back,
        so the mission is not held behind the monitor's model call."""
        with self._raising:
            self._stopping.set()
        self._wake.set()

    # ------------------------------------------------------------------ the thread

    def _watch(self, task: str, plan_text: str, subgoal: Subgoal, recent, stopping,
               wake) -> None:
        while not stopping.is_set():
            # Cleared BEFORE the look, so a notify that lands while the call is out survives
            # to the wait below and is not swallowed by the look it arrived during.
            wake.clear()
            try:
                self._look(task, plan_text, subgoal, recent, stopping)
            except Exception as exc:                  # a fault must not end the watch
                # The same meta keys as a real look, so the caller can index them blind.
                self._raise(Alert(level="attention", finding="unsure",
                                  because="the monitor could not judge the scene: {}".format(exc)),
                            {"capture_time": self._judged, "elapsed_s": 0.0,
                             "prompt": "", "images": []}, stopping)
            wake.wait(self.period_s)

    def _look(self, task: str, plan_text: str, subgoal: Subgoal, recent, stopping) -> None:
        observation = self.latest()
        if observation is None:
            return
        capture = float(getattr(observation, "capture_time", 0.0) or 0.0)
        if self._judged is not None and capture <= self._judged:
            return
        names = ["{} (now)".format(n) for n in getattr(observation, "names", []) or []]
        prompt = self.prompts["monitor"]["text"].format(
            task=task.strip(),
            plan=plan_text.strip() or "the plan was not supplied",
            subgoal=subgoal.subgoal.strip(),
            criterion=subgoal.criterion.strip() or "no criterion was written",
            recent=(recent() or "").strip() or "nothing has happened yet in this subgoal",
            robot=(getattr(observation, "robot_text", "") or "").strip()
            or "the robot reported nothing about itself",
            cameras=describe_cameras(names))
        # ...and where the planner marked this step's object when it wrote the plan.
        hint = marked_at(subgoal.target_boxes, subgoal.target)
        if hint:
            prompt = prompt + "\n\n" + hint

        started = time.monotonic()
        # Marked judged before the call, not after: a client that RAISES would otherwise leave
        # this frame unjudged and the thread would ask about it again every period, for the
        # whole subgoal, filling history with the same failure.
        self._judged = capture
        reply = self.client.ask_json(prompt, images=list(getattr(observation, "images", []) or []),
                                     system=self.prompts["monitor"]["system"])
        meta = {"capture_time": capture, "elapsed_s": round(time.monotonic() - started, 2),
                # What this call cost the window it lives in.
                "prompt_tokens": reply.prompt_tokens,
                "completion_tokens": reply.completion_tokens,
                "latency_s": round(reply.latency_s, 2),
                "prompt": prompt, "images": names}

        if not reply.ok:
            meta["reply"] = _raw(reply)
            self._raise(Alert(level="attention", finding="unsure",
                              because="the monitor did not answer: {}".format(
                                  reply.error or "no JSON in the reply")), meta, stopping)
            return
        try:
            alert = Alert(**reply.data)
        except ValidationError as exc:
            # The complaint alone says which field was wrong and never what was IN it, and
            # five of the eight attention alerts of the first mission sweep were this branch.
            meta["reply"] = _raw(reply)
            self._raise(Alert(level="attention", finding="unsure",
                              because="the reply was not a valid alert and was refused whole "
                                      "({}), so it cannot stop anything".format(
                                          _problems(exc))), meta, stopping)
            return
        self._raise(alert, meta, stopping)

    def _raise(self, alert: Alert, meta: Dict[str, Any], stopping) -> None:
        with self._raising:
            if stopping.is_set():                     # about a subgoal that has ended
                return
            self.history.append(alert)
            try:
                self.on_alert(alert, meta)
            except Exception:
                # The caller's bookkeeping is not worth the watch.
                pass


def _raw(reply, limit: int = 300) -> str:
    """What the model actually said, kept short: a refused reply is only diagnosable from it."""
    text = reply.text or (json.dumps(reply.data, default=str) if reply.data is not None else "")
    return text[:limit]
