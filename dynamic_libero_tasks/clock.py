"""Keep a dynamic scene moving while the harness thinks.

Without this the sim only advances when the arm is commanded: during every model call the
belt stops. The clock steps the sim with the arm held (an OSC delta of zero, the adapter's
own "stop") so that sim time keeps pace with the wall clock. A command's own steps count
toward that pace, so a motion that already runs in real time gets no idle ticks.

Every step goes through the adapter's env thread (its single-worker pool), so idle ticks
interleave with commands and renders and never race them.
"""
from __future__ import annotations

import threading
import time

import numpy as np

#: How far the sim may fall behind the wall clock before the backlog is dropped (the sim
#: cannot render fast enough), and the most idle ticks one wake may drive to catch up.
BACKLOG_S = 1.0
BURST = 4


class Clock:
    def __init__(self, adapter):
        self.adapter = adapter
        self.idle_ticks, self.dropped_s = 0, 0.0
        self._stop = threading.Event()
        threading.Thread(target=self._run, name="libero-clock", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        a = self.adapter
        env, dt = a._env.env, a.control_dt
        wall0, tick0 = time.monotonic(), env.timestep
        while not self._stop.wait(dt / 2):
            if env.timestep < tick0:                     # a reset put the counter back
                wall0, tick0 = time.monotonic(), env.timestep
            behind = (time.monotonic() - wall0) / dt - (env.timestep - tick0)
            if behind > BACKLOG_S / dt:
                self.dropped_s += (behind - 1) * dt
                wall0 += (behind - 1) * dt
                behind = 1
            if behind < 1 or a.episode_over():
                continue
            n = min(int(behind), BURST)
            try:
                a._on_env_thread(a._drive_now, np.zeros(6), n)
                self.idle_ticks += n
            except Exception:
                return                                    # the env went away under us

    def report(self) -> dict:
        return {"idle": self.idle_ticks, "clock_dropped_s": round(self.dropped_s, 2)}


def start(adapter):
    """A Clock for an adapter whose scene moves on its own, else None."""
    return Clock(adapter) if hasattr(adapter._env.env, "motion") else None
