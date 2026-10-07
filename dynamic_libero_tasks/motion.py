"""Scripted object motion for dynamic LIBERO tasks: pure trajectory maths, no MuJoCo.

A task's motion lives beside its BDDL as ``<task>.motion.yaml`` (BDDL has no slot for a
velocity). The BDDL places the objects; the first physics step after a reset *anchors* each
moving object where it actually landed, and from then on its planar pose is a function of the
time since that anchor:

    linear    (conveyor):  s(t) = s0 + v t   along start -> end; past the end it wraps to the
                           start if ``loop``, else it has left the belt (one pass)
    circular  (carousel):  x(t) = c + r (cos(th0 + w t), sin(th0 + w t)),  yaw(t) = yaw0 + w t

Only x, y and orientation are scripted; z is left to physics, so an object rests on the table
however tall it is.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml


@dataclass
class Anchor:
    """Where one object was when the motion started: planar position and its quaternion."""
    xy: np.ndarray
    quat: np.ndarray                      # MuJoCo order, w x y z


@dataclass
class PlanarState:
    xy: np.ndarray
    quat: np.ndarray                      # w x y z
    vel_xy: np.ndarray
    yaw_rate: float
    on_belt: bool = True                  # False once a one-pass belt has carried it off the end


@dataclass
class Motion:
    kind: str                             # "linear" | "circular"
    moving: List[str]
    # linear
    start: np.ndarray = field(default_factory=lambda: np.zeros(2))
    end: np.ndarray = field(default_factory=lambda: np.zeros(2))
    speed: float = 0.0                    # m/s along start -> end
    loop: bool = True
    width: float = 0.14                   # belt width, for the visual only
    # circular
    center: np.ndarray = field(default_factory=lambda: np.zeros(2))
    omega: float = 0.0                    # rad/s, counter-clockwise seen from above
    radius: float = 0.22                  # platform radius, for the visual only

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start))

    @property
    def direction(self) -> np.ndarray:
        return (self.end - self.start) / max(self.length, 1e-9)

    def state(self, anchor: Anchor, t: float) -> PlanarState:
        """The scripted planar pose of an object anchored at ``anchor``, ``t`` s later."""
        if self.kind == "linear":
            u = self.direction
            rel = anchor.xy - self.start
            s0, lateral = float(rel @ u), rel - float(rel @ u) * u
            s = s0 + self.speed * t
            if self.loop:
                s %= self.length
            return PlanarState(self.start + s * u + lateral, anchor.quat.copy(),
                               self.speed * u, 0.0, on_belt=s <= self.length)
        if self.kind == "circular":
            rel = anchor.xy - self.center
            r, th = float(np.hypot(*rel)), float(np.arctan2(rel[1], rel[0])) + self.omega * t
            xy = self.center + r * np.array([np.cos(th), np.sin(th)])
            vel = r * self.omega * np.array([-np.sin(th), np.cos(th)])
            return PlanarState(xy, quat_mul(yaw_quat(self.omega * t), anchor.quat),
                               vel, self.omega)
        raise ValueError("unknown motion kind {!r}".format(self.kind))

    def cleats(self, t: float, spacing: float = 0.08) -> List[np.ndarray]:
        """Linear: where the belt's visible cross-bars are at time ``t`` (they wrap)."""
        n = max(1, int(self.length // spacing))
        u = self.direction
        return [self.start + ((i * spacing + self.speed * t) % (n * spacing)) * u
                for i in range(n)]


def yaw_quat(a: float) -> np.ndarray:
    return np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, w x y z."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw])


def load(path: str | Path) -> Motion:
    d: Dict = yaml.safe_load(Path(path).read_text())
    kind = d.pop("type")
    for key in ("start", "end", "center"):
        if key in d:
            d[key] = np.asarray(d[key], dtype=float)
    return Motion(kind=kind, **d)


def sidecar(bddl_path: str | Path) -> Path:
    """``foo.bddl`` -> ``foo.motion.yaml``."""
    p = Path(bddl_path)
    return p.with_name(p.stem + ".motion.yaml")

