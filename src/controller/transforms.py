"""Rotation maths the controller needs, with no robot's conventions in it."""

from __future__ import annotations

import math
from typing import Iterable, List

import numpy as np

__all__ = ["axis_angle_to_matrix", "matrix_to_axis_angle", "rotation_magnitude",
           "unit", "quat_wxyz_to_matrix", "matrix_to_quat_wxyz"]


def unit(vector: Iterable[float]) -> np.ndarray:
    """Normalise, refusing a zero vector rather than returning a NaN direction."""
    v = np.asarray(list(vector), dtype=float)
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        raise ValueError("a zero vector names no direction")
    return v / norm


def axis_angle_to_matrix(axis: Iterable[float], angle_rad: float) -> np.ndarray:
    """Rodrigues. A zero angle or a zero axis gives identity rather than an error."""
    a = np.asarray(list(axis), dtype=float)
    norm = float(np.linalg.norm(a))
    if norm < 1e-12 or abs(angle_rad) < 1e-15:
        return np.eye(3)
    a = a / norm
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(angle_rad) * K + (1.0 - math.cos(angle_rad)) * (K @ K)


def rotation_magnitude(R: np.ndarray) -> float:
    """How far this rotation turns, in radians."""
    cos_theta = (float(np.trace(np.asarray(R, dtype=float)[:3, :3])) - 1.0) / 2.0
    return math.acos(min(1.0, max(-1.0, cos_theta)))


def matrix_to_axis_angle(R: np.ndarray):
    """``(axis, angle)``."""
    R = np.asarray(R, dtype=float)
    angle = rotation_magnitude(R)
    if angle < 1e-9:
        return np.array([0.0, 0.0, 1.0]), 0.0
    if abs(angle - math.pi) < 1e-6:
        M = (R + np.eye(3)) / 2.0
        diagonal = np.sqrt(np.clip(np.diag(M), 0.0, 1.0))
        k = int(np.argmax(diagonal))
        axis = M[:, k] / diagonal[k] if diagonal[k] > 1e-9 else np.eye(3)[k]
        return axis / float(np.linalg.norm(axis)), angle
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return axis / (2.0 * math.sin(angle)), angle


def quat_wxyz_to_matrix(q: Iterable[float]) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if n < 1e-12:
        raise ValueError("a zero quaternion names no rotation")
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def matrix_to_quat_wxyz(R: np.ndarray) -> List[float]:
    R = np.asarray(R, dtype=float)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return [s/4.0, (R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s]
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
        return [(R[2,1]-R[1,2])/s, s/4.0, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s]
    if i == 1:
        s = math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
        return [(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, s/4.0, (R[1,2]+R[2,1])/s]
    s = math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
    return [(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, s/4.0]
