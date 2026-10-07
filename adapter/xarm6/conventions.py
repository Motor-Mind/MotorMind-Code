"""The xArm's own way of describing a pose."""

from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np

from src.controller.types import Pose


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def matrix_to_rpy(R: np.ndarray):
    """Inverse of :func:`rpy_to_matrix`, in (-pi, pi]."""
    R = np.asarray(R, dtype=float)
    sp = min(1.0, max(-1.0, float(-R[2, 0])))
    pitch = math.asin(sp)
    if abs(sp) > 1.0 - 1e-9:
        # Gimbal lock: at pitch = +-90 the matrix only fixes roll +- yaw, so one is free.
        roll, yaw = 0.0, math.atan2(-R[0, 1], R[1, 1])
    else:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    return _wrap(roll), _wrap(pitch), _wrap(yaw)


def _wrap(angle: float) -> float:
    wrapped = math.atan2(math.sin(angle), math.cos(angle))
    return math.pi if wrapped == -math.pi else wrapped


def flange_pose_from_bridge(tcp_pose: Sequence[float]) -> Pose:
    """``/state.tcp_pose`` -> the flange, in metres and a rotation matrix."""
    x, y, z, roll, pitch, yaw = (float(v) for v in tcp_pose)
    return Pose.of([x / 1000.0, y / 1000.0, z / 1000.0], rpy_to_matrix(roll, pitch, yaw))


def flange_pose_to_bridge(pose: Pose) -> List[float]:
    """The flange -> the six numbers ``/cartesian`` wants."""
    roll, pitch, yaw = matrix_to_rpy(pose.rotation)
    return [pose.position_m[0] * 1000.0, pose.position_m[1] * 1000.0,
            pose.position_m[2] * 1000.0, roll, pitch, yaw]


def flange_to_tool(flange: Pose, offset_m: float) -> Pose:
    """The tool point sits ``offset_m`` along the flange's own +z."""
    return Pose(flange.position_m + flange.rotation @ np.array([0.0, 0.0, offset_m]),
                flange.rotation)


def tool_to_flange(tool: Pose, offset_m: float) -> Pose:
    """Where the flange must be to put the tool there."""
    return Pose(tool.position_m - tool.rotation @ np.array([0.0, 0.0, offset_m]),
                tool.rotation)
