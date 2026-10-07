"""How far is the tool above whatever is under it, measured by the wrist camera."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .types import Pose, approach_axis

try:
    import cv2
except ImportError:
    cv2 = None


@dataclass
class Clearance:
    mm: Optional[float]                 # nearest surface below the tool, along the approach axis
    median_mm: Optional[float]
    surface_z_mm: Optional[float]       # that surface in the base frame, for cross-checking
    points: int
    blind: bool                         # no usable measurement: the surface is out of range
    near_min_range: bool = False        # measured, but close to dropping out
    reason: str = ""
    at_range_floor: bool = False        # blind BECAUSE the surface has crossed inside the
                                        # minimum range: a lower bound, never a face

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mm": None if self.mm is None else round(self.mm, 1),
            "median_mm": None if self.median_mm is None else round(self.median_mm, 1),
            "surface_z_mm": None if self.surface_z_mm is None else round(self.surface_z_mm, 1),
            "points": self.points,
            "blind": self.blind,
            "near_min_range": self.near_min_range,
            "at_range_floor": self.at_range_floor,
            "reason": self.reason,
        }


def _blind(reason: str, points: int = 0, **flags: bool) -> Clearance:
    """No usable measurement, and why."""
    return Clearance(None, None, None, points, blind=True, reason=reason, **flags)


#: how far past the table a wrist reading may put what is under the tool before it is not
#: believed: the table's own uncertainty (the rig's +-6 mm) and the depth's noise
TABLE_SLACK_MM = 15.0


def held_to_the_table(clearance: Optional[Dict[str, Any]], tool: Optional[Pose],
                      table_z_m: Optional[float],
                      slack_mm: float = TABLE_SLACK_MM) -> Optional[Dict[str, Any]]:
    """Nothing under the tool is lower than the table. A clearance the wrist reads PAST it is a
    camera that is wrong, not air: 2026-10-06, a wrist camera mounted the other way up read the
    table 190 mm down, the block said 498 mm of clearance at 304 mm up, and 'down 350 mm' drove
    the fingertips into the table. Such a reading is held to the tool's height above the table,
    and says so. Only while the jaws point down; a reading nearer than the table is kept."""
    if not clearance or tool is None or table_z_m is None \
            or float(approach_axis(tool)[2]) > -0.99:
        return clearance
    height = (float(tool.position_m[2]) - float(table_z_m)) * 1000.0
    held = dict(clearance)
    for key in ("mm", "median_mm", "tracked_mm"):
        value = held.get(key)
        if value is not None and float(value) > height + slack_mm:
            held[key] = round(max(height, 0.0), 1)
            held["table_held"] = True
    if held.get("table_held"):
        if clearance.get("mm") is not None:
            held["surface_z_mm"] = round(float(table_z_m) * 1000.0, 1)
        held["reason"] = ("the wrist camera's depth put the surface below the table, so the "
                          "tool's height above the table is used instead -- the camera's "
                          "calibration needs checking. "
                          + str(clearance.get("reason") or "")).strip()
    return held


def from_wrist(camera: Dict[str, Any], flange: Pose, control_offset_m: float,
               radius_m: float = 0.04, min_range_m: Optional[float] = None,
               min_points: int = 50, inner_radius_m: float = 0.0) -> Clearance:
    """``camera`` is one entry of a ``/frames`` payload; ``flange`` the matching pose."""
    if min_range_m is None:
        min_range_m = float(camera.get("depth_min_range_m") or 0.30)
    if cv2 is None:
        return _blind("opencv is not installed")
    for key in ("depth_png16", "intrinsic", "cam2base"):
        if not camera.get(key):
            return _blind("the frame carries no " + key)

    T_base_flange = np.eye(4)
    T_base_flange[:3, :3] = flange.rotation
    T_base_flange[:3, 3] = flange.position_m
    cam2base = np.asarray(camera["cam2base"], dtype=float)
    # The rigid mount.
    T_flange_cam = np.linalg.inv(T_base_flange) @ cam2base

    K = np.asarray(camera["intrinsic"], dtype=float)
    depth = cv2.imdecode(np.frombuffer(base64.b64decode(camera["depth_png16"]), np.uint8),
                         cv2.IMREAD_UNCHANGED)
    if depth is None:
        return _blind("the depth image did not decode")
    depth = depth.astype(np.float32) * float(camera.get("depth_scale", 0.001))

    height, width = depth.shape
    rows, columns = np.mgrid[0:height, 0:width]

    def unproject(mask):
        """Those pixels as points, in the camera frame and in the flange's."""
        z = depth[mask]
        x = (columns[mask] - K[0, 2]) / K[0, 0] * z
        y = (rows[mask] - K[1, 2]) / K[1, 1] * z
        camera_frame = np.stack([x, y, z], 1)
        return camera_frame, (T_flange_cam[:3, :3] @ camera_frame.T).T + T_flange_cam[:3, 3]

    def in_the_cone(in_flange):
        """+z of the flange is the approach axis; the tool point sits at control_offset_m
        along it."""
        off_axis = np.hypot(in_flange[:, 0], in_flange[:, 1])
        return ((off_axis < radius_m) & (off_axis >= inner_radius_m)
                & (in_flange[:, 2] > control_offset_m + 0.005))

    # A depth camera returns nothing closer than its minimum range, so anything nearer than that
    # is not a surface -- and on this robot the thing sitting right under the lens is the
    # gripper's own fingers.
    valid = (depth > 0) & (depth >= min_range_m)
    in_camera, in_flange = unproject(valid)
    near = in_the_cone(in_flange)
    if int(near.sum()) < min_points:
        nearest_return = float(in_camera[:, 2].min()) if in_camera.size else float("inf")
        return _blind(
            points=int(near.sum()), near_min_range=True, at_range_floor=True,
            reason="nothing within {:.0f} mm of the tool axis: the nearest depth return in the whole "
            "frame is {:.3f} m against a {:.2f} m minimum range, so the surface below the tool "
            "is inside it".format(radius_m * 1000, nearest_return, min_range_m))

    # THE MINIMUM RANGE IS A FLOOR, AND A FLOOR IS NOT A SURFACE -- nor is the band just over
    # it, where a real camera's first returns sit (the rig's D455 from 0.306 m against 0.30).
    quantum = float(camera.get("depth_scale", 0.001))
    # ...as a share of that range the camera declares (depth_floor_band_share, a rig profile's
    # cameras: fact -- the rig's D455 returns from 0.306 m against 0.30); exact depth has none.
    band = max(2.0 * quantum, min_range_m * float(camera.get("depth_floor_band_share") or 0.0))
    _, hidden = unproject((depth >= min_range_m - 2.0 * quantum) & (depth < min_range_m))
    culled = int(in_the_cone(hidden).sum()) if hidden.size else 0
    if culled >= max(min_points // 5, 1) \
            or float(in_camera[near][:, 2].min()) <= min_range_m + band:
        return _blind(
            points=int(near.sum()), near_min_range=True, at_range_floor=True,
            reason="what is under the tool is inside the {:.2f} m minimum range, so the camera "
            "cannot return it and the nearest thing it CAN return stands in for it: this is a "
            "lower bound on the surface, not a measurement of it".format(min_range_m))

    along = in_flange[near][:, 2] - control_offset_m
    nearest = float(along.min()) * 1000.0
    median = float(np.median(along)) * 1000.0
    # Where that surface is, along the approach axis: a level only while it points down.
    down = float(approach_axis(flange)[2])
    surface_z = None if down != -1.0 else \
        float(flange.position_m[2]) * 1000.0 + down * control_offset_m * 1000.0 + down * nearest

    # is the surface we are watching close to dropping out of range?
    ray = float(np.linalg.norm(in_camera[near].mean(axis=0)))
    near_limit = ray <= min_range_m * 1.25
    return Clearance(nearest, median, surface_z, int(near.sum()),
                     blind=False, near_min_range=near_limit,
                     reason="surface {:.2f} m from the lens against a {:.2f} m minimum range: this "
                     "reading is about to drop out, take the target height from it now".format(
                         ray, min_range_m) if near_limit else "")


# --------------------------------------------------------------------------- the grasp


@dataclass
class Grasp:
    """Did the jaws close on something, or on air?"""

    holding: Optional[bool]          # None when it cannot be told, which is not the same as no
    opening_mm: Optional[float] = None
    closed_fraction: Optional[float] = None   # 0 at the closed endpoint, 1 wide open
    source: str = "unavailable"
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"holding": self.holding,
                "opening_mm": None if self.opening_mm is None else round(self.opening_mm, 2),
                "closed_fraction": None if self.closed_fraction is None
                else round(self.closed_fraction, 4),
                "source": self.source, "reason": self.reason}


# An empty close settles at 1.0-1.3 mm (measured in LIBERO, 32 steps).
EMPTY_BAND_M = 0.003
EMPTY_BAND_FRACTION = 0.04
# Past this much of the span, a "closed" gripper is not gripping a very wide object -- it is
# open.
WIDE_OPEN_FRACTION = 0.90


def grasp_state(reading, empty_band_m: float = EMPTY_BAND_M,
                empty_band_fraction: float = EMPTY_BAND_FRACTION,
                wide_open_fraction: float = WIDE_OPEN_FRACTION) -> Grasp:
    """Decide whether a closed gripper is holding anything."""
    # What the jaws are doing is measurable even when the verdict is not -- a reading taken mid-
    # travel, or with the jaws open, still says how far apart they are.
    if reading is None:
        return Grasp(holding=None, reason="the backend reports nothing about its jaws")

    opening_mm = None if reading.opening_m is None else reading.opening_m * 1000.0
    fraction = None
    if reading.opening_m is not None and reading.open_span_m:
        fraction = reading.opening_m / reading.open_span_m
    elif (reading.raw is not None and reading.raw_open is not None
            and reading.raw_closed is not None):
        span = float(reading.raw_open) - float(reading.raw_closed)
        if abs(span) > 1e-9:
            fraction = (float(reading.raw) - float(reading.raw_closed)) / span
    here = dict(opening_mm=opening_mm, closed_fraction=fraction)

    if (reading.commanded or "").lower() != "close":
        return Grasp(holding=None, source="n/a",
                     reason="the jaws were not asked to close (last command: {})".format(
                         reading.commanded or "none"), **here)
    empty = reading.opening_m <= empty_band_m if reading.opening_m is not None else None
    if reading.moving and not empty:
        return Grasp(holding=None, source="n/a", **here,
                     reason="the jaws are still moving; a reading now would call empty air an "
                            "object")
    if fraction is not None and fraction > wide_open_fraction:
        return Grasp(holding=None, source="opening" if opening_mm is not None else "raw",
                     reason=_wide_open(fraction, wide_open_fraction), **here)
    if empty is not None:
        return Grasp(holding=not empty, source="opening", **here,
                     reason="{:.1f} mm between the fingers, against a {:.0f} mm empty band"
                            .format(opening_mm, empty_band_m * 1000))

    if fraction is not None:
        holding = fraction > empty_band_fraction
        blind = "Nothing thinner than the pad gap at that endpoint registers at all"
        return Grasp(holding=holding, source="raw", **here,
                     reason="{:.0f} counts from the closed endpoint, {:.1%} of the span, "
                            "against a {:.0%} empty band. The width is uncalibrated, so this "
                            "says held or not, never how thick{}."
                            .format(float(reading.raw) - float(reading.raw_closed), fraction,
                                    empty_band_fraction,
                                    "" if holding else " -- and the closed endpoint is the end "
                                    "of travel, not the pads meeting. " + blind))

    return Grasp(holding=None, source="unavailable", **here,
                 reason="no width and no position count: nothing to judge a grasp by")


def _wide_open(fraction: float, threshold: float) -> str:
    return ("the jaws are {:.0%} of the way open after a close command, past the {:.0%} mark. "
            "An object that wide is less likely than a close that never executed -- check "
            "whether the gripper moved at all.".format(fraction, threshold))
