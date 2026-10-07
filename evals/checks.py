"""Ground truth for the subgoal suite, read out of MuJoCo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from src.controller.clearance import grasp_state


@dataclass
class CheckResult:
    passed: bool
    detail: str
    measured: Optional[float] = None

    def __post_init__(self):
        # numpy comparisons give np.bool_ and np.float64, neither of which json.dump will take.
        self.passed = bool(self.passed)
        if self.measured is not None:
            self.measured = float(self.measured)

    def as_dict(self) -> Dict[str, Any]:
        return {"passed": self.passed, "detail": self.detail,
                "measured": None if self.measured is None else round(self.measured, 4)}


def _in_base(adapter, world) -> np.ndarray:
    world = np.asarray(world, dtype=float)
    return adapter.base_from_world(world) if hasattr(adapter, "base_from_world") else world


def body_position(adapter, name: str) -> np.ndarray:
    """Where an object is, in the same frame the adapter reports the tool in."""
    env = adapter._env.env
    return _in_base(adapter, env.sim.data.body_xpos[env.sim.model.body_name2id(name)])


def tool_position(adapter) -> np.ndarray:
    return adapter.tcp_pose().position_m


def gripper_opening_m(adapter) -> Optional[float]:
    reading = adapter.gripper_reading()
    return reading.opening_m


# --------------------------------------------------------------------------- the checks


def moved_axis(adapter, start, axis: str = "z", at_least_mm: float = None,
               at_most_mm: float = None) -> CheckResult:
    """Displacement along one base axis since the subgoal started."""
    index = "xyz".index(axis)
    moved = (tool_position(adapter)[index] - start["tool"][index]) * 1000.0
    if at_least_mm is None and at_most_mm is None:
        at_least_mm = 80.0
    ok = True
    wanted = []
    if at_least_mm is not None:
        ok = ok and moved >= at_least_mm
        wanted.append("at least {:+.0f}".format(at_least_mm))
    if at_most_mm is not None:
        ok = ok and moved <= at_most_mm
        wanted.append("at most {:+.0f}".format(at_most_mm))
    return CheckResult(ok, "moved {:+.0f} mm along {}, wanted {}".format(
        moved, axis, " and ".join(wanted)), moved)


def above_object(adapter, start, body: str = "", within_mm: float = 70.0) -> CheckResult:
    tool, target = tool_position(adapter), body_position(adapter, body)
    gap = float(np.linalg.norm(tool[:2] - target[:2])) * 1000.0
    return CheckResult(gap <= within_mm,
                       "{:.0f} mm from {} in the horizontal plane, wanted within {:.0f}".format(
                           gap, body, within_mm), gap)


def height_between(adapter, start, low_m: float = 0.0, high_m: float = 1.0) -> CheckResult:
    z = float(tool_position(adapter)[2])
    return CheckResult(low_m <= z <= high_m,
                       "tool at z = {:.3f} m, wanted {:.2f}-{:.2f}".format(z, low_m, high_m), z)


def _jaws(adapter, bound_mm: float, at_least: bool) -> CheckResult:
    opening = gripper_opening_m(adapter)
    if opening is None:
        return CheckResult(False, "the gripper reports no opening")
    millimetres = opening * 1000.0
    ok = millimetres >= bound_mm if at_least else millimetres <= bound_mm
    return CheckResult(ok, "jaws {:.1f} mm apart, wanted at {} {:.0f}".format(
        millimetres, "least" if at_least else "most", bound_mm), millimetres)


def gripper_open(adapter, start, at_least_mm: float = 60.0) -> CheckResult:
    return _jaws(adapter, at_least_mm, at_least=True)


def gripper_closed(adapter, start, at_most_mm: float = 10.0) -> CheckResult:
    return _jaws(adapter, at_most_mm, at_least=False)


def site_position(adapter, name: str) -> np.ndarray:
    """Where a BDDL goal region is."""
    env = adapter._env.env
    return _in_base(adapter, env.sim.data.site_xpos[env.sim.model.site_name2id(name)])


def _horizontal_gap_mm(tool, target) -> float:
    return float(np.linalg.norm(np.asarray(tool)[:2] - np.asarray(target)[:2])) * 1000.0


def above_site(adapter, start, site: str = "", within_mm: float = 80.0) -> CheckResult:
    gap = _horizontal_gap_mm(tool_position(adapter), site_position(adapter, site))
    return CheckResult(gap <= within_mm,
                       "{:.0f} mm from {} in the horizontal plane, wanted within {:.0f}".format(
                           gap, site, within_mm), gap)


def _near(adapter, target, name: str, within_mm: float) -> CheckResult:
    gap = float(np.linalg.norm(tool_position(adapter) - target)) * 1000.0
    return CheckResult(gap <= within_mm,
                       "{:.0f} mm from {} in 3-D, wanted within {:.0f}".format(
                           gap, name, within_mm), gap)


def near_object(adapter, start, body: str = "", within_mm: float = 90.0) -> CheckResult:
    """Straight-line distance, not just horizontal: this is what "the jaws are level with it
    rather than above it" means, and a purely horizontal check cannot tell those apart."""
    return _near(adapter, body_position(adapter, body), body, within_mm)


def near_site(adapter, start, site: str = "", within_mm: float = 140.0) -> CheckResult:
    return _near(adapter, site_position(adapter, site), site, within_mm)


def gripper_closed_or_holding(adapter, start, at_most_mm: float = 60.0) -> CheckResult:
    """Closing on an object stops the jaws well short of the 10 mm they reach on empty air,
    so the plain ``gripper_closed`` threshold reports a successful grasp as a failure."""
    state = grasp_state(adapter.gripper_reading())
    opening = gripper_opening_m(adapter)
    if opening is None:
        return CheckResult(bool(state.holding), "no width reading; holding={}".format(state.holding))
    millimetres = opening * 1000.0
    ok = bool(state.holding) or millimetres <= at_most_mm
    return CheckResult(ok, "jaws {:.1f} mm apart, holding={}".format(millimetres, state.holding),
                       millimetres)


def holding_object(adapter, start) -> CheckResult:
    """Did the jaws actually close on something, rather than on air?"""
    state = grasp_state(adapter.gripper_reading())
    opening = gripper_opening_m(adapter)
    millimetres = None if opening is None else opening * 1000.0
    return CheckResult(state.holding is True,
                       "holding={} ({})".format(state.holding, state.reason), millimetres)


def object_lifted(adapter, start, body: str = "", at_least_mm: float = 60.0,
                  still_held: bool = True) -> CheckResult:
    """Did the OBJECT go up -- not just the tool?"""
    now = float(body_position(adapter, body)[2]) * 1000.0
    then = (start.get("objects") or {}).get(body)
    if then is None:
        return CheckResult(False, "no start height recorded for {}".format(body))
    rise = now - float(np.asarray(then)[2]) * 1000.0
    held = grasp_state(adapter.gripper_reading()).holding
    passed = rise >= at_least_mm and (held is True or not still_held)
    return CheckResult(passed, "{} rose {:.0f} mm (wanted {:.0f}); holding={}".format(
        body, rise, at_least_mm, held), rise)


# --------------------------------------------------------------------------- grasp points

def body_extents(adapter, name: str):
    """World-frame bounding box of everything under a body, from its geoms' actual shapes."""
    env = adapter._env.env
    sim, model = env.sim, env.sim.model
    ids = [model.body_name2id(name)]
    grew = True
    while grew:                       # children too: the mesh may hang off a sub-body
        grew = False
        for b in range(model.nbody):
            if model.body_parentid[b] in ids and b not in ids:
                ids.append(b)
                grew = True
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in range(model.ngeom):
        if model.geom_bodyid[g] not in ids:
            continue
        pos = sim.data.geom_xpos[g]
        R = sim.data.geom_xmat[g].reshape(3, 3)
        if model.geom_type[g] == 7:                       # mesh
            mesh = model.geom_dataid[g]
            first, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            verts = model.mesh_vert[first:first + count] @ R.T + pos
        else:
            size = model.geom_size[g]
            verts = np.array([[sx, sy, sz] for sx in (-size[0], size[0])
                              for sy in (-size[1], size[1])
                              for sz in (-size[2], size[2])]) @ R.T + pos
        lo, hi = np.minimum(lo, verts.min(0)), np.maximum(hi, verts.max(0))
    return lo, hi


def grasp_point(adapter, body: str, part: str = "top") -> np.ndarray:
    """Where a gripper should actually go for an object, in the adapter's base frame."""
    lo, hi = body_extents(adapter, body)
    centre_world = (lo + hi) / 2.0
    top_world = centre_world.copy()
    top_world[2] = hi[2]
    top = adapter.base_from_world(top_world)
    if part == "top":
        return top
    if part == "rim":
        radius = float(min(hi[0] - lo[0], hi[1] - lo[1])) / 2.0
        pose = adapter.tcp_pose()
        jaw = np.asarray(pose.rotation, dtype=float) @ np.array([0.0, 1.0, 0.0])
        jaw[2] = 0.0
        norm = float(np.linalg.norm(jaw))
        if norm < 1e-6:                 # jaws vertical: no rim point is takeable
            return top
        jaw /= norm
        tool = pose.position_m
        ends = [top + radius * jaw, top - radius * jaw]
        return min(ends, key=lambda e: float(np.linalg.norm((tool - e)[:2])))
    raise ValueError("unknown part {!r}: top or rim".format(part))


def at_grasp_point(adapter, start, body: str = "", part: str = "top", within_mm: float = 25.0,
                   above_mm=(0.0, 50.0)) -> CheckResult:
    """Is the tool over the grasp point, at a height a grasp could be made from?"""
    target = grasp_point(adapter, body, part)
    diff = (tool_position(adapter) - target) * 1000.0
    gap = float(np.linalg.norm(diff[:2]))
    low, high = (float(v) for v in above_mm)
    height_ok = low <= float(diff[2]) <= high
    return CheckResult(gap <= within_mm and height_ok,
                       "{:.0f} mm off the {} of {} horizontally ({:+.0f} x, {:+.0f} y), wanted "
                       "within {:.0f}; {:+.0f} mm above it, wanted {:.0f} to {:.0f}".format(
                           gap, part, body, diff[0], diff[1], within_mm, diff[2], low, high),
                       gap)


def task_success(adapter, start) -> CheckResult:
    """The environment's own goal predicate -- the only thing that decides whether the task
    was actually done."""
    env = adapter._env.env
    done = bool(env._check_success())
    return CheckResult(done, "the task predicate says {}".format("SUCCESS" if done else "not done"))


CHECKS = {"moved_axis": moved_axis, "above_object": above_object,
          "height_between": height_between, "gripper_open": gripper_open,
          "gripper_closed": gripper_closed, "above_site": above_site,
          "near_object": near_object, "near_site": near_site,
          "gripper_closed_or_holding": gripper_closed_or_holding,
          "holding_object": holding_object, "task_success": task_success,
          "at_grasp_point": at_grasp_point, "object_lifted": object_lifted}


def run_check(spec: Dict[str, Any], adapter, start: Dict[str, Any]) -> CheckResult:
    arguments = {k: v for k, v in spec.items() if k != "fn"}
    function = CHECKS.get(spec.get("fn"))
    if function is None:
        return CheckResult(False, "no check named {!r}".format(spec.get("fn")))
    return function(adapter, start, **arguments)
