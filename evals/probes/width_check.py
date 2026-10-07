"""How wide does the locator think a thing is, against how wide it actually is?"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from adapter.libero.adapter import LiberoAdapter        # noqa: E402
from src.executor import geometry                        # noqa: E402

# How far above an object's top face the synthetic wrist is put: the band the wrist is asked
# in at all (locate.WRIST_USEFUL_BELOW_MM is 150 mm of clearance).
WRIST_ABOVE_MM = 120.0


def _body_points(model, data, body_id):
    """Every mesh vertex of a body, in world coordinates."""
    points = []
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] != body_id:
            continue
        pos, mat = data.geom_xpos[geom], data.geom_xmat[geom].reshape(3, 3)
        if model.geom_type[geom] == 7:                   # a mesh
            mesh = model.geom_dataid[geom]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            local = model.mesh_vert[start:start + count].reshape(-1, 3)
        else:
            size = model.geom_size[geom]
            half = np.array([size[0], size[1] or size[0], size[2] or size[0]])
            local = np.array([[x, y, z] for x in (-half[0], half[0])
                              for y in (-half[1], half[1]) for z in (-half[2], half[2])])
        points.append((mat @ local.T).T + pos)
    return None if not points else np.vstack(points)


def _box(camera, points):
    """The box a perfect detector would draw round those points in that camera."""
    K = np.asarray(camera["intrinsic"], float).reshape(3, 3)
    T = np.asarray(camera["cam2base"], float).reshape(4, 4)
    in_camera = (points - T[:3, 3]) @ T[:3, :3]
    in_camera = in_camera[in_camera[:, 2] > 1e-6]
    if not len(in_camera):
        return None
    uv = in_camera @ K.T
    u, v = uv[:, 0] / uv[:, 2], uv[:, 1] / uv[:, 2]
    return [float(u.min()), float(v.min()), float(u.max()), float(v.max())]


def _nadir_over(like, centre, height_m):
    """A camera with ``like``'s optics, looking straight down from over ``centre``."""
    pose = np.eye(4)
    pose[:3, :3] = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    pose[:3, 3] = [float(centre[0]), float(centre[1]), float(height_m)]
    return {**like, "cam2base": pose.tolist()}


def _truth(points):
    """The widest and the narrowest way across the real footprint, in millimetres."""
    flat = points[:, :2]
    gaps = np.linalg.norm(flat[:, None, :] - flat[None, :, :], axis=-1)
    angles = np.linspace(0.0, np.pi, 180, endpoint=False)
    axes = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    spread = flat @ axes.T
    return float(gaps.max()) * 1000.0, \
        float((spread.max(axis=0) - spread.min(axis=0)).min()) * 1000.0


def check(task: int, seed: int, suite: str = "libero_10"):
    arm = LiberoAdapter(suite=suite, task_id=task, seed=seed)
    arm.connect()
    try:
        table_z = arm.capabilities().table_z_m
        cameras = arm.frames()["cameras"]
        sim = arm._env.env.sim
        model, data = sim.model, sim.data
        rows = []
        for body in range(model.nbody):
            name = model.body_id2name(body)
            if not name or not name.endswith("_main"):
                continue
            world = _body_points(model, data, body)
            if world is None or len(world) < 4:
                continue
            if len(world) > 4000:
                world = world[np.random.RandomState(0).choice(len(world), 4000, False)]
            base = np.array([arm.base_from_world(point) for point in world])
            span, narrow = _truth(base)
            top = float(base[:, 2].max())
            centre = base[:, :2].mean(axis=0)
            views = {n: c for n, c in cameras.items() if c.get("intrinsic")}
            wrist = views.get("wrist")
            if wrist is not None:
                views["wrist (over it)"] = _nadir_over(
                    wrist, centre, top + WRIST_ABOVE_MM / 1000.0)
            for view_name, view in views.items():
                box = _box(view, base)
                if box is None:
                    continue
                rows.append({
                    "task": task, "seed": seed, "object": name[:-5], "camera": view_name,
                    "nadir": geometry.looking_down(view),
                    "tall_mm": (top - table_z) * 1000.0,
                    "truth_mm": span, "narrow_mm": narrow,
                    "at_table": geometry.footprint_width_mm(view, box, table_z),
                    "at_top": geometry.footprint_width_mm(view, box, table_z, top_z_m=top)})
        return rows
    finally:
        arm.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--tasks", default="3,4")
    parser.add_argument("--seeds", default="0,1")
    args = parser.parse_args()
    rows = []
    for task in [int(t) for t in args.tasks.split(",")]:
        for seed in [int(s) for s in args.seeds.split(",")]:
            rows += check(task, seed, args.suite)
    print("\n| task/seed | object | camera | nadir | tall | truth | @table (before) "
          "| @top (after) |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        print("| {}/{} | {} | {} | {} | {:.0f} | {:.0f} | {:.0f} ({:+.0f}%) "
              "| {:.0f} ({:+.0f}%) |".format(
                  row["task"], row["seed"], row["object"], row["camera"],
                  "yes" if row["nadir"] else "no", row["tall_mm"], row["truth_mm"],
                  row["at_table"], 100.0 * (row["at_table"] / row["truth_mm"] - 1.0),
                  row["at_top"], 100.0 * (row["at_top"] / row["truth_mm"] - 1.0)))
    for nadir in (False, True):
        picked = [r for r in rows if r["nadir"] == nadir]
        if not picked:
            continue
        for column in ("at_table", "at_top"):
            errors = [abs(r[column] / r["truth_mm"] - 1.0) * 100.0 for r in picked]
            print("{:12s} {:9s} n={:2d}  median error {:5.1f}%  worst {:5.1f}%".format(
                "near-nadir" if nadir else "oblique", column, len(picked),
                float(np.median(errors)), float(np.max(errors))))


if __name__ == "__main__":
    main()
