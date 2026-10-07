"""Render each dynamic task to MP4 with the arm idle, and check the script is followed.

    cd storm
    tools/on_gpu.sh 1 \
        python -m dynamic_libero_tasks.render --seconds 20 --out dynamic_libero_tasks/videos
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio
import numpy as np

import dynamic_libero_tasks as D

CAMERAS = ("agentview", "birdview")
FPS = 20                                   # == LIBERO's control_freq, one frame per step


def make_env(bddl: Path, size: int, seconds: float = 60.0):
    from libero.libero.envs import OffScreenRenderEnv
    return OffScreenRenderEnv(bddl_file_name=str(bddl), camera_names=list(CAMERAS),
                              camera_heights=size, camera_widths=size,
                              horizon=int(seconds * FPS) + 10)


def tracking_error(inner) -> float:
    """Largest planar distance between a scripted object and its analytic trajectory."""
    t = float(inner.sim.data.time) - inner._t0
    worst = 0.0
    for name, anchor in inner._anchors.items():
        if name in inner.released or name in inner.gone:
            continue
        q = inner.sim.data.get_joint_qpos(inner.objects_dict[name].joints[-1])
        worst = max(worst, float(np.linalg.norm(q[:2] - inner.motion.state(anchor, t).xy)))
    return worst


def render(bddl: Path, seconds: float, size: int, out: Path, seed: int) -> dict:
    np.random.seed(seed)
    env = make_env(bddl, size, seconds)
    env.seed(seed)
    obs = env.reset()
    inner = env.env
    idle = np.zeros(7)
    idle[-1] = -1.0                                          # gripper open
    frames, worst = [], 0.0
    for _ in range(int(seconds * FPS)):
        obs, _, _, _ = env.step(idle)
        worst = max(worst, tracking_error(inner))
        # robosuite renders upside down
        frames.append(np.concatenate([obs[f"{c}_image"][::-1] for c in CAMERAS], axis=1))
    env.close()
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{bddl.stem}.mp4"
    imageio.mimsave(path, frames, fps=FPS, macro_block_size=8)
    return {"video": str(path), "max_tracking_err_mm": 1000 * worst,
            "fell_off": sorted(inner.gone)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="", help="substring of the task file to render")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "videos")
    a = ap.parse_args()
    D.register()
    for bddl in D.task_files():
        if a.only in bddl.stem:
            print(bddl.stem, render(bddl, a.seconds, a.size, a.out, a.seed), flush=True)


if __name__ == "__main__":
    main()
