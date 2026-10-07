"""An oracle interception on a dynamic task, recorded to MP4: proof that the task is solvable
and that the grasp handoff (scripted motion -> physics) works. It reads the true trajectory
from the env -- privileged, so it is a demo of the ENVIRONMENT, not a policy.

    wait until the target will stay inside the reach window  ->  hover over it, matching
    its velocity, jaws turned across its narrow side  ->  descend  ->  close (the env hands
    the object to physics; a miss goes round again)  ->  lift
    ->  carry to the receiver  ->  release

    cd storm
    tools/on_gpu.sh 1 \
        python -m dynamic_libero_tasks.demo --only alphabet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio
import numpy as np

import dynamic_libero_tasks as D
from dynamic_libero_tasks.render import CAMERAS, FPS, make_env

REACH_Y = 0.30            # |y| the arm grasps inside comfortably
REACH_M = 0.70            # ...and how far from the base (x = -0.66)
HOVER = 0.12              # m above the grasp point while tracking
LEAD_S = 0.10             # aim this far ahead of a moving object, for the servo's lag
STEP_M = 0.05             # OSC_POSE: action 1.0 == 5 cm per control step


def world_box(sim, body_names):
    """World AABB (lo, hi) of every collision geom under the given bodies."""
    m, d = sim.model, sim.data
    ids = {m.body_name2id(b) for b in body_names}
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in range(m.ngeom):
        if m.geom_bodyid[g] not in ids or m.geom_contype[g] == 0:
            continue
        c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
        R = d.geom_xmat[g].reshape(3, 3)
        centre, half = d.geom_xpos[g] + R @ c, np.abs(R) @ h
        lo, hi = np.minimum(lo, centre - half), np.maximum(hi, centre + half)
    return lo, hi


def bodies_of(obj):
    return [obj.root_body] + [b for b in getattr(obj, "bodies", []) if b != obj.root_body]


def footprint(sim, obj):
    """The object's collision-box corners projected on the table: centre, narrow axis (unit,
    the way the jaws should close) and its width across that axis."""
    m, d = sim.model, sim.data
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    pts = [d.geom_xpos[g] + (m.geom_aabb[g][:3] + signs * m.geom_aabb[g][3:])
           @ d.geom_xmat[g].reshape(3, 3).T
           for g in range(m.ngeom) if m.geom_contype[g]
           and m.body_id2name(m.geom_bodyid[g]).startswith(obj.naming_prefix)]
    xy = np.concatenate(pts)[:, :2]
    centre = (xy.min(0) + xy.max(0)) / 2
    _, vecs = np.linalg.eigh(np.cov((xy - xy.mean(0)).T))
    narrow = vecs[:, 0]                                  # smallest spread
    width = float(np.ptp(xy @ narrow))
    return centre, narrow, width


def jaw_axis(sim):
    """World xy direction the fingers close along."""
    d, m = sim.data, sim.model
    left = d.body_xpos[m.body_name2id("gripper0_leftfinger")]
    right = d.body_xpos[m.body_name2id("gripper0_rightfinger")]
    v = (right - left)[:2]
    return v / np.linalg.norm(v)


class Oracle:
    JAWS_M = 0.075        # what the open jaws take; anything wider is taken by its rim

    def __init__(self, inner):
        self.e = inner
        target, receiver = inner.obj_of_interest[:2]
        self.target, self.receiver = inner.objects_dict[target], inner.objects_dict[receiver]
        self.name = target
        self.phase, self.ticks = "wait", 0
        lo, hi = world_box(inner.sim, bodies_of(self.target))
        self.height = hi[2] - lo[2]
        self.table = inner.table_offset[2]
        _, _, width = footprint(inner.sim, self.target)
        self.rim = width > self.JAWS_M
        self.grasp_z = self.table + (self.height - 0.015 if self.rim else 0.55 * self.height)

    def _future(self, dt):
        e = self.e
        t = float(e.sim.data.time) - e._t0 + dt
        return e.motion.state(e._anchors[self.name], t)

    def _grasp(self, dt):
        """Where to grasp ``dt`` s from now (xy) and the axis the jaws should close along.
        The footprint is read now and carried along the known motion."""
        centre, narrow, width = footprint(self.e.sim, self.target)
        if self.name in self.e.released:           # dropped: it lies still where it fell
            dt = 0.0
        now, then = self._future(0.0), self._future(dt)
        xy = centre + (then.xy - now.xy)
        turn = self.e.motion.omega * dt if self.e.motion.kind == "circular" else 0.0
        c, s = np.cos(turn), np.sin(turn)
        axis = np.array([c * narrow[0] - s * narrow[1], s * narrow[0] + c * narrow[1]])
        if self.rim:
            # the jaws straddle the wall (10% of the width in from the rim) on the side nearer
            # the robot's base
            ends = [xy + k * (0.4 * width) * axis for k in (-1, 1)]
            xy = min(ends, key=lambda p: np.hypot(p[0] + 0.66, p[1]))
        return xy, axis

    def _grasp_xy(self, dt):
        return self._grasp(dt)[0]

    def _stays_reachable(self, horizon=3.0):
        """Inside the reach window for the next `horizon` s."""
        return all(abs(p[1]) < REACH_Y and np.hypot(p[0] + 0.66, p[1]) < REACH_M
                   for p in (self._grasp_xy(dt) for dt in np.arange(0.0, horizon, 0.25)))

    def _yaw_action(self, want_axis):
        """World-z rotation that turns the jaw axis onto ``want_axis`` (either sign)."""
        have = jaw_axis(self.e.sim)
        err = np.arctan2(have[0] * want_axis[1] - have[1] * want_axis[0], have @ want_axis)
        err = (err + np.pi / 2) % np.pi - np.pi / 2
        return float(np.clip(err / 0.5 * 2.0, -1.0, 1.0)), abs(err)

    def act(self, eef):
        """One control step: the OSC action for the current phase."""
        self.ticks += 1
        grip, goal, yaw, yaw_err = -1.0, eef.copy(), 0.0, 0.0
        held = self.e._check_grasp(gripper=self.e.robots[0].gripper, object_geoms=self.target)
        if self.phase in ("wait", "track", "descend", "close") and not held:
            yaw, yaw_err = self._yaw_action(self._grasp(LEAD_S)[1])
        if self.phase == "wait":
            # sit over the belt's near entry of the reach window until the target is coming
            goal = np.r_[self._grasp_xy(LEAD_S), self.grasp_z + HOVER]
            goal[1] = np.clip(goal[1], -REACH_Y, REACH_Y)
            if self._stays_reachable():
                self.phase, self.ticks = "track", 0
        elif self.phase == "track":
            goal = np.r_[self._grasp_xy(LEAD_S), self.grasp_z + HOVER]
            if np.linalg.norm(goal[:2] - eef[:2]) < 0.01 and yaw_err < 0.05 and self.ticks > 10:
                self.phase, self.ticks = "descend", 0
        elif self.phase == "descend":
            goal = np.r_[self._grasp_xy(LEAD_S), self.grasp_z]
            if abs(eef[2] - self.grasp_z) < 0.008 or self.ticks > 60:
                self.phase, self.ticks = "close", 0
        elif self.phase == "close":
            grip = 1.0
            goal = np.r_[self._grasp_xy(LEAD_S) if not held else eef[:2], self.grasp_z]
            if self.ticks > 15:
                self.phase, self.ticks = "lift", 0
        elif self.phase == "lift":
            grip, goal = 1.0, np.r_[eef[:2], self.grasp_z + 0.18]
            if not held and self.ticks > 5:          # closed on nothing: go round again
                self.phase, self.ticks = "wait", 0
            elif eef[2] > self.grasp_z + 0.16:
                self.phase, self.ticks = "carry", 0
        elif self.phase == "carry":
            lo, hi = world_box(self.e.sim, bodies_of(self.receiver))
            if self.ticks == 1:
                # put the OBJECT's centre over the receiver, not the hand (a rim grasp is
                # off-centre); measured once, or the goal chases the object's sway in the hand
                olo, ohi = world_box(self.e.sim, bodies_of(self.target))
                self.hand_off = eef[:2] - (olo[:2] + ohi[:2]) / 2
            above = np.r_[(lo[:2] + hi[:2]) / 2 + self.hand_off,
                          max(eef[2], hi[2] + self.height + 0.06)]
            grip, goal = 1.0, above
            if np.linalg.norm(above[:2] - eef[:2]) < 0.01:
                self.place = np.r_[above[:2], hi[2] + 0.55 * self.height + 0.02]
                self.phase, self.ticks = "lower", 0
        elif self.phase == "lower":
            # a fixed xy: re-aiming at the hand's own xy lets the descent drift uncorrected
            grip, goal = 1.0, self.place
            if eef[2] < self.place[2] + 0.01 or self.ticks > 60:
                self.phase, self.ticks = "release", 0
        elif self.phase == "release":
            goal = self.place
            if self.ticks > 15:
                self.phase, self.ticks = "retreat", 0
        elif self.phase == "retreat":
            goal = np.r_[eef[:2], self.table + 0.35]
        a = np.zeros(7)
        a[:3] = np.clip((goal - eef) / STEP_M * 2.0, -1.0, 1.0)
        a[5] = yaw
        a[-1] = grip
        return a


def run(bddl: Path, seconds: float, size: int, out: Path, seed: int) -> dict:
    np.random.seed(seed)
    env = make_env(bddl, size, seconds)
    env.seed(seed)
    obs = env.reset()
    inner = env.env
    oracle = None
    frames, log, success = [], [], False
    for k in range(int(seconds * FPS)):
        if oracle is None and inner._anchors:
            oracle = Oracle(inner)
        a = oracle.act(obs["robot0_eef_pos"]) if oracle else np.r_[np.zeros(6), -1.0]
        obs, _, done, _ = env.step(a)
        success = success or bool(done)
        if oracle and (not log or log[-1][1] != oracle.phase):
            log.append((round(k / FPS, 2), oracle.phase))
        frames.append(np.concatenate([obs[f"{c}_image"][::-1] for c in CAMERAS], axis=1))
        if oracle and oracle.phase == "retreat" and oracle.ticks > 30:
            break
    env.close()
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"demo_{bddl.stem}.mp4"
    imageio.mimsave(path, frames, fps=FPS, macro_block_size=8)
    return {"video": str(path), "success": success, "released": sorted(inner.released),
            "phases": log}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "videos")
    a = ap.parse_args()
    D.register()
    for bddl in D.task_files():
        if a.only in bddl.stem:
            print(bddl.stem, run(bddl, a.seconds, a.size, a.out, a.seed), flush=True)


if __name__ == "__main__":
    main()
