"""A simulated Panda in a LIBERO / robosuite env, driven by the same controller.

    action      [dx, dy, dz, drx, dry, drz, gripper]   7 numbers
"""

from __future__ import annotations

import base64
import io
import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from src.controller import types
from src.controller.transforms import (axis_angle_to_matrix, matrix_to_axis_angle,
                                       matrix_to_quat_wxyz, quat_wxyz_to_matrix,
                                       rotation_magnitude)
from src.controller.clearance import grasp_state
from src.controller.types import (Capabilities, GripperReading, Pose, Step, StepResult,
                                  TcpDelta)
from src.executor.geometry import depth_metres, points_in_box

import importlib.util as _imputil
import os as _os

# This repo's own copy of the LIBERO assets: sim/libero/{assets,bddl_files,init_files}.
SIM_LIBERO = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
    "sim", "libero")


def use_repo_libero() -> None:
    """Point LIBERO at SIM_LIBERO before it is imported, unless LIBERO_CONFIG_PATH is set.

    LIBERO reads its path table from $LIBERO_CONFIG_PATH/config.yaml; it is written here, for
    wherever this checkout lives, into an ignored directory, so ~/.libero is never touched.
    """
    if "LIBERO_CONFIG_PATH" in _os.environ:
        return
    # A namespace package: finding it does not run libero.libero, which reads the table.
    package = next(iter(_imputil.find_spec("libero").submodule_search_locations))
    # ...the meshes and textures from the LIBERO install when this checkout has no copy of them
    # (they are not committed: 617 MB), which is where LIBERO found them anyway.
    assets = _os.path.join(SIM_LIBERO, "assets")
    if not _os.path.isdir(assets):
        assets = _os.path.join(package, "libero", "assets")
    table = "".join("{}: {}\n".format(k, v if _os.path.isabs(v) else _os.path.join(SIM_LIBERO, v))
                    for k, v in (("assets", assets), ("bddl_files", "bddl_files"),
                                 ("init_states", "init_files"), ("datasets", "datasets"),
                                 ("benchmark_root", _os.path.join(package, "libero"))))
    config = _os.path.join(SIM_LIBERO, ".config", "config.yaml")
    if not _os.path.exists(config) or open(config).read() != table:
        _os.makedirs(_os.path.dirname(config), exist_ok=True)
        with open(config, "w") as f:
            f.write(table)
    _os.environ["LIBERO_CONFIG_PATH"] = _os.path.dirname(config)

# ---------------------------------------------------------------- the side camera's mount
#
# A SECOND FIXED CAMERA, commissioned. This is the one-time mount choice a rig makes with a
# tripod and a tape measure, written down: the camera is mounted where the workspace is seen
# from the other side -- the opposite side of the table from the scene camera -- at the scene
# camera's own standoff and looking down on the workspace at SIDE_ELEVATION_DEG.
#
# It is NOT derived from anything in the scene: the aim point is a spot on the table a fixed
# distance in front of the robot's own base (SIDE_AIM_BASE_XY, the middle of what this arm can
# reach), and the standoff is however far the scene camera already stands from that spot. So
# the rule reads the robot's calibration and the arena's table, never an object -- pointing a
# camera at where the objects happen to be today is not a mount, it is a cheat that moves
# every time the scene does.
#
# Why the +y side and not -y: measured 2026-09-19, from +y the cans' printing faces the
# camera and the whole-picture locate goes 7/8 against agentview's 3/8, while from -y it is
# 2/8. That is a fact about THIS room's labels, not about +y, and it does not transfer: the
# transferable form of it is "mount the second camera on the side the printing faces", which
# is a decision made once at commissioning and is what this constant records.
SIDE_AIM_BASE_XY = (0.47, 0.0)
SIDE_ELEVATION_DEG = 35.0
SIDE_SIGN_Y = +1.0
# Its own render size, the same as the other two: every camera renders 768, because that is
# what the planner is given and a camera cannot be asked for two sizes at once.
SIDE_CAMERA_PX = 768

GRIPPER_CLOSE = 1.0
GRIPPER_OPEN = -1.0
# The jaws stay where they are.
GRIPPER_HOLD = 0.0
# Fingers moving more than this per control step are still travelling.
MOVING_PER_STEP_M = 0.0005


def _look_at_quat(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """The MuJoCo camera quaternion (wxyz) that puts `target` in the middle of the frame."""
    eye = np.asarray(eye, dtype=float)
    forward = np.asarray(target, dtype=float) - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, dtype=float))
    right = right / np.linalg.norm(right)
    frame = np.column_stack([right, np.cross(right, forward), -forward])
    return np.asarray(matrix_to_quat_wxyz(frame), dtype=float)


def _pose_from_obs(obs: Dict[str, Any], base_pos=None, base_rot=None) -> Pose:
    """robosuite reports the end effector as position + an XYZW quaternion, in the WORLD
    frame."""
    x, y, z, w = (float(v) for v in obs["robot0_eef_quat"])
    position = np.asarray(obs["robot0_eef_pos"], dtype=float)
    rotation = quat_wxyz_to_matrix([w, x, y, z])
    if base_pos is not None and base_rot is not None:
        position = base_rot.T @ (position - base_pos)
        rotation = base_rot.T @ rotation
    return Pose.of(position, rotation)


class LiberoAdapter:
    """Implements ``src.controller.types.RobotAdapter`` for a LIBERO task."""

    # sqrt((110 + 94)^2 + 55^2) mm: the ray from this wrist camera to the point under the
    # tool at the clearance where the rig's depth camera goes blind. See frames().
    DEPTH_MIN_RANGE_M = 0.21

    def __init__(self, env=None, suite: str = "libero_10", task_id: int = 0, seed: int = 0,
                 camera_size: int = 768, side_camera_size: Optional[int] = None,
                 settle_steps: int = 32,
                 horizon: Optional[int] = None,
                 position_tolerance_m: float = 0.003,
                 rotation_tolerance_rad: float = 0.03, max_steps_per_segment: int = 200):
        self._env = env
        self.suite_name = suite
        #: The suite folder the task is actually in -- the same as `suite_name` except under
        #: an aggregate like `libero_pro`. Learned at connect.
        self.task_folder = suite
        self.task_id = task_id
        # Which of LIBERO's 50 saved initial states the task starts from.
        self.seed = int(seed)
        self._init_states: Optional[np.ndarray] = None
        # The scene and wrist cameras' render size.
        self.camera_size = camera_size
        # The side camera's own render size.
        self.side_camera_size = SIDE_CAMERA_PX if side_camera_size is None \
            else int(side_camera_size)
        self.settle_steps = settle_steps
        self.horizon = self.EPISODE_HORIZON if horizon is None else int(horizon)
        self.position_tolerance_m = position_tolerance_m
        self.rotation_tolerance_rad = rotation_tolerance_rad
        self.max_steps_per_segment = max_steps_per_segment
        self._obs: Optional[Dict[str, Any]] = None
        self._gripper = GRIPPER_OPEN
        self._last_opening: Optional[float] = None
        self._opening_creep = 0.0
        # the sim has no wall clock of its own; supervision needs a capture time that
        # advances once per rendered step, so it can tell a new picture from an old one
        self._frame_time = 0.0
        #: Set to ``recorder.on_frame`` to have every picture this adapter hands out kept.
        #: It is given the JPEG this adapter had already encoded, so it costs an append.
        self.on_frame = None
        # True once the episode has ended; every step after that is refused by robosuite.
        self._episode_over = False
        # Control ticks driven since the episode last began.
        self._ticks = 0
        # The task predicate went true at some point in this episode (for evals; the harness
        # never reads it).
        self._task_done = False
        # Where the robot's own base sits in robosuite's world frame, learned at connect.
        self._base_pos = None
        self._base_rot = None
        # The table's surface in base z, read ONCE at connect off the scene camera's depth.
        self._table_z_m: Optional[float] = None
        # MuJoCo's EGL context belongs to the thread that made it.
        self._pool: Optional[ThreadPoolExecutor] = None
        self.task_language = ""
        self.max_translation_m = 0.05
        self.max_rotation_rad = 0.5
        self.control_dt = 0.05
        self.open_span_m = 0.0788          # measured on a reset env, 2026-09-18
        # The pad face along the approach axis, measured off the gripper's own mesh against
        # the site the controller drives: it begins just above that point and ends below it.
        self.pad_length_m = 0.017

    EPISODE_HORIZON = 300000

    # How a step tells being blocked from still travelling, and how far it may wander.
    STALL_FRACTION = 0.25
    STALL_TICKS = 8
    STALL_SETTLE_TICKS = 3
    # How far out this arm has been WORKED: over round 11's twenty missions every grasp that
    # worked was inside 720 mm, and this is that number and the arena's, not the arm's. It is
    # reported and nothing is held to it.
    COMFORTABLE_REACH_M = 0.727
    # How far out this arm still goes where it is sent, measured with evals/probes/reach_envelope.py:
    # 780 mm is the last radius at which every move a grasp makes (down to a 20 mm outward
    # correction at grasp height) arrives in full; by 800 that correction delivers under 5 mm.
    REACH_M = 0.78
    # From jaws pointing down, asked for 90 degrees (2026-09-23, libero_goal drawer scenes):
    # where each tilt stops, and the lowest the tool point then came down to over clear table
    # (100-143 mm roll_cw, depending on where the arm stood).
    TILT_LIMITS_DEG = {"roll_cw": 88.6, "roll_ccw": 83.7, "pitch_down": 91.7, "pitch_up": 53.6}
    TILTED_FLOOR_M = 0.100
    LATERAL_FRACTION = 0.30
    VERTICAL_LATERAL_M = 0.005
    # The smallest that allowance may become however short the move.
    LATERAL_FLOOR_M = 0.004
    HOLD_TICKS = 2

    # ------------------------------------------------------------------ protocol

    def capabilities(self) -> Capabilities:
        return Capabilities(
            name="libero",
            max_translation_m=self.max_translation_m,
            max_rotation_rad=self.max_rotation_rad,
            # what the interface can be *commanded* at, not what the OSC will track
            max_speed_m_s=self.max_translation_m / self.control_dt,
            max_rotation_speed_rad_s=self.max_rotation_rad / self.control_dt,
            # measured: at 12.5 mm/s the per-step delta is 0.6 mm and the OSC tracks about a
            # sixth of it. 50 mm/s gives 2.5 mm a step, which it follows.
            default_speed_m_s=0.05,
            default_rotation_speed_rad_s=math.radians(60.0),
            control_point_offset_m=0.0,
            # A step stops the moment it stops making headway along its own axis, which is
            # what a contact stop is for.
            supports_contact_stop=True,
            contact_stop_trusted=True,
            supports_home=True,
            gripper_gap_measured=True,
            gripper_open_span_m=self.open_span_m,
            gripper_pad_length_m=self.pad_length_m,
            table_z_m=self._table_z_m,
            comfortable_reach_m=self.COMFORTABLE_REACH_M,
            reach_m=self.REACH_M,
            arrival_tolerance_m=self.position_tolerance_m,
            tilt_limits_deg=dict(self.TILT_LIMITS_DEG), tilted_floor_m=self.TILTED_FLOOR_M,
            # measured here: the drawer scenes' tilted takes, the tallest LIBERO object, and the
            # subgoal median and verdict time of the round-30 sweeps
            tilt_reach_deg=80.0, max_object_height_m=0.16, subgoal_median_s=21.0,
            subgoal_verdict_s=4.6,
            notes=["home is env.reset(): it restores the task's initial state, not a pose",
                   "a motion stops itself when it stops making headway along its own axis, "
                   "and says how far it got",
                   "this simulator ends an episode after {:,} controller steps and accepts "
                   "nothing after that; a long mission has been measured at under a tenth "
                   "of it".format(self.horizon)],
        )

    def connect(self) -> None:
        self._ensure_pool()
        self._on_env_thread(self._connect)

    def _ensure_pool(self) -> None:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="libero-env")

    def _on_env_thread(self, fn, *args):
        """Everything that touches MuJoCo goes through the one thread that owns its context."""
        if self._pool is None:
            return fn(*args)
        return self._pool.submit(fn, *args).result()

    def _connect(self) -> None:
        if self._env is None:
            use_repo_libero()
            from libero.libero import benchmark, get_libero_path
            from libero.libero.envs import OffScreenRenderEnv

            from . import pro
            # LIBERO-PRO's four real perturbation suites per base suite, vendored under
            # sim/libero. Their languages come from the BDDL, not the (unchanged) filename.
            pro.register()
            suite = benchmark.get_benchmark_dict()[self.suite_name]()
            task = suite.get_task(self.task_id)
            self._init_states = self._states_for(suite)
            bddl = _os.path.join(get_libero_path("bddl_files"), task.problem_folder,
                                task.bddl_file)
            self.task_language = task.language
            # `libero_pro` is twenty suites end to end: this is the one THIS task came from,
            # which is what a 200-task report has to group by.
            self.task_folder = task.problem_folder
            # camera_depths is what makes the wrist a range sensor rather than just a picture.
            names = [self.MUJOCO_CAMERAS[name] for name, _ in self.CAMERAS]
            sizes = [self.side_camera_size if name == "side" else self.camera_size
                     for name, _ in self.CAMERAS]
            depths = [name in self.DEPTHS for name, _ in self.CAMERAS]
            self._env = OffScreenRenderEnv(bddl_file_name=bddl,
                                           camera_names=names,
                                           camera_heights=sizes,
                                           camera_widths=sizes,
                                           camera_depths=depths,
                                           horizon=self.horizon)
        if self.seed and self._init_states is None:
            raise RuntimeError(
                "seed {} was asked for but {} task {} ships no initial states here -- only "
                "seed 0 (the plain reset) is available".format(self.seed, self.suite_name,
                                                               self.task_id))
        self._obs = self._reset_and_settle()
        self._frame_time = time.monotonic()
        robot = self._env.env.robots[0]
        controller = robot.controller
        self.max_translation_m = float(np.max(controller.output_max[:3]))
        self.max_rotation_rad = float(np.max(controller.output_max[3:]))
        self.control_dt = 1.0 / float(self._env.env.control_freq)
        self._read_base_frame()
        self._table_z_m = self._measure_table()

    def _measure_table(self) -> Optional[float]:
        """The table's surface in base z, from the scene camera's depth, read once: the height
        the most returns share, since a level surface puts all of its returns at one height and
        a wall or an arm spreads them over many."""
        camera = dict(self.frames(["scene"])["cameras"].get("scene") or {},
                      depth_png16=self._depth_png16("scene", self._env.env.sim))
        depth = depth_metres(camera)
        points = None if depth is None else points_in_box(
            camera, [0, 0, camera["width"], camera["height"]], depth, step=2)
        if points is None or len(points) < 100:
            return None
        z = points[:, 2]
        counts, edges = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.01, 0.005))
        mode = float(edges[int(np.argmax(counts))]) + 0.0025
        return float(np.median(z[np.abs(z - mode) <= 0.005]))

    def _reset_and_settle(self):
        """``env.reset()`` and then let the scene come to rest."""
        self._env.reset()
        # hard_reset=True rebuilds the MjSim from the arena XML, which puts `sideview` back
        # where the room's own XML left it -- 1.68 m out and aimed at the room, where the
        # objects fall off the bottom of the frame.
        self._place_side_camera()
        if self._init_states is not None:
            self._obs = self._env.set_init_state(self._init_states[self.seed])
        self._drive_now(np.zeros(6), self.settle_steps)    # updates self._obs as it goes
        return self._obs

    def _place_side_camera(self) -> None:
        """Put `sideview` on its commissioned mount."""
        try:
            sim = self._env.env.sim
            model, data = sim.model, sim.data
            side = model.camera_name2id("sideview")
            scene = model.camera_name2id(self.MUJOCO_CAMERAS["scene"])
        except Exception:
            return
        index = self._base_body(model)
        base = None if index is None else np.asarray(data.body_xpos[index], dtype=float)
        table_z = self._table_top_world(model, data)
        if base is None or table_z is None:
            return
        aim = np.array([base[0] + SIDE_AIM_BASE_XY[0], base[1] + SIDE_AIM_BASE_XY[1],
                        float(table_z)])
        scene_pos = np.asarray(model.cam_pos[scene], dtype=float)
        standoff = float(np.linalg.norm(scene_pos - aim))
        elevation = math.radians(SIDE_ELEVATION_DEG)
        eye = aim + np.array([0.0, SIDE_SIGN_Y * standoff * math.cos(elevation),
                              standoff * math.sin(elevation)])
        model.cam_pos[side] = eye
        model.cam_quat[side] = _look_at_quat(eye, aim)
        model.cam_fovy[side] = float(model.cam_fovy[scene])
        # cam_xpos / cam_xmat -- which is what the extrinsic is read from -- are derived, so
        # nothing has moved until the model is stepped forward.
        sim.forward()

    #: A thing may rest with its body origin a little below the surface it stands on.
    STANDS_BELOW_M = 0.03

    @staticmethod
    def _table_top_world(model, data) -> Optional[float]:
        """Where the side camera's mount is aimed, in world z: the top of the static geom the
        most free-moving things stand on -- their xy inside its footprint and their origins not
        below its top -- then the highest. Scene setup only, like mounting a camera; nothing the
        harness perceives comes from it."""
        things = [np.asarray(data.body_xpos[body], dtype=float)
                  for body in range(model.nbody)
                  if any(int(model.jnt_type[joint]) == 0          # a free joint: it can move
                         for joint in range(model.njnt)
                         if model.jnt_bodyid[joint] == body)]
        best = None
        for geom in range(model.ngeom):
            body = int(model.geom_bodyid[geom])
            if model.body_dofnum[body] or model.body_jntnum[body]:
                continue
            kind = int(model.geom_type[geom])
            size = np.asarray(model.geom_size[geom], dtype=float)
            if kind == 6:
                half = size
            elif kind in (3, 5):
                half = np.array([size[0], size[0], size[1]])
            elif kind == 0:
                # a ground plane: a zero half-size means it goes on for ever
                half = np.array([size[0] or 1e3, size[1] or 1e3, 0.0])
            else:
                continue                       # a mesh has no cheap top face; skip it
            rotation = np.asarray(data.geom_xmat[geom], dtype=float).reshape(3, 3)
            here = np.asarray(data.geom_xpos[geom], dtype=float)
            reach = np.abs(rotation) @ half    # extent along each WORLD axis, tilt and all
            top = float(here[2] + reach[2])
            standing = sum(1 for thing in things
                           if abs(thing[0] - here[0]) <= reach[0]
                           and abs(thing[1] - here[1]) <= reach[1]
                           and thing[2] >= top - LiberoAdapter.STANDS_BELOW_M)
            if standing and (best is None or (standing, top) > best):
                best = (standing, top)
        return None if best is None else best[1]

    def _states_for(self, suite):
        """LIBERO's saved initial states for this task -- 50 of them -- or None."""
        try:
            states = np.asarray(suite.get_task_init_states(self.task_id))
        except Exception:
            return None
        if len(states) == 0:
            return None
        if self.seed >= len(states):
            raise ValueError("{} task {} ships {} initial states, so seed {} does not exist"
                             .format(self.suite_name, self.task_id, len(states), self.seed))
        return states

    @staticmethod
    def _base_body(model) -> Optional[int]:
        """The body id of the robot's base, whichever name this robot model gives it."""
        for name in ("robot0_base", "robot0_link0", "robot0_fixed_base_link"):
            try:
                return model.body_name2id(name)
            except Exception:
                continue
        return None

    def _read_base_frame(self) -> None:
        """Find the robot base in world, so every pose can be reported relative to it."""
        sim = self._env.env.sim
        index = self._base_body(sim.model)
        if index is None:
            # Better to report world coordinates and say so than to invent a base.
            self._base_pos = self._base_rot = None
            return
        self._base_pos = np.asarray(sim.data.body_xpos[index], dtype=float).copy()
        self._base_rot = np.asarray(sim.data.body_xmat[index], dtype=float).reshape(3, 3).copy()

    def _to_base_frame(self, transform) -> np.ndarray:
        """A 4x4 pose in world -> the same pose in the robot's base frame."""
        matrix = np.asarray(transform, dtype=float)
        if self._base_pos is None or self._base_rot is None:
            return matrix
        base = np.eye(4)
        base[:3, :3] = self._base_rot
        base[:3, 3] = self._base_pos
        return np.linalg.inv(base) @ matrix

    def base_from_world(self, position) -> np.ndarray:
        point = np.asarray(position, dtype=float)
        if self._base_pos is None or self._base_rot is None:
            return point
        return self._base_rot.T @ (point - self._base_pos)

    def tcp_pose(self) -> Pose:
        if self._obs is None:
            raise RuntimeError("connect() first")
        return _pose_from_obs(self._obs, self._base_pos, self._base_rot)

    def episode_over(self) -> bool:
        """Has the simulator stopped accepting motion?"""
        return bool(self._episode_over)

    def _lateral_limit(self, asked_m: float, vertical: bool) -> float:
        """How far this step may wander off its own axis before it is called a skid."""
        limit = self.LATERAL_FRACTION * asked_m if asked_m else 0.0
        if not limit:
            return limit
        limit = max(limit, self.LATERAL_FLOOR_M)
        if grasp_state(self.gripper_reading()).holding is True:
            return max(types.PAYLOAD_LATERAL_M, limit)
        return min(limit, self.VERTICAL_LATERAL_M) if vertical else limit

    def ticks(self) -> Dict[str, Any]:
        """How many control ticks this episode has spent, out of what it is allowed."""
        env = getattr(self._env, "env", None)
        return {"driven": self._ticks,
                "env_timestep": None if env is None else int(getattr(env, "timestep", 0)),
                "env_horizon": None if env is None else int(getattr(env, "horizon", 0)),
                "asked_for": int(self.horizon)}

    def stop(self) -> None:
        """Hold position: an OSC delta of zero, which is what stopping means here."""
        self._drive(np.zeros(6), steps=1)

    def close(self) -> None:
        if self._env is not None:
            self._on_env_thread(self._env.close)
            self._env = None
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    # ------------------------------------------------------------------ tasks

    def set_task(self, suite: str, task_id: int) -> None:
        """Rebuild the environment for another task, on the env's own thread."""
        # Check before tearing anything down, so a bad name leaves the working environment in
        # place.
        known = self.available_tasks()
        if suite not in known:
            raise ValueError("unknown suite {!r}; the suites here are {}".format(
                suite, ", ".join(sorted(known))))
        if not 0 <= int(task_id) < len(known[suite]):
            raise ValueError("{} has tasks 0-{}, not {}".format(
                suite, len(known[suite]) - 1, task_id))
        previous = (self.suite_name, self.task_id)
        if self._env is not None:
            self._on_env_thread(self._env.close)
            self._env = None
        self.suite_name, self.task_id = suite, int(task_id)
        self._episode_over, self._ticks, self._task_done = False, 0, False
        self._gripper = GRIPPER_OPEN
        self._obs = None
        self._ensure_pool()
        try:
            self._on_env_thread(self._connect)
        except Exception:
            # a scene that will not build (a missing asset, say) must not take the working
            # one with it: put the previous task back, then report the failure
            self._env = None
            self.suite_name, self.task_id = previous
            self._on_env_thread(self._connect)
            raise

    @staticmethod
    def available_tasks() -> Dict[str, List[Dict[str, Any]]]:
        """Every suite LIBERO knows here, base and LIBERO-PRO alike, with task languages."""
        use_repo_libero()
        from libero.libero.benchmark import BENCHMARK_MAPPING, libero_task_map, task_maps

        from . import pro
        pro.register()
        out: Dict[str, List[Dict[str, Any]]] = {}
        for suite in BENCHMARK_MAPPING:
            names = libero_task_map.get(suite) or []
            if not names or suite not in task_maps:
                continue
            out[suite] = [{"index": i, "language": task_maps[suite][n].language}
                          for i, n in enumerate(names)]
        return out

    # ------------------------------------------------------------------ planning

    def steps_for(self, delta: TcpDelta, start: Pose) -> List[Step]:
        if delta.kind == types.GRIPPER:
            return [Step(delta_index=delta.index, label=delta.label, kind=types.GRIPPER,
                         duration_s=self.settle_steps * self.control_dt,
                         detail={"state": delta.gripper_state,
                                 "open_to_mm": delta.open_to_mm})]
        if delta.kind == types.WAIT:
            return [Step(delta_index=delta.index, label=delta.label, kind=types.WAIT,
                         duration_s=delta.seconds, detail={"seconds": delta.seconds})]
        if delta.kind == types.HOME:
            return [Step(delta_index=delta.index, label=delta.label, kind=types.HOME,
                         duration_s=0.5)]

        if delta.kind == types.TRANSLATE:
            total, cap = delta.magnitude, self.max_translation_m
        else:
            total, cap = delta.magnitude, self.max_rotation_rad
        n = max(1, math.ceil(total / cap - 1e-9))

        steps, previous = [], start
        for i in range(1, n + 1):
            fraction = i / float(n)
            if delta.kind == types.TRANSLATE:
                goal = start.moved(delta.axis * (delta.magnitude * fraction))
                span = float(np.linalg.norm(goal.position_m - previous.position_m))
            else:
                goal = start.turned(axis_angle_to_matrix(delta.axis, delta.magnitude * fraction))
                span = rotation_magnitude(goal.rotation @ previous.rotation.T)
            steps.append(Step(
                delta_index=delta.index, chunk=i, of=n, kind=delta.kind,
                label="{} [{}/{}]".format(delta.label, i, n), goal=goal,
                duration_s=max(self.control_dt, span / delta.speed if delta.speed else 0.0),
                detail={"distance_m": span if delta.kind == types.TRANSLATE else 0.0,
                        "speed": delta.speed,
                        "push": bool(delta.push),
                        "stop_on_contact": bool(delta.stop_on_contact or delta.push),
                        "axis": None if delta.axis is None else [round(float(v), 4)
                                                                 for v in delta.axis]}))
            previous = goal
        return steps

    # ------------------------------------------------------------------ execution

    def run_step(self, step: Step) -> StepResult:
        before = self.tcp_pose()
        started = time.monotonic()

        # Everything but home: home IS the way out, and the refusal below tells the caller so.
        if self._episode_over and step.kind != types.HOME:
            return StepResult(step=step, outcome=types.REFUSED, pose_after=before,
                              elapsed_s=time.monotonic() - started,
                              message="this LIBERO episode has ended -- it reached its step "
                                      "horizon ({}), so the simulator will not accept any "
                                      "further motion. Send a \"home\" action to reset the "
                                      "task and start again.".format(self.ticks()))

        if step.kind == types.WAIT:
            self._drive(np.zeros(6), steps=max(1, int(step.detail["seconds"] / self.control_dt)))
            return StepResult(step=step, outcome=types.DONE, pose_after=self.tcp_pose(),
                              elapsed_s=time.monotonic() - started)

        if step.kind == types.GRIPPER:
            asked_mm = step.detail.get("open_to_mm")
            if step.detail["state"] == "open" and asked_mm is not None \
                    and self._opening() is not None:
                return self._open_to(step, float(asked_mm), started)
            self._gripper = GRIPPER_CLOSE if step.detail["state"] == "close" else GRIPPER_OPEN
            # Long enough for the fingers to actually stop: an empty close reads 21 mm after
            # 8 control steps and only settles at 1.0 mm by about 32 (measured 2026-09-18).
            self._drive(np.zeros(6), steps=self.settle_steps)
            grasp = grasp_state(self.gripper_reading())
            message = ("holding something" if grasp.holding
                       else "nothing held" if grasp.holding is False
                       else "grasp unknown") + " -- " + grasp.reason
            opening = self._opening()
            return StepResult(step=step, outcome=types.DONE, message=message,
                              detail={} if opening is None
                              else {"opening_mm": round(opening * 1000.0, 2)},
                              pose_after=self.tcp_pose(), elapsed_s=time.monotonic() - started)

        if step.kind == types.HOME:
            self._obs = self._on_env_thread(self._reset_and_settle)
            self._episode_over, self._ticks, self._task_done = False, 0, False
            self._frame_time += self.control_dt
            self._gripper = GRIPPER_OPEN
            return StepResult(step=step, outcome=types.DONE, message="env.reset()",
                              pose_after=self.tcp_pose(), elapsed_s=time.monotonic() - started)

        goal = step.goal
        speed = float(step.detail.get("speed") or 0.0)
        per_step_m = min(self.max_translation_m,
                         speed * self.control_dt if speed else self.max_translation_m)
        per_step_rad = min(self.max_rotation_rad,
                           speed * self.control_dt if speed else self.max_rotation_rad)
        stalled = 0
        commanded_axis = step.detail.get("axis")
        unit_axis = None if not commanded_axis or step.kind != types.TRANSLATE \
            else np.asarray(commanded_axis, dtype=float)
        pushing = bool(step.detail.get("push"))
        vertical = unit_axis is not None and abs(float(unit_axis[2])) > 0.99
        descending = vertical and float(unit_axis[2]) < 0.0
        asked_m = float(step.detail.get("distance_m") or 0.0)
        lateral_limit = self._lateral_limit(asked_m, vertical)
        blocked, best_rate, abort = 0, 0.0, ""
        for tick in range(self.max_steps_per_segment):
            here = self.tcp_pose()
            gap = goal.position_m - here.position_m
            axis, angle = matrix_to_axis_angle(goal.rotation @ here.rotation.T)
            if (np.linalg.norm(gap) <= self.position_tolerance_m
                    and angle <= self.rotation_tolerance_rad):
                break
            command = np.zeros(6)
            distance = float(np.linalg.norm(gap))
            if distance > 1e-9:
                command[:3] = gap / distance * min(distance, per_step_m)
            if angle > 1e-9:
                command[3:] = axis * min(angle, per_step_rad)
            self._drive(command, steps=1)
            now = self.tcp_pose()
            if unit_axis is not None:
                # Progress ALONG the commanded axis, which is the only progress this step is
                # about.
                rate = float((now.position_m - here.position_m) @ unit_axis)
                best_rate = max(best_rate, rate)
                short = float(np.linalg.norm(goal.position_m - now.position_m)) \
                    > self.position_tolerance_m * 3
                if tick >= self.STALL_SETTLE_TICKS and best_rate > 0.0 and short:
                    if rate < self.STALL_FRACTION * best_rate:
                        blocked += 1
                        if blocked >= self.STALL_TICKS:
                            abort = "blocked"
                            break
                    else:
                        blocked = 0
                travelled = now.position_m - before.position_m
                along_now = float(travelled @ unit_axis)
                lateral_now = float(np.linalg.norm(travelled - along_now * unit_axis))
                if lateral_limit and lateral_now > lateral_limit:
                    # Wandering while it travels is being pushed off its line.
                    abort = "blocked" if descending and along_now <= self.position_tolerance_m \
                        else "skidded"
                    break
            # A pure rotation barely moves the tool point, so a position-only stall test
            # calls it stuck after a dozen steps and abandons the turn. Watch both.
            crept = (float(np.linalg.norm(now.position_m - here.position_m))
                     + rotation_magnitude(now.rotation @ here.rotation.T) * 0.05)
            if crept < 1e-5:
                stalled += 1
                if stalled > 12:
                    break
            else:
                stalled = 0
        if abort:
            # Stop commanding and HOLD where the tool is.
            self._drive(np.zeros(6), steps=self.HOLD_TICKS)

        after = self.tcp_pose()
        moved = after.position_m - before.position_m
        axis = step.detail.get("axis")
        if axis and step.kind == types.TRANSLATE:
            a = np.asarray(axis, dtype=float)
            along = float(moved @ a)
            lateral = float(np.linalg.norm(moved - along * a))
        else:
            along, lateral = float(np.linalg.norm(moved)), 0.0
        remaining_m = float(np.linalg.norm(goal.position_m - after.position_m))
        remaining_rad = rotation_magnitude(goal.rotation @ after.rotation.T)
        # Both are checked: a rotation about the tool point hardly moves it.
        short_position = remaining_m > self.position_tolerance_m * 3
        short_rotation = remaining_rad > self.rotation_tolerance_rad * 3
        outcome = types.LIMITED if (short_position or short_rotation) else types.DONE
        shortfall = []
        if short_position:
            shortfall.append("{:.1f} mm".format(remaining_m * 1000.0))
        if short_rotation:
            shortfall.append("{:.1f} deg".format(float(np.degrees(remaining_rad))))
        message = "" if outcome == types.DONE else \
            "stopped {} short of the goal".format(" and ".join(shortfall))
        sideways = "moved {:.0f} mm along the axis and {:.0f} mm sideways".format(
            along * 1000.0, lateral * 1000.0)
        if abort == "blocked":
            # A push is MEANT to end against something: arriving there is the result, not a
            # failure, and the distance it got is what the caller wants back.
            if pushing:
                outcome, message = types.CONTACT, (
                    "pushed {:.0f} mm and stopped against something -- it will not go "
                    "further this way".format(along * 1000.0))
            else:
                outcome = types.LIMITED
                message = ("blocked: {}, then stopped making headway and was stopped "
                           "there -- something is under the tool, or the arm will not go "
                           "further this way. If this was a descent onto the object, that "
                           "is the fingers meeting it.".format(sideways))
        elif abort == "skidded":
            outcome = types.LIMITED
            message = ("skidded: {} -- more sideways than this motion was allowed to "
                       "drift, so it was stopped. Whatever is under the tool is pushing "
                       "it aside; measure where the target is again before moving on "
                       "this way.".format(sideways))
        elif outcome == types.LIMITED and step.kind == types.TRANSLATE:
            message = "{} ({})".format(message, sideways)
        elif outcome == types.DONE and pushing:
            message = "pushed {:.0f} mm without meeting anything".format(along * 1000.0)
        return StepResult(
            step=step, outcome=outcome, message=message, abort_reason=abort,
            moved_m=float(np.linalg.norm(moved)), along_axis_m=along, lateral_m=lateral,
            turned_rad=rotation_magnitude(after.rotation @ before.rotation.T),
            pose_after=after, elapsed_s=time.monotonic() - started)

    # ------------------------------------------------------------------ optional extras
    #
    # Not part of RobotAdapter: the page and the executor feature-detect these. Without them
    # the executor would be driving this simulator blind.

    # "side" is the commissioned second fixed camera -- see SIDE_AIM_BASE_XY.
    CAMERAS = (("scene", "agentview_image"), ("side", "sideview_image"),
               ("wrist", "robot0_eye_in_hand_image"))
    DEPTHS = {"scene": "agentview_depth", "wrist": "robot0_eye_in_hand_depth"}
    MUJOCO_CAMERAS = {"scene": "agentview", "side": "sideview",
                      "wrist": "robot0_eye_in_hand"}

    def camera_names(self) -> List[str]:
        return [name for name, _ in self.CAMERAS]

    def frames(self, cameras: Optional[List[str]] = None) -> Dict[str, Any]:
        """The same payload shape the xArm bridge serves, so one page renders both."""
        from robosuite.utils.camera_utils import (get_camera_extrinsic_matrix,
                                                  get_camera_intrinsic_matrix)

        wanted = set(cameras) if cameras else set(self.camera_names())
        sim = self._env.env.sim
        out: Dict[str, Any] = {}
        for name, key in self.CAMERAS:
            if name not in wanted:
                continue
            image = (self._obs or {}).get(key)
            if image is None:
                continue
            upright = np.asarray(image)[::-1]
            buffer = io.BytesIO()
            Image.fromarray(upright.astype(np.uint8)).save(buffer, format="JPEG", quality=90)
            height, width = upright.shape[:2]
            try:
                # robosuite hands back camera-to-WORLD.
                extrinsic = self._to_base_frame(
                    get_camera_extrinsic_matrix(sim, self.MUJOCO_CAMERAS[name]))
                intrinsic = get_camera_intrinsic_matrix(sim, self.MUJOCO_CAMERAS[name],
                                                        height, width)
            except Exception:
                extrinsic = intrinsic = None
            # Only the wrist is a range sensor to the harness; the scene camera's depth is read
            # once, to commission the table (_measure_table).
            depth_blob = self._depth_png16(name, sim) if name == "wrist" else None
            if self.on_frame is not None:
                try:
                    self.on_frame(name, buffer.getvalue(), width, height, self._frame_time)
                except Exception:
                    pass                  # a lost frame is never a lost picture
            out[name] = {"rgb_jpeg": base64.b64encode(buffer.getvalue()).decode("ascii"),
                         "depth_png16": depth_blob,
                         # A simulated camera sees everything, down to the fingers; the rig's
                         # RealSense returns nothing inside ~0.30 m. Applying that limit
                         # unchanged here was measured to blind the executor below ~200 mm of
                         # clearance, because this camera sits 94 mm above the tool while the
                         # rig's sits 122: the rig reads down to ~110 mm.
                         "depth_min_range_m": self.DEPTH_MIN_RANGE_M,
                         "width": width, "height": height,
                         "timestamp": self._frame_time,
                         "intrinsic": None if intrinsic is None else np.asarray(intrinsic).tolist(),
                         "cam2base": None if extrinsic is None else np.asarray(extrinsic).tolist()}
        return {"cameras": out}

    def _depth_png16(self, name: str, sim) -> Optional[str]:
        """A camera's depth as a 16-bit PNG in millimetres, the shape the bridge sends."""
        raw = (self._obs or {}).get(self.DEPTHS.get(name, ""))
        if raw is None:
            return None
        try:
            from robosuite.utils.camera_utils import get_real_depth_map
            metres = np.asarray(get_real_depth_map(sim, np.asarray(raw))).squeeze()
        except Exception:
            return None
        upright = metres[::-1]
        millimetres = np.clip(np.nan_to_num(upright) * 1000.0, 0, 65535).astype(np.uint16)
        buffer = io.BytesIO()
        Image.fromarray(millimetres, mode="I;16").save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def extra_status(self) -> Dict[str, Any]:
        """Anything the page should know that is not a pose or a gripper."""
        return {"suite": self.suite_name, "task_id": self.task_id,
                "task_language": self.task_language,
                "episode_over": self._episode_over, "ticks": self.ticks(),
                "episode_note": ("this episode has ended (LIBERO step horizon); send a home "
                                 "action to reset the task" if self._episode_over else "")}

    def gripper_reading(self) -> GripperReading:
        """The Panda's two finger joints: their separation is the opening."""
        opening = self._opening()
        if opening is None:
            return GripperReading(commanded=self._commanded())
        return GripperReading(opening_m=opening, open_span_m=self.open_span_m,
                              commanded=self._commanded(),
                              moving=self._opening_creep > MOVING_PER_STEP_M)

    def _opening(self) -> Optional[float]:
        qpos = (self._obs or {}).get("robot0_gripper_qpos")
        return None if qpos is None else abs(float(qpos[0]) - float(qpos[1]))

    def _commanded(self) -> str:
        return "close" if self._gripper > 0 else "open"

    def _open_to(self, step, asked_mm: float, started: float) -> StepResult:
        """Open the jaws only as far as asked, hold them there, and let the scene settle."""
        asked_m = max(0.0, float(asked_mm) / 1000.0)
        self._on_env_thread(self._open_to_now, asked_m, self.settle_steps)
        # The jaws are off the thing; now stand still while it comes to rest.
        self._drive(np.zeros(6), steps=self.settle_steps)
        opening = self._opening()
        gap_mm = None if opening is None else opening * 1000.0
        return StepResult(
            step=step, outcome=types.DONE, pose_after=self.tcp_pose(),
            elapsed_s=time.monotonic() - started,
            detail={"asked_mm": round(float(asked_mm), 1),
                    **({} if gap_mm is None else {"opening_mm": round(gap_mm, 2)})},
            message=("opened to {:.0f} mm and held there".format(gap_mm)
                     if gap_mm is not None else "opened part way")
            + " -- asked for {:.0f} mm, which is what it takes to let go of what was in the "
              "jaws rather than the whole {:.0f} mm of travel".format(
                  float(asked_mm), self.open_span_m * 1000.0))

    def _open_to_now(self, asked_m: float, steps: int) -> Optional[float]:
        """Runs ON the env thread: open a step at a time, stop at the width, then hold."""
        self._gripper = GRIPPER_OPEN
        opening = self._opening()
        for _ in range(max(1, int(steps))):
            if opening is not None and opening >= asked_m:
                break
            self._drive_now(np.zeros(6), 1)
            opening = self._opening()
        self._gripper = GRIPPER_HOLD
        return opening

    # ------------------------------------------------------------------ internals

    def _drive(self, delta6: np.ndarray, steps: int = 1) -> None:
        self._on_env_thread(self._drive_now, delta6, steps)

    def _drive_now(self, delta6: np.ndarray, steps: int = 1) -> None:
        """Issue `steps` OSC actions. The controller normalises by its own output range."""
        robot = self._env.env.robots[0].controller
        scale = np.concatenate([robot.output_max[:3], robot.output_max[3:]])
        action = np.zeros(7)
        action[:6] = np.clip(np.asarray(delta6, dtype=float) / scale, -1.0, 1.0)
        action[6] = self._gripper
        for _ in range(max(1, steps)):
            try:
                self._obs, _, done, _ = self._env.step(action)
            except Exception as exc:
                # robosuite raises rather than returning done when it is stepped past the end,
                # so the flag has to be set from the failure as well as from the flag.
                if "terminated episode" in str(exc):
                    self._episode_over = True
                raise
            self._frame_time += self.control_dt
            self._ticks += 1
            if bool(getattr(getattr(self._env, "env", None), "done", False)):
                self._episode_over = True
                break
            # The `done` that comes back here is the task predicate, NOT the simulator stopping:
            # recorded, never a reason to cut a WAIT or a settle short.
            self._task_done = self._task_done or bool(done)
            # remember how fast the fingers are travelling, so a reading taken mid-close can
            # be refused rather than mistaken for an object
            opening = self._opening()
            if opening is not None:
                self._opening_creep = (abs(opening - self._last_opening)
                                       if self._last_opening is not None else 0.0)
                self._last_opening = opening
