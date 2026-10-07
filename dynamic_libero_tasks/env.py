"""The dynamic LIBERO problem: a tabletop scene whose listed objects ride a conveyor belt or a
carousel until the gripper takes hold of them.

Every physics substep (robosuite calls ``_pre_action`` once per substep) each still-scripted
object gets its pose AND velocity from :mod:`motion` -- the velocity so that contacts see a
moving body when the fingers close. A scripted object is fully kinematic: its height is pinned
at its resting height and it does not collide with the table (collision bit 2, which only the
robot and the other objects answer to). Dragging it over the table instead makes friction tip
it every substep, and the hops grow until the object is launched (measured: a mug on the
carousel reached z = 2.4 m in 13 s). On the first control step where both finger pads
touch an object, that object leaves the script for good and is plain MuJoCo physics from then
on, carrying the belt's velocity into the hand. On a one-pass belt (``loop: false``) an object
that reaches the end falls off it: it is moved off the table, out of every camera, and listed
in ``gone`` -- a target there can no longer be picked, so the episode is lost.

The belt / platform is visual only (no collision); a released object rests on the table.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Dict, Set

import mujoco
import numpy as np
from libero.libero.envs import TASK_MAPPING
from libero.libero.envs.bddl_base_domain import register_problem
from libero.libero.envs.regions import REGION_SAMPLERS

from . import motion as M

_BELT_RGBA = "0.12 0.12 0.13 1"
_CLEAT_RGBA = "0.55 0.55 0.58 1"
_SPOKE_RGBA = "0.85 0.62 0.20 1"
_SKIN = 0.0015                     # the visual's half-thickness above the table top, m


def _el(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def _visual(parent, **attrs):
    """A geom that is seen and never touched."""
    return _el(parent, "geom", contype=0, conaffinity=0, group=1, **attrs)


# LIBERO's `register_problem` returns None, so its module-level names are None: the class
# has to be taken from the registry it was filed in.
Libero_Tabletop_Manipulation = TASK_MAPPING["libero_tabletop_manipulation"]
# the same table, so the same region sampler
REGION_SAMPLERS["libero_dynamic_tabletop_manipulation"] = \
    REGION_SAMPLERS["libero_tabletop_manipulation"]


@register_problem
class Libero_Dynamic_Tabletop_Manipulation(Libero_Tabletop_Manipulation):

    def __init__(self, bddl_file_name, *args, **kwargs):
        self.motion = M.load(M.sidecar(bddl_file_name))
        self._anchors: Dict[str, M.Anchor] = {}
        self._t0 = 0.0
        #: objects that have left the script (grasped at some point this episode)
        self.released: Set[str] = set()
        #: objects a one-pass belt has carried off its end
        self.gone: Set[str] = set()
        super().__init__(bddl_file_name, *args, **kwargs)

    # ------------------------------------------------------------- scene

    def _load_fixtures_in_arena(self, mujoco_arena):
        super()._load_fixtures_in_arena(mujoco_arena)
        m, top = self.motion, self.table_offset[2]
        world = mujoco_arena.worldbody
        if m.kind == "linear":
            mid, u = (m.start + m.end) / 2, m.direction
            yaw = np.arctan2(u[1], u[0])
            quat = " ".join(map(str, M.yaw_quat(yaw)))
            _visual(world, name="belt", type="box", rgba=_BELT_RGBA, quat=quat,
                    pos="{} {} {}".format(mid[0], mid[1], top + _SKIN / 2),
                    size="{} {} {}".format(m.length / 2, m.width / 2, _SKIN / 2))
            for i, _ in enumerate(m.cleats(0.0)):
                body = _el(world, "body", name="cleat{}".format(i), mocap="true", quat=quat)
                _visual(body, type="box", rgba=_CLEAT_RGBA,
                        size="0.006 {} {}".format(m.width / 2, _SKIN))
            for side in (-1, 1):                      # the belt's side rails
                c = mid + side * (m.width / 2 + 0.008) * np.array([-u[1], u[0]])
                _visual(world, type="box", rgba=_CLEAT_RGBA, quat=quat,
                        pos="{} {} {}".format(c[0], c[1], top + 0.006),
                        size="{} 0.006 0.006".format(m.length / 2))
        else:
            c = m.center
            _visual(world, name="platform", type="cylinder", rgba=_BELT_RGBA,
                    pos="{} {} {}".format(c[0], c[1], top + _SKIN / 2),
                    size="{} {}".format(m.radius, _SKIN / 2))
            body = _el(world, "body", name="spokes", mocap="true",
                       pos="{} {} {}".format(c[0], c[1], top + _SKIN))
            for k in range(6):
                q = " ".join(map(str, M.yaw_quat(k * np.pi / 6)))
                _visual(body, type="box", rgba=_SPOKE_RGBA, quat=q,
                        size="{} 0.004 {}".format(m.radius, _SKIN))

    # ------------------------------------------------------------- motion

    def _reset_internal(self):
        super()._reset_internal()
        self._anchors = {}
        self._rest_z: Dict[str, float] = {}
        self.released = set()
        self.gone = set()

    def _geoms(self, *prefixes):
        """Collision geoms (by original bits) of the bodies whose names start with a prefix."""
        m = self.sim.model
        return [g for g in range(m.ngeom) if self._bits[g] != (0, 0)
                and m.body_id2name(m.geom_bodyid[g]).startswith(prefixes)]

    def _anchor(self):
        """Pin each moving object's start to wherever the reset (or an init state) put it, at
        the height where its lowest point meets the table, and take it off the table's
        collision bit."""
        m, d = self.sim.model, self.sim.data
        if getattr(self, "_bits_of", None) is not m:     # a hard reset builds a new model
            self._bits_of, self._bits = m, list(zip(m.geom_contype, m.geom_conaffinity))
        m.geom_contype[:], m.geom_conaffinity[:] = zip(*self._bits)
        for g in self._geoms("robot0_", "gripper0_", *(o.naming_prefix
                                                       for o in self.objects_dict.values())):
            m.geom_conaffinity[g] |= 2
        self._t0 = float(d.time)
        top = self.table_offset[2]
        for name in self.motion.moving:
            q = d.get_joint_qpos(self.objects_dict[name].joints[-1])
            self._anchors[name] = M.Anchor(np.array(q[:2]), np.array(q[3:7]))
            geoms = self._geoms(self.objects_dict[name].naming_prefix)
            self._rest_z[name] = float(q[2]) - (self._lowest(geoms) - top)
            for g in geoms:
                m.geom_contype[g], m.geom_conaffinity[g] = 2, 2

    def _lowest(self, geoms) -> float:
        """The lowest corner of these geoms' bounding boxes, in world z."""
        m, d = self.sim.model, self.sim.data
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        low = np.inf
        for g in geoms:
            corners = m.geom_aabb[g][:3] + signs * m.geom_aabb[g][3:]
            world = d.geom_xpos[g] + corners @ d.geom_xmat[g].reshape(3, 3).T
            low = min(low, float(world[:, 2].min()))
        return low

    def _to_physics(self, name):
        """Hand an object back to MuJoCo: its own collision bits again (plus bit 2, so the
        scripted ones still meet it)."""
        m = self.sim.model
        for g in self._geoms(self.objects_dict[name].naming_prefix):
            m.geom_contype[g] = self._bits[g][0]
            m.geom_conaffinity[g] = self._bits[g][1] | 2

    def _pre_action(self, action, policy_step=False):
        if not self._anchors:
            self._anchor()
        if policy_step:
            self._release_grasped()
        t = float(self.sim.data.time) - self._t0
        data = self.sim.data
        for name, anchor in self._anchors.items():
            if name in self.released or name in self.gone:
                continue
            s = self.motion.state(anchor, t)
            joint = self.objects_dict[name].joints[-1]
            if not s.on_belt:
                self._fall_off(name, joint)
                continue
            data.set_joint_qpos(joint, np.r_[s.xy, self._rest_z[name], s.quat])
            # a free joint's angular velocity is in the body frame
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, s.quat)
            w_local = rot.reshape(3, 3).T @ np.array([0.0, 0.0, s.yaw_rate])
            data.set_joint_qvel(joint, np.r_[s.vel_xy, 0.0, w_local])
        self._move_visuals(t)
        super()._pre_action(action, policy_step=policy_step)

    def _fall_off(self, name, joint):
        """Off the end of a one-pass belt: parked on the floor well outside the room's views."""
        k = len(self.gone)
        self.gone.add(name)
        self._to_physics(name)
        self.sim.data.set_joint_qpos(joint, np.r_[-2.0 + 0.3 * k, 3.0, 0.1, 1.0, 0.0, 0.0, 0.0])
        self.sim.data.set_joint_qvel(joint, np.zeros(6))

    def _release_grasped(self):
        gripper = self.robots[0].gripper
        for name in self.motion.moving:
            if name not in self.released and self._check_grasp(
                    gripper=gripper, object_geoms=self.objects_dict[name]):
                self.released.add(name)
                self._to_physics(name)

    def _move_visuals(self, t: float):
        m, model, data = self.motion, self.sim.model, self.sim.data
        top = self.table_offset[2] + _SKIN
        if m.kind == "linear":
            for i, p in enumerate(m.cleats(t)):
                k = model.body_mocapid[model.body_name2id("cleat{}".format(i))]
                data.mocap_pos[k] = [p[0], p[1], top]
        else:
            k = model.body_mocapid[model.body_name2id("spokes")]
            data.mocap_quat[k] = M.yaw_quat(m.omega * t)
