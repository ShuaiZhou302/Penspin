"""PenSpin Allegro + pencil in SuperDex, matching the Isaac Gym task's low-level control.

Isaac Gym reference (penspin/tasks/allegro_hand_hora.py, configs/task/AllegroHandHora.yaml):
  sim dt 0.005 s, controlFrequencyInv 10 (20 Hz policy), torque control
  tau = p*(target - q) - d*(q - q_prev)/dt, p=3, d=0.1, clipped to +-0.7 N m, joint friction 0.01, armature 0.001,
  hand base fixed at p=(0,0,0.5), r=Rot_y(-pi/2)*Rot_x(pi/2), hand gravity disabled,
  pencil = cylinder r=0.04 l=0.4 (URDF) scaled by obj_scale, mass from randomization,
  tactile: 32 sensor links, binary contact if |net contact force| >= 0.05 N (latency/dropout applied afterwards).
Joint values are exchanged in Isaac Gym DOF order (IG_DOF_NAMES).
"""
import math
import numpy as np
import superdex.physics as sdp
import superdex.robotics as sdr

REAL = np.float64 if sdp.uses_double_precision() else np.float32  # dtype of Span[float] arguments

import os
# Compliance sweep (penspin-superdex-compliance-20261107): SDX_CONTACT="field=value,..." overrides SuperDex ContactParams
# fields on every hand link and on the pen. Unset = SuperDex defaults, identical to penspin-superdex-penlen-20261031.
CONTACT_OVERRIDE = {k: float(v) for k, v in (kv.split('=') for kv in os.environ.get('SDX_CONTACT', '').split(',') if kv)}

# Admissibility search (penspin-superdex-admissible-20261107): SDX_JOINT="limit_stiffness=..,limit_damping=.." on the 16
# revolute joints; SDX_SUBSTEPS=n splits each 5 ms Isaac Gym physics step into n SuperDex steps with the torque held
# (zero-order hold, so the PD law and its 5 ms update stay identical to Isaac Gym). Unset = previous behaviour.
JOINT_OVERRIDE = {k: float(v) for k, v in (kv.split('=') for kv in os.environ.get('SDX_JOINT', '').split(',') if kv)}
SUBSTEPS = int(os.environ.get('SDX_SUBSTEPS', '1'))

def contact_params(friction):
    cp = sdp.ContactParams(); cp.coulomb_friction_coefficient = friction
    for k, v in CONTACT_OVERRIDE.items():
        assert hasattr(cp, k) and not callable(getattr(cp, k)), k
        setattr(cp, k, v)
    return cp


IG_DT, CONTROL_INV, P_GAIN, D_GAIN, TORQUE_LIMIT = 0.005, 10, 3.0, 0.1, 0.7
JOINT_FRICTION, ARMATURE, CONTACT_THRESH = 0.01, 0.001, 0.05
# Isaac Gym calibrated limits, in IG DOF order (allegro_hand_hora._parse_hand_dof_props).
IG_LOWER = [-0.5585, -0.27924, -0.27924, -0.27924, 0.27924, -0.331603, -0.27924, -0.27924,
            -0.5585, -0.27924, -0.27924, -0.27924, -0.5585, -0.27924, -0.27924, -0.27924]
IG_UPPER = [0.5585, 1.727825, 1.727825, 1.727825, 1.57075, 1.1518833, 1.727825, 1.76273055,
            0.5585, 1.727825, 1.727825, 1.727825, 0.5585, 1.727825, 1.727825, 1.727825]
SENSOR_NAMES = ["link_1.0_fsr", "link_2.0_fsr", "link_5.0_fsr", "link_6.0_fsr", "link_9.0_fsr", "link_10.0_fsr",
                "link_14.0_fsr", "link_15.0_fsr", "link_0.0_fsr", "link_4.0_fsr", "link_8.0_fsr", "link_13.0_fsr"]
SENSOR_NAMES += [f"link_{t}_tip_fsr_{i}" for t in ('3.0', '15.0', '7.0', '11.0') for i in (2, 5, 8, 11, 13)]
PEN_SENSOR = "link_13.0_fsr"


def quat_mul(a, b):  # xyzw
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return [aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz]


def axis_angle(axis, ang):
    s = math.sin(ang / 2)
    return [axis[0] * s, axis[1] * s, axis[2] * s, math.cos(ang / 2)]


HAND_POS = [0.0, 0.0, 0.5]
HAND_QUAT = quat_mul(axis_angle([0, 1, 0], -math.pi / 2), axis_angle([1, 0, 0], math.pi / 2))


def cylinder_mesh(radius, length, n=48):
    v = [[0, 0, -length / 2], [0, 0, length / 2]]
    for z in (-length / 2, length / 2):
        v += [[radius * math.cos(2 * math.pi * k / n), radius * math.sin(2 * math.pi * k / n), z] for k in range(n)]
    f = []
    for k in range(n):
        a, b = 2 + k, 2 + (k + 1) % n          # bottom ring
        c, d = 2 + n + k, 2 + n + (k + 1) % n  # top ring
        f += [[0, b, a], [1, c, d], [a, b, d], [a, d, c]]
    return np.asarray(v, REAL), np.asarray(f, np.int32)


def transform(pos, quat_xyzw):
    return sdp.TransformRT(sdp.Quaternion(*quat_xyzw), sdp.Real3(*pos))  # sdp.Quaternion is (x, y, z, w)


class PenSpinSDX:
    def __init__(self, urdf, ig_dof_names, pen_scale, pen_mass, friction=1.0, dt=IG_DT, pen_inertia=None, pen_length=None, term_length=None):
        sdp.initialize(num_worker_threads=0) if not sdp.is_initialized() else None
        self.dt = dt
        self.scene = sdp.create_scene("penspin")
        self.scene.set_gravity([0, 0, -9.81])
        pf = sdr.load_bot_prefab_from_urdf_file(urdf)
        pf.joints[0].type = sdp.ArticulatedJointType.HARD
        self.friction = friction
        for i in range(len(pf.links)):
            pf.links[i].has_gravity = False
            cp = contact_params(friction)
            pf.links[i].contact = cp
        joints = {pf.joints[i].name: pf.joints[i] for i in range(len(pf.joints))}
        for name, lo, hi in zip(ig_dof_names, IG_LOWER, IG_UPPER):
            j = joints[name]; axis = [float(x) for x in j.axis]
            j.min_limit = [lo * a for a in axis]; j.max_limit = [hi * a for a in axis]
            j.effort_limit = TORQUE_LIMIT
            for k, v in JOINT_OVERRIDE.items():
                assert hasattr(j, k), k
                setattr(j, k, v)
        self.ctx = sdr.create_context()
        self.bot = sdr.create_bot(self.scene, pf, self.ctx)
        self.hand = self.bot.get_articulated_actor()
        self.hand.set_root_transform(transform(HAND_POS, HAND_QUAT))
        self.ndof = self.hand.get_num_dofs()
        assert self.ndof == 16, self.ndof
        rev = [pf.joints[i].name for i in range(len(pf.joints)) if pf.joints[i].type == sdp.ArticulatedJointType.REVOLUTE]
        assert sorted(rev) == sorted(ig_dof_names), (rev, ig_dof_names)
        self.sdx_from_ig = np.array([ig_dof_names.index(n) for n in rev])  # sdx[k] = ig[sdx_from_ig[k]]
        self.dof_idx = np.arange(16, dtype=np.int32)
        fr = self.hand.get_articulated_joint_friction_params()
        fr = [fr[i] for i in range(len(fr))]
        for p in fr:
            p.coulomb = JOINT_FRICTION
        self.hand.set_articulated_joint_friction_params(fr)
        self.hand.set_articulated_joint_inertia_params(np.full(len(fr), ARMATURE, REAL))
        links = {}
        for h in self.hand.get_nested_link_actors():
            a = self.scene.get_actor(h); links[a.get_name().split('/')[-1]] = a
        self.links = links
        # 'link_13.0_fsr' is not in the URDF: Isaac Gym's handle lookup returns -1 and contacts[:, -1] reads the last
        # rigid body of the env, i.e. the object's net contact force. Reproduced here as the pen actor.
        missing = [n for n in SENSOR_NAMES if n not in links]
        assert missing == [PEN_SENSOR], ('unexpected missing sensor links', missing)
        pen_length = 0.4 * pen_scale if pen_length is None else pen_length  # Isaac Gym assets: 0.40/0.44/0.48/0.52 x scale
        v, f = cylinder_mesh(0.04 * pen_scale, pen_length)
        shape = sdp.create_tri_mesh_shape(v.reshape(-1), f.reshape(-1))
        cp = contact_params(friction)
        # pen_inertia: body-frame 3x3 inertia [kg m^2]; None = computed from the mesh (physical solid cylinder).
        extra = {}
        if pen_inertia is not None:
            I = np.asarray(pen_inertia, float).reshape(3, 3)
            # sdp.Real6 = (Ixx, Iyy, Izz, Ixy, Ixz, Iyz); Isaac Gym's pen inertia is diagonal (isotropic).
            extra = {'moment_of_inertia': sdp.Real6(I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]),
                     'center_of_mass': sdp.Real3(0.0, 0.0, 0.0)}
        self.pen = self.scene.create_rigid_actor(name="pen", shape=shape, mass=pen_mass, contact=cp, **extra)
        self.pen_length = pen_length if term_length is None else term_length  # length used by the end-point termination
        self.sensors = [self.pen if n == PEN_SENSOR else links[n] for n in SENSOR_NAMES]
        self.scene.create_rigid_actor(name="ground", shape=sdp.create_plane_shape(normal=[0, 0, 1], distance=0), is_static=True)
        # Base-welded links (palm pads) are static colliders without contact sample points, so per-link forces are
        # read from the pen's side: sum of pen contact-point forces grouped by the other actor.
        self.pen.register_query(sdp.QueryType.CONTACT_POINTS)
        self.pen.register_query(sdp.QueryType.TOTAL_CONTACT_FORCE)
        # Actors are keyed by ActorHandle.value: str(handle) is the Python wrapper's repr (memory address), which differs
        # between wrapper objects for the same actor.
        self.pen_key = self.pen.get_handle().value
        self.sensor_index = {a.get_handle().value: i for i, a in enumerate(self.sensors) if a is not self.pen}
        assert len(self.sensor_index) == len(SENSOR_NAMES) - 1, 'sensor links do not have distinct handles'
        self.pen_sensor_index = SENSOR_NAMES.index(PEN_SENSOR)
        self.q_prev = None

    def close(self):
        sdr.destroy_bot(self.scene, self.bot)
        sdp.destroy_scene(self.scene)

    def q(self):
        out = np.zeros(16, REAL); self.hand.get_dof_values(np.zeros(0, np.int32), out)
        ig = np.zeros(16); ig[self.sdx_from_ig] = out
        return ig

    def pen_pose(self):
        t = self.pen.get_root_transform(); r = t.rotation; p = t.translation
        return np.array([p[0], p[1], p[2], r[0], r[1], r[2], r[3]])  # sdp.Quaternion indexes as (x, y, z, w)

    def reset(self, q_ig, pen_pos, pen_quat_xyzw, qd_ig=None, pen_linvel=(0, 0, 0), pen_angvel=(0, 0, 0)):
        self.hand.set_articulated_pose_from_joints(np.asarray(q_ig, REAL)[self.sdx_from_ig])
        qd = np.zeros(16) if qd_ig is None else np.asarray(qd_ig)
        self.hand.set_articulated_joint_velocities(qd[self.sdx_from_ig].astype(REAL))
        self.pen.set_root_transform(transform(pen_pos, pen_quat_xyzw))
        self.pen.set_velocity([float(x) for x in pen_linvel], [float(x) for x in pen_angvel])
        self.q_prev = self.q()
        self.min_dist = float('inf')
        self.max_jex = -float('inf')

    def physics_step(self, target_ig):
        q = self.q()
        tau = P_GAIN * (target_ig - q) - D_GAIN * (q - self.q_prev) / self.dt
        tau = np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT)
        self.q_prev = q
        for _ in range(SUBSTEPS):
            self.hand.set_external_forces_on_dofs(self.dof_idx, tau[self.sdx_from_ig].astype(REAL))
            self.scene.step(self.dt / SUBSTEPS)
            for c in self.pen.get_contact_points_world():
                self.min_dist = min(self.min_dist, float(c.distance))
            qs = self.q()
            self.max_jex = max(self.max_jex, float(np.max(qs - np.asarray(IG_UPPER))), float(np.max(np.asarray(IG_LOWER) - qs)))
        return tau

    def control_step(self, target_ig):
        """One 20 Hz control step (10 physics steps). Returns contact force norms (32) after the last substep."""
        target_ig = np.clip(np.asarray(target_ig, float), IG_LOWER, IG_UPPER)
        # Isaac Gym's finite-difference velocity is 0 at the first substep of each control step (its cached q was
        # refreshed after the previous simulate), so the D term only acts from the second substep on.
        self.q_prev = self.q()
        for _ in range(CONTROL_INV):
            self.physics_step(target_ig)
        return self.contact_norms()

    def contact_norms(self):
        """|contact force| per sensor (32): pen-to-link force for hand pads, total pen contact force for the
        IG index -1 channel."""
        f = np.zeros((len(SENSOR_NAMES), 3))
        for p in self.pen.get_contact_points_world():
            a, b = p.actor_a.value, p.actor_b.value
            other = b if a == self.pen_key else a
            i = self.sensor_index.get(other)
            if i is not None:
                f[i] += [p.force[0], p.force[1], p.force[2]]
        t = self.pen.get_contact_force_world()
        f[self.pen_sensor_index] = [t[0], t[1], t[2]]
        return np.linalg.norm(f, axis=1)

    def pen_ends_z(self):
        p = self.pen_pose(); x, y, z, w = p[3:]
        axis_z = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
        e1, e2 = p[:3] + axis_z * self.pen_length / 2, p[:3] - axis_z * self.pen_length / 2
        return min(e1[2], e2[2]), max(e1[2], e2[2])
