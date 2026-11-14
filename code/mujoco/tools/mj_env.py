"""PenSpin Allegro + pencil in MuJoCo (official `mujoco` package), same interface as tools/sdx_env.PenSpinSDX.

Isaac Gym reference (penspin/tasks/allegro_hand_hora.py, configs/task/AllegroHandHora.yaml), as in sdx_env.py:
  sim dt 0.005 s, controlFrequencyInv 10 (20 Hz), torque control tau = p*(target - q) - d*(q - q_prev)/dt, p=3, d=0.1,
  clipped to +-0.7 N m, D term 0 at the first substep of a control step; joint friction 0.01, armature 0.001;
  hand fixed at p=(0,0,0.5), r=Rot_y(-pi/2)*Rot_x(pi/2), hand gravity disabled (here: gravcomp=1);
  hand asset = Isaac Gym's own URDF (assets/round_tip/allegro_hand_right_fsr_round_dense.urdf, 63 boxes + 4 tip meshes),
  fixed joints kept as separate bodies (collapse_fixed_joints=False); hand actor collision filter -1 = no self-collision;
  pencil = cylinder r=0.04 x scale, IG length (0.40/0.44/0.48/0.52 x scale), mass from the record, inertia from geometry;
  tactile: 32 sensor links, |pen-to-link contact force| (world vector sum per link) after the last substep; the IG index -1
  channel (link_13.0_fsr, absent from the URDF) = |total contact force on the pen|.
Options (environment variables, all fixed per run):
  MJ_SUBSTEPS=n   split each 5 ms physics step into n MuJoCo steps, torque held (zero-order hold)
  MJ_CONTACT="solref=a,b;solimp=a,b,c"   geom solref/solimp on every hand geom and the pen
  MJ_LIMIT="solreflimit=a,b;solimplimit=a,b,c"   on the 16 hinge joints
  MJ_OPT="integrator=implicitfast;cone=elliptic;condim=3;noslip=0"
Audits: min signed pen contact distance (negative = penetration) and max joint-limit excess, both over every MuJoCo step.
"""
import math
import os
import re
import tempfile
from pathlib import Path
import numpy as np
import mujoco

IG_DT, CONTROL_INV, P_GAIN, D_GAIN, TORQUE_LIMIT = 0.005, 10, 3.0, 0.1, 0.7
JOINT_FRICTION, ARMATURE, CONTACT_THRESH = 0.01, 0.001, 0.05
IG_LOWER = [-0.5585, -0.27924, -0.27924, -0.27924, 0.27924, -0.331603, -0.27924, -0.27924,
            -0.5585, -0.27924, -0.27924, -0.27924, -0.5585, -0.27924, -0.27924, -0.27924]
IG_UPPER = [0.5585, 1.727825, 1.727825, 1.727825, 1.57075, 1.1518833, 1.727825, 1.76273055,
            0.5585, 1.727825, 1.727825, 1.727825, 0.5585, 1.727825, 1.727825, 1.727825]
SENSOR_NAMES = ["link_1.0_fsr", "link_2.0_fsr", "link_5.0_fsr", "link_6.0_fsr", "link_9.0_fsr", "link_10.0_fsr",
                "link_14.0_fsr", "link_15.0_fsr", "link_0.0_fsr", "link_4.0_fsr", "link_8.0_fsr", "link_13.0_fsr"]
SENSOR_NAMES += [f"link_{t}_tip_fsr_{i}" for t in ('3.0', '15.0', '7.0', '11.0') for i in (2, 5, 8, 11, 13)]
PEN_SENSOR = "link_13.0_fsr"
FINGERTIPS = ['link_3.0_tip', 'link_15.0_tip', 'link_7.0_tip', 'link_11.0_tip']
IG_URDF = '<PROJECT_ROOT>/penspin-superdex-port-20261028/isaac/assets/round_tip/allegro_hand_right_fsr_round_dense.urdf'


def _kv(name):
    out = {}
    for part in [p for p in os.environ.get(name, '').split(';') if p]:
        k, v = part.split('=', 1)
        out[k] = [float(x) for x in v.split(',')] if k not in ('integrator', 'cone') else v
        if k.startswith('solimp') and len(out[k]) == 3:
            out[k] += [0.5, 2.0]  # MuJoCo defaults for midpoint and power
    return out


SUBSTEPS = int(os.environ.get('MJ_SUBSTEPS', '1'))
CONTACT_OVERRIDE = _kv('MJ_CONTACT')
JOINT_OVERRIDE = _kv('MJ_LIMIT')
OPT = {'integrator': 'implicitfast', 'cone': 'pyramidal', 'condim': [3.0], 'noslip': [0.0], **_kv('MJ_OPT')}


def quat_mul(a, b):  # xyzw
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return [aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz]


def axis_angle(axis, ang):
    s = math.sin(ang / 2)
    return [axis[0] * s, axis[1] * s, axis[2] * s, math.cos(ang / 2)]


HAND_POS = [0.0, 0.0, 0.5]
HAND_QUAT = quat_mul(axis_angle([0, 1, 0], -math.pi / 2), axis_angle([1, 0, 0], math.pi / 2))  # xyzw
_URDF_CACHE = {}


def _urdf_for_mujoco(src):
    """Copy of the IG URDF with a <mujoco><compiler/> block: keep fixed-joint links as bodies, collision meshes dir."""
    if src in _URDF_CACHE:
        return _URDF_CACHE[src]
    s = Path(src).read_text()
    meshdir = str(Path(src).parent / 'meshes' / 'collision')
    s = re.sub(r'filename="[^"]*/([^"/]+)"', r'filename="\1"', s)

    def first_inertial(m):  # link_3.0 has two <inertial> blocks; urdfdom (Isaac Gym) reads the first one
        body = m.group(2); blocks = re.findall(r'<inertial>.*?</inertial>', body, re.S)
        for b in blocks[1:]:
            body = body.replace(b, '', 1)
        return m.group(1) + body + '</link>'
    s = re.sub(r'(<link name="[^"]+">)(.*?)</link>', first_inertial, s, flags=re.S)
    i = s.find('>', s.find('<robot'))
    s = s[:i + 1] + f'\n  <mujoco><compiler meshdir="{meshdir}/" fusestatic="false" discardvisual="true" balanceinertia="true"/></mujoco>' + s[i + 1:]
    fd, path = tempfile.mkstemp(suffix='.urdf'); os.write(fd, s.encode()); os.close(fd)
    _URDF_CACHE[src] = path
    return path


class PenSpinMJ:
    def __init__(self, urdf, ig_dof_names, pen_scale, pen_mass, friction=1.0, dt=IG_DT, pen_inertia=None, pen_length=None, term_length=None):
        assert pen_inertia is None, 'only physical (geometry) inertia is implemented'
        self.dt = dt
        spec = mujoco.MjSpec.from_file(_urdf_for_mujoco(IG_URDF if urdf is None or 'superdex' in str(urdf) else urdf))
        spec.option.timestep = dt / SUBSTEPS
        spec.option.integrator = getattr(mujoco.mjtIntegrator, 'mjINT_' + OPT['integrator'].upper())
        spec.option.cone = getattr(mujoco.mjtCone, 'mjCONE_' + OPT['cone'].upper())
        spec.option.noslip_iterations = int(OPT['noslip'][0])
        spec.option.gravity = [0, 0, -9.81]
        root = spec.worldbody.first_body()
        root.pos = HAND_POS
        qx, qy, qz, qw = HAND_QUAT; root.quat = [qw, qx, qy, qz]
        for b in spec.bodies:
            if b.name != 'world':
                b.gravcomp = 1.0
        for g in spec.geoms:
            g.contype, g.conaffinity, g.condim = 1, 2, int(OPT['condim'][0])
            g.friction = [friction, 0.005, 0.0001]
            if 'solref' in CONTACT_OVERRIDE: g.solref = CONTACT_OVERRIDE['solref']
            if 'solimp' in CONTACT_OVERRIDE: g.solimp = CONTACT_OVERRIDE['solimp']
        hinges = {j.name: j for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_HINGE}
        assert sorted(hinges) == sorted(ig_dof_names), (sorted(hinges), ig_dof_names)
        for name, lo, hi in zip(ig_dof_names, IG_LOWER, IG_UPPER):
            j = hinges[name]
            j.range = [lo, hi]; j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
            j.armature, j.frictionloss = ARMATURE, JOINT_FRICTION
            j.damping = np.zeros(3)  # MuJoCo 4 joint damping is a 3-vector; IG dof damping is 0 in effort mode
            if 'solreflimit' in JOINT_OVERRIDE: j.solref_limit = JOINT_OVERRIDE['solreflimit']
            if 'solimplimit' in JOINT_OVERRIDE: j.solimp_limit = JOINT_OVERRIDE['solimplimit']
        pen_length = 0.4 * pen_scale if pen_length is None else pen_length
        pb = spec.worldbody.add_body(name='pen', pos=[0, 0, 1])
        pb.add_freejoint(name='pen_free')
        pg = pb.add_geom(name='pen_geom', type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.04 * pen_scale, pen_length / 2, 0],
                         mass=pen_mass, contype=2, conaffinity=1, condim=int(OPT['condim'][0]), friction=[friction, 0.005, 0.0001])
        if 'solref' in CONTACT_OVERRIDE: pg.solref = CONTACT_OVERRIDE['solref']
        if 'solimp' in CONTACT_OVERRIDE: pg.solimp = CONTACT_OVERRIDE['solimp']
        self.model = spec.compile(); self.data = mujoco.MjData(self.model)
        m = self.model
        self.jnt = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ig_dof_names])
        self.qadr = m.jnt_qposadr[self.jnt]; self.vadr = m.jnt_dofadr[self.jnt]
        self.pen_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'pen')
        self.pen_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'pen_geom')
        pj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'pen_free'); self.pq = m.jnt_qposadr[pj]; self.pv = m.jnt_dofadr[pj]
        bid = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(m.nbody)}
        missing = [n for n in SENSOR_NAMES if n not in bid]
        assert missing == [PEN_SENSOR], ('unexpected missing sensor links', missing)
        self.sensor_of_body = {bid[n]: i for i, n in enumerate(SENSOR_NAMES) if n != PEN_SENSOR}
        self.pen_sensor_index = SENSOR_NAMES.index(PEN_SENSOR)
        self.tips = [bid[n] for n in FINGERTIPS]
        self.pen_length = pen_length if term_length is None else term_length
        self.lo, self.hi = np.asarray(IG_LOWER), np.asarray(IG_UPPER)
        self.q_prev = None; self.min_dist = float('inf'); self.max_jex = -float('inf')
        self._f6 = np.zeros(6)

    def close(self):
        pass

    def q(self):
        return self.data.qpos[self.qadr].copy()

    def pen_pose(self):
        p = self.data.xpos[self.pen_body]; w, x, y, z = self.data.xquat[self.pen_body]
        return np.array([p[0], p[1], p[2], x, y, z, w])

    def fingertips(self):
        return self.data.xpos[self.tips].copy()

    def reset(self, q_ig, pen_pos, pen_quat_xyzw, qd_ig=None, pen_linvel=(0, 0, 0), pen_angvel=(0, 0, 0)):
        d = self.data; mujoco.mj_resetData(self.model, d)
        d.qpos[self.qadr] = q_ig
        d.qvel[self.vadr] = np.zeros(16) if qd_ig is None else qd_ig
        x, y, z, w = pen_quat_xyzw
        d.qpos[self.pq:self.pq + 3] = pen_pos; d.qpos[self.pq + 3:self.pq + 7] = [w, x, y, z]
        R = np.zeros(9); mujoco.mju_quat2Mat(R, np.array([w, x, y, z], float)); R = R.reshape(3, 3)
        d.qvel[self.pv:self.pv + 3] = pen_linvel
        d.qvel[self.pv + 3:self.pv + 6] = R.T @ np.asarray(pen_angvel, float)  # free-joint angular velocity is body-frame
        mujoco.mj_forward(self.model, d)
        self.q_prev = self.q(); self.min_dist = float('inf'); self.max_jex = -float('inf')
        self.k = 0; self.min_dist_late = float('inf')  # late = after the first control step (inherited IG overlap excluded)

    def physics_step(self, target_ig):
        q = self.q()
        tau = P_GAIN * (target_ig - q) - D_GAIN * (q - self.q_prev) / self.dt
        tau = np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT)
        self.q_prev = q
        d = self.data
        for _ in range(SUBSTEPS):
            d.qfrc_applied[:] = 0; d.qfrc_applied[self.vadr] = tau
            mujoco.mj_step(self.model, d)
            for c in d.contact[:d.ncon]:
                if c.geom1 == self.pen_geom or c.geom2 == self.pen_geom:
                    self.min_dist = min(self.min_dist, float(c.dist))
                    if self.k >= CONTROL_INV:
                        self.min_dist_late = min(self.min_dist_late, float(c.dist))
            qs = d.qpos[self.qadr]
            self.max_jex = max(self.max_jex, float(np.max(qs - self.hi)), float(np.max(self.lo - qs)))
        self.k += 1
        return tau

    def control_step(self, target_ig):
        target_ig = np.clip(np.asarray(target_ig, float), IG_LOWER, IG_UPPER)
        self.q_prev = self.q()
        for _ in range(CONTROL_INV):
            self.physics_step(target_ig)
        return self.contact_norms()

    def contact_norms(self):
        m, d = self.model, self.data
        f = np.zeros((len(SENSOR_NAMES), 3)); tot = np.zeros(3)
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 != self.pen_geom and c.geom2 != self.pen_geom:
                continue
            mujoco.mj_contactForce(m, d, i, self._f6)
            fw = c.frame.reshape(3, 3).T @ self._f6[:3]  # contact frame -> world, MuJoCo convention geom1 -> geom2
            fw = fw if c.geom2 == self.pen_geom else -fw  # force on the pen; the link gets -fw (same norm)
            other = c.geom2 if c.geom1 == self.pen_geom else c.geom1
            tot += fw
            k = self.sensor_of_body.get(int(m.geom_bodyid[other]))
            if k is not None:
                f[k] += fw
        f[self.pen_sensor_index] = tot
        return np.linalg.norm(f, axis=1)

    def pen_ends_z(self):
        p = self.pen_pose(); x, y, z, w = p[3:]
        axis_z = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
        e1, e2 = p[:3] + axis_z * self.pen_length / 2, p[:3] - axis_z * self.pen_length / 2
        return min(e1[2], e2[2]), max(e1[2], e2[2])
