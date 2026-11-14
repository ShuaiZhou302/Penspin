"""Numpy re-implementation of the frozen PenSpin teacher (705613 best_reward) and of the Isaac Gym observation
pipeline that feeds it, so it can run inside SuperDex (Python 3.12, no torch).

Teacher (ActorCritic with priv_info, point cloud, not student):
  extrin = tanh(cat(env_mlp(priv_n), max_points(point_mlp(pc_n))));  mu = mu(actor_mlp(cat(obs_n, extrin)))
  MLPs use ELU; env_mlp has no final activation; point_mlp and actor_mlp do. Normalizers: (x - mean) / sqrt(var + 1e-5),
  computed in float32 and clamped to [-5, 5]. Action = clamp(mu, -1, 1).
Observation pipeline (allegro_hand_hora.compute_observations / compute_reward, one call per 20 Hz step):
  obs96 = last 3 history frames [noisy_q(16), target(16)] taken BEFORE appending the current frame;
  priv61 = [obj_pos 3, scale, mass, friction, com 3, obj_quat 4, obj_angvel 3, fingertip_pos 12, restitution, tactile 32],
  obj_angvel = axis_angle(q_now * conj(q_prev)) / 0.05; tactile = binary |F| >= 0.05 through latency (p=0.2 keep the
  previous value) and dropout (p=0.1 zero an active channel); point cloud = R(q) pc_local + p.
"""
import numpy as np

ACTION_SCALE = 0.04167
CONTROL_DT = 0.05
FINGERTIPS = ['link_3.0_tip', 'link_15.0_tip', 'link_7.0_tip', 'link_11.0_tip']


def elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


class TeacherNP:
    def __init__(self, npz):
        w = np.load(npz)
        self.w = {k: w[k] for k in w.files}

    def _mlp(self, prefix, x, n_layers, last_act):
        for i in range(n_layers):
            x = x @ self.w[f'model.{prefix}.mlp.{2 * i}.weight'].T + self.w[f'model.{prefix}.mlp.{2 * i}.bias']
            if i < n_layers - 1 or last_act:
                x = elu(x)
        return x

    def _norm(self, name, x):
        m = self.w[f'{name}.running_mean'].astype(np.float32)
        v = self.w[f'{name}.running_var'].astype(np.float32)
        return np.clip((x - m) / np.sqrt(v + np.float32(1e-5)), -5.0, 5.0).astype(np.float32)

    def act(self, obs96, priv61, pc):
        """Batched: obs96 (B,96), priv61 (B,61), pc (B,100,3) raw. Returns clamped actions (B,16)."""
        obs_n = self._norm('running_mean_std', np.asarray(obs96, np.float32))
        priv_n = self._norm('priv_mean_std', np.asarray(priv61, np.float32))
        pc = np.asarray(pc, np.float32)
        pc_n = self._norm('point_cloud_mean_std', pc.reshape(-1, 3)).reshape(pc.shape)
        extrin = self._mlp('env_mlp', priv_n, 3, False)
        pcs = self._mlp('point_mlp', pc_n, 3, True).max(axis=1)
        x = np.concatenate([obs_n, np.tanh(np.concatenate([extrin, pcs], -1))], -1)
        x = self._mlp('actor_mlp', x, 3, True)
        mu = x @ self.w['model.mu.weight'].T + self.w['model.mu.bias']
        return np.clip(mu, -1, 1)


def quat_mul(a, b):  # xyzw
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def quat_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_apply(q, v):
    """Rotate v (3,) or (N, 3) by xyzw quaternion q."""
    xyz, w = q[:3], q[3]
    t = 2 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def quat_to_axis_angle(q):
    """PyTorch3D quaternion_to_axis_angle for real-last quaternions (as in allegro_hand_hora.quat_to_axis_angle)."""
    xyz, w = q[:3], q[3]
    n = np.linalg.norm(xyz)
    half = np.arctan2(n, w)
    angle = 2 * half
    eps = 1e-6
    s = np.sin(half) / angle if abs(angle) > eps else 0.5 - angle * angle / 48
    return xyz / s


class TeacherObs:
    """Per-episode Isaac Gym observation state for one env, continuing from a recorded step-0 observation."""

    def __init__(self, init_pose, obs0, priv0, pc_local, scale, mass, friction, restitution, rng,
                 obs_noise_t=0.01, obs_noise_e=0.02, latency=0.2, dropout=0.1, contact_thresh=0.05):
        self.frames = [np.concatenate([init_pose, init_pose])] * 3  # history refilled with the grasp pose at reset
        # Student inputs: last 30 frames of the lag history INCLUDING the newest frame; at reset all frames are
        # [grasp pose, grasp pose] with zero tactile.
        self.proprio_hist = np.tile(np.concatenate([init_pose, init_pose]), (30, 1)).astype(np.float32)
        self.tactile_hist = np.zeros((30, 32), np.float32)
        self.obs = np.asarray(obs0, np.float32)
        self.priv = np.asarray(priv0, np.float32)
        self.const = dict(scale=scale, mass=mass, friction=friction, restitution=restitution)
        self.pc_local = np.asarray(pc_local)
        self.rng = rng
        self.noise_t, self.latency, self.dropout, self.thresh = obs_noise_t, latency, dropout, contact_thresh
        self.noise_e = rng.normal(0, obs_noise_e, 16) if obs_noise_e > 0 else np.zeros(16)
        self.last_contacts = np.zeros(32)

    def update(self, q, target, contact_norm, ft_pos, obj_pos, obj_quat, obj_quat_prev, noise_t=None, sensed=None):
        """sensed: optional override of the tactile model output (port checks against recorded Isaac Gym values)."""
        noisy_q = q + (self.rng.normal(0, self.noise_t, 16) if noise_t is None else noise_t) + self.noise_e
        contacts = (np.asarray(contact_norm) >= self.thresh).astype(float)
        lat = (self.rng.random(32) < self.latency).astype(float)
        self.last_contacts = self.last_contacts * lat + contacts * (1 - lat)
        mask = (self.rng.random(32) >= self.dropout).astype(float)
        if sensed is None:
            sensed = np.where(self.last_contacts > 0.1, mask * self.last_contacts, self.last_contacts)
        self.obs = np.concatenate(self.frames[-3:]).astype(np.float32)
        self.frames = self.frames[-2:] + [np.concatenate([noisy_q, target])]
        self.proprio_hist = np.concatenate([self.proprio_hist[1:], self.frames[-1][None]]).astype(np.float32)
        self.tactile_hist = np.concatenate([self.tactile_hist[1:], sensed[None]]).astype(np.float32)
        angvel = quat_to_axis_angle(quat_mul(obj_quat, quat_conj(obj_quat_prev))) / CONTROL_DT
        c = self.const
        self.priv = np.concatenate([obj_pos, [c['scale'], c['mass'], c['friction']], np.zeros(3), obj_quat, angvel,
                                    np.ravel(ft_pos), [c['restitution']], sensed]).astype(np.float32)
        pc = quat_apply(np.asarray(obj_quat, float), self.pc_local) + obj_pos
        return self.obs, self.priv, pc.astype(np.float32), angvel
