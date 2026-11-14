"""Pen end-point estimate error of a tactile student, against the clean pose (no box noise), in SuperDex closed loop and
in the student's own Isaac Gym record. Same loop and rng order as closed_loop.py penlen=ig, so rewards must reproduce.
usage: diag_pose.py <record.npz> <teacher.npz> <out.json> <ckpt> <seed> [ref.json] [friction=1.0]"""
import json
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sdx_env import PenSpinSDX, IG_LOWER, IG_UPPER  # noqa: E402
from teacher_np import TeacherObs, ACTION_SCALE, FINGERTIPS  # noqa: E402
from student_policy import StudentPolicy  # noqa: E402
from closed_loop import URDF, RESET_Z, END_Z  # noqa: E402

HAND = np.array([0, 0, 0.5])
MODE = 'proprio_tactile_ends_est'


def true_ends(pose, length):
    p = pose[:3]; x, y, z, w = pose[3:7]
    axis = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    return np.concatenate([p - axis * length / 2 - HAND, p + axis * length / 2 - HAND])


def err_cm(pred, truth):
    a = np.abs(pred - truth).mean(); b = np.abs(pred - np.concatenate([truth[3:], truth[:3]])).mean()
    return 100 * float(min(a, b))


@torch.no_grad()
def predict(sp, proprio, tactile, obs):
    inp = {'obs': sp.rms(torch.as_tensor(np.asarray(obs, np.float32))[None]),
           'proprio_hist': torch.as_tensor(np.asarray(proprio, np.float32))[None], 'obj_ends': torch.zeros(1, 3, 6),
           'tactile_hist': torch.as_tensor(np.asarray(tactile, np.float32))[None]}
    return sp.model.act_inference(inp)[2].reshape(3, 6)[-1].numpy()


def main(record, teacher, out, ckpt, seed, ref=None, friction='1.0'):
    r = dict(np.load(record)); names = [str(x) for x in r['dof_names']]
    n = len(r['lengths']); T = r['target'].shape[1]; seed = int(seed); friction = float(friction)
    rng = np.random.default_rng(seed); sp = StudentPolicy(ckpt, MODE)
    scale = r['obj_scale'].astype(float); pc = r['obj_point_cloud_local']
    lens = np.round((pc[..., 2].max(1) - pc[..., 2].min(1)) / scale, 2) * scale; term = 0.52 * scale[-1]
    sdx_err, sdx_t, rewards = [], [], []
    for e in range(n):
        env = PenSpinSDX(URDF, names, pen_scale=float(scale[e]), pen_mass=float(r['obj_mass'][e]), friction=friction,
                         pen_length=float(lens[e]), term_length=term)
        o = r['init_obj'][e]
        env.reset(r['init_q'][e], o[:3], o[3:7], r['init_qd'][e], o[7:10], o[10:13])
        init_pose = r['in_obs'][e, 1, :16] if r['lengths'][e] > 1 else r['init_q'][e]
        ob = TeacherObs(init_pose, r['in_obs'][e, 0], r['in_priv'][e, 0], pc[e], float(scale[e]), float(r['obj_mass'][e]),
                        friction, float(r['obj_restitution'][e]), rng, obs_noise_t=0.01, obs_noise_e=0.02)
        obs = r['in_obs'][e, 0]; target = r['init_target'][e].astype(float); prev = o[3:7]; pose = o[:7]; rot = 0.0
        for t in range(T):
            a = sp.act(obs, ob.proprio_hist, ob.tactile_hist)
            sdx_err.append(err_cm(predict(sp, ob.proprio_hist, ob.tactile_hist, obs), true_ends(pose, term))); sdx_t.append(t)
            target = np.clip(target + ACTION_SCALE * a, IG_LOWER, IG_UPPER)
            cn = env.control_step(target); pose = env.pen_pose()
            ft = np.array([[env.links[k].get_root_transform().translation[i] for i in range(3)] for k in FINGERTIPS])
            obs, _, _, angvel = ob.update(env.q(), target, cn, ft, pose[:3], pose[3:7], prev)
            prev = pose[3:7]; rot += float(np.clip(angvel[2], -0.5, 0.5))
            zmin, zmax = env.pen_ends_z()
            if pose[2] < RESET_Z or zmin < END_Z[0] or zmax > END_Z[1]:
                break
        env.close(); rewards.append(rot)
    sdx_err, sdx_t = np.array(sdx_err), np.array(sdx_t)
    res = {'record': record, 'sdx_rot': float(np.mean(rewards)), 'sdx_err_cm': float(sdx_err.mean()),
           'sdx_err_cm_t': {f'{a}-{b}': float(sdx_err[(sdx_t >= a) & (sdx_t < b)].mean()) for a, b in ((0, 10), (10, 30), (30, 60), (60, 400)) if ((sdx_t >= a) & (sdx_t < b)).any()}}
    if ref:
        res['ref_rot'] = json.load(open(ref))['summary']['sdx_rot_reward_mean']
    if 'in_tactile' in r:  # student record: the same estimator on its own Isaac Gym rollout
        ig_err, ig_t = [], []
        for e in range(n):
            L = int(r['lengths'][e])
            for t in range(L):
                pose = r['init_obj'][e, :7] if t == 0 else r['obj_pose'][e, t - 1]
                ig_err.append(err_cm(predict(sp, r['in_proprio'][e, t], r['in_tactile'][e, t], r['in_obs'][e, t]), true_ends(pose, term))); ig_t.append(t)
        ig_err, ig_t = np.array(ig_err), np.array(ig_t)
        res['ig_err_cm'] = float(ig_err.mean())
        res['ig_err_cm_t'] = {f'{a}-{b}': float(ig_err[(ig_t >= a) & (ig_t < b)].mean()) for a, b in ((0, 10), (10, 30), (30, 60), (60, 400)) if ((ig_t >= a) & (ig_t < b)).any()}
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])
