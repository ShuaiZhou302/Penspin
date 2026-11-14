"""Closed-loop oracle rollouts in SuperDex for the fine-tuning baseline (privileged target state; not PenSpin's open-loop replay).

The frozen teacher runs from each record's initial state, with the pen length and termination of closed_loop.py penlen=ig.
Each attempt stores the student's inputs before every action (obs96, proprio_hist, tactile_hist) and the teacher action.
usage: collect_oracle.py <record.npz> <teacher.npz> <out.npz> [episodes=a:b] [friction=1.0] [seed=0]"""
import json
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sdx_env import PenSpinSDX, IG_LOWER, IG_UPPER  # noqa: E402
from teacher_np import TeacherNP, TeacherObs, ACTION_SCALE, CONTROL_DT, FINGERTIPS  # noqa: E402
from closed_loop import URDF, RESET_Z, END_Z  # noqa: E402


def main(record, teacher_npz, out, episodes=None, friction='1.0', seed='0'):
    r = dict(np.load(record)); teacher = TeacherNP(teacher_npz); names = [str(x) for x in r['dof_names']]
    n = len(r['lengths']); T = r['target'].shape[1]; a, b = (int(x) for x in episodes.split(':')) if episodes else (0, n)
    mu = float(friction); rng = np.random.default_rng(int(seed))
    scale = r['obj_scale'].astype(float); pc_local = r['obj_point_cloud_local']
    pen_lens = np.round((pc_local[..., 2].max(1) - pc_local[..., 2].min(1)) / scale, 2) * scale; term = 0.52 * scale[-1]
    keep = {k: [] for k in ('episode', 'steps', 'n_frames', 'rot_rad', 'rot_reward')}
    fr = {k: [] for k in ('obs', 'proprio_hist', 'tactile_hist', 'action')}
    t0 = time.time()
    for e in range(a, b):
        L = int(r['lengths'][e])
        if L < 1:  # same attempts as the pair dataset
            continue
        env = PenSpinSDX(URDF, names, pen_scale=float(scale[e]), pen_mass=float(r['obj_mass'][e]), friction=mu,
                         pen_length=float(pen_lens[e]), term_length=term)
        o = r['init_obj'][e]
        env.reset(r['init_q'][e], o[:3], o[3:7], r['init_qd'][e], o[7:10], o[10:13])
        init_pose = r['in_obs'][e, 1, :16] if L > 1 else r['init_q'][e]
        ob = TeacherObs(init_pose, r['in_obs'][e, 0], r['in_priv'][e, 0], pc_local[e], float(scale[e]), float(r['obj_mass'][e]),
                        mu, float(r['obj_restitution'][e]), rng, obs_noise_t=0.01, obs_noise_e=0.02)
        obs, priv, pc = r['in_obs'][e, 0], r['in_priv'][e, 0], r['in_pc'][e, 0]
        target = r['init_target'][e].astype(float); prev = o[3:7]; rot_rad = rot_reward = 0.0; steps = T; frames = 0
        for t in range(T):
            act = teacher.act(obs[None], priv[None], pc[None])[0]
            fr['obs'].append(np.asarray(obs, np.float32)); fr['proprio_hist'].append(ob.proprio_hist.copy())
            fr['tactile_hist'].append(ob.tactile_hist.copy()); fr['action'].append(np.asarray(act, np.float32)); frames += 1
            target = np.clip(target + ACTION_SCALE * act, IG_LOWER, IG_UPPER)
            cn = env.control_step(target); pose = env.pen_pose()
            ft = np.array([[env.links[k].get_root_transform().translation[i] for i in range(3)] for k in FINGERTIPS])
            obs, priv, pc, angvel = ob.update(env.q(), target, cn, ft, pose[:3], pose[3:7], prev)
            prev = pose[3:7]; vz = float(angvel[2]); rot_rad += vz * CONTROL_DT; rot_reward += float(np.clip(vz, -0.5, 0.5))
            zmin, zmax = env.pen_ends_z()
            if pose[2] < RESET_Z or zmin < END_Z[0] or zmax > END_Z[1]:
                steps = t
                break
        env.close()
        row = dict(episode=e, steps=steps, n_frames=frames, rot_rad=rot_rad, rot_reward=rot_reward)
        for k, v in row.items():
            keep[k].append(v)
        print(json.dumps({**row, 'wall_s': round(time.time() - t0, 1)}), flush=True)
    np.savez_compressed(out, record=str(record), friction=mu, **{k: np.array(v, np.float32) for k, v in fr.items()},
                        **{k: np.array(v) for k, v in keep.items()})
    print('ORACLE_DONE', out, len(keep['episode']), 'frames', len(fr['action']), flush=True)


if __name__ == '__main__':
    kv = dict(x.split('=', 1) for x in sys.argv[4:])
    main(sys.argv[1], sys.argv[2], sys.argv[3], **kv)
