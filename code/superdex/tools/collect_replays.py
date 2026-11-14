"""Open-loop replay of Isaac Gym oracle targets in SuperDex for PenSpin-style policy fine-tuning.

Each attempted replay starts from the record's initial state and applies its joint targets until the record ends or an
Isaac Gym termination rule fires (pen length and termination as closed_loop.py penlen=ig). Saved per frame: the noisy
proprio frame [noisy_q, target] the student would observe (same noise model as closed_loop) and the oracle action.
usage: collect_replays.py <record.npz> <out.npz> [episodes=a:b] [friction=1.0] [seed=0]"""
import json
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sdx_env import PenSpinSDX  # noqa: E402
from teacher_np import TeacherObs, CONTROL_DT, FINGERTIPS  # noqa: E402
from closed_loop import URDF, RESET_Z, END_Z  # noqa: E402


def main(record, out, episodes=None, friction='1.0', seed='0'):
    r = dict(np.load(record)); names = [str(x) for x in r['dof_names']]
    n = len(r['lengths']); a, b = (int(x) for x in episodes.split(':')) if episodes else (0, n)
    mu = float(friction); rng = np.random.default_rng(int(seed))
    scale = r['obj_scale'].astype(float); pc = r['obj_point_cloud_local']
    pen_lens = np.round((pc[..., 2].max(1) - pc[..., 2].min(1)) / scale, 2) * scale; term = 0.52 * scale[-1]
    keep = {k: [] for k in ('episode', 'ig_steps', 'steps', 'rot_rad', 'rot_reward', 'ig_rot_rad')}
    frames, actions, starts, t0 = [], [], [], time.time()
    for e in range(a, b):
        L = int(r['lengths'][e])
        if L < 1:
            continue
        env = PenSpinSDX(URDF, names, pen_scale=float(scale[e]), pen_mass=float(r['obj_mass'][e]), friction=mu,
                         pen_length=float(pen_lens[e]), term_length=term)
        o = r['init_obj'][e]
        env.reset(r['init_q'][e], o[:3], o[3:7], r['init_qd'][e], o[7:10], o[10:13])
        init_pose = r['in_obs'][e, 1, :16] if L > 1 else r['init_q'][e]
        ob = TeacherObs(init_pose, r['in_obs'][e, 0], r['in_priv'][e, 0], pc[e], float(scale[e]), float(r['obj_mass'][e]),
                        mu, float(r['obj_restitution'][e]), rng, obs_noise_t=0.01, obs_noise_e=0.02)
        prev = o[3:7]; rot_rad = rot_reward = 0.0; steps = L
        starts.append(len(frames))
        for t in range(L):
            frames.append(ob.proprio_hist[-1].copy()); actions.append(np.clip(r['action'][e, t], -1, 1))
            cn = env.control_step(r['target'][e, t]); pose = env.pen_pose()
            ft = np.array([[env.links[k].get_root_transform().translation[i] for i in range(3)] for k in FINGERTIPS])
            _, _, _, angvel = ob.update(env.q(), r['target'][e, t], cn, ft, pose[:3], pose[3:7], prev)
            prev = pose[3:7]; vz = float(angvel[2]); rot_rad += vz * CONTROL_DT; rot_reward += float(np.clip(vz, -0.5, 0.5))
            zmin, zmax = env.pen_ends_z()
            if pose[2] < RESET_Z or zmin < END_Z[0] or zmax > END_Z[1]:
                steps = t
                break
        env.close()
        row = dict(episode=e, ig_steps=L, steps=steps, rot_rad=rot_rad, rot_reward=rot_reward, ig_rot_rad=float(r['rot_rad'][e, :L].sum()))
        for k, v in row.items():
            keep[k].append(v)
        print(json.dumps({**row, 'wall_s': round(time.time() - t0, 1)}), flush=True)
    np.savez_compressed(out, record=str(record), friction=mu, frames=np.array(frames, np.float32), actions=np.array(actions, np.float32),
                        starts=np.array(starts), **{k: np.array(v) for k, v in keep.items()})
    print('REPLAYS_DONE', out, len(keep['episode']), 'frames', len(frames), flush=True)


if __name__ == '__main__':
    kv = dict(x.split('=', 1) for x in sys.argv[3:])
    main(sys.argv[1], sys.argv[2], **kv)
