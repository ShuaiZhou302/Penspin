"""Open-loop replay of recorded Isaac Gym teacher episodes in SuperDex, with per-episode fidelity metrics.

usage: replay_compare.py <record.npz> <out_prefix> [max_episodes]
For every recorded episode: same pen size/mass/friction, same initial hand/pen state, then the recorded per-step joint
targets through the same 200 Hz torque PD. SuperDex keeps going after Isaac Gym's episode ended only up to its length.
Termination in SuperDex uses the Isaac Gym rules (object z < 0.605 or pencil end z outside [0.56, 0.71]).
Writes <out_prefix>.json (summary + per-episode metrics) and <out_prefix>.npz (SuperDex traces, for training f).
"""
import json
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sdx_env import PenSpinSDX, CONTACT_THRESH  # noqa: E402

URDF = str(Path(__file__).resolve().parent.parent / 'assets/allegro/allegro_superdex.urdf')
RESET_Z, END_Z = 0.605, (0.56, 0.71)


def shaft_heading(quat_xyzw):
    x, y, z, w = quat_xyzw
    axis = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    return np.arctan2(axis[1], axis[0])


def f1(a, b):
    tp = float((a & b).sum()); fp = float((~a & b).sum()); fn = float((a & ~b).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1.0)


def main(record, out_prefix, max_eps=None, inertia='mesh'):
    r = dict(np.load(record))  # NpzFile re-decompresses on every key access
    names = [str(x) for x in r['dof_names']]
    n = len(r['lengths']) if max_eps is None else min(int(max_eps), len(r['lengths']))
    traces = {k: [] for k in ('q', 'obj_pose', 'contact_norm', 'alive')}
    eps, t0 = [], time.time()
    for e in range(n):
        L = int(r['lengths'][e])
        env = PenSpinSDX(URDF, names, pen_scale=float(r['obj_scale'][e]), pen_mass=float(r['obj_mass'][e]),
                         friction=float(r['obj_friction'][e]),
                         pen_inertia=r['obj_inertia'][e] if inertia == 'ig' else None)
        o = r['init_obj'][e]
        env.reset(r['init_q'][e], o[:3], o[3:7], r['init_qd'][e], o[7:10], o[10:13])
        T = r['target'].shape[1]
        q = np.full((T, 16), np.nan); pose = np.full((T, 7), np.nan); cn = np.full((T, 32), np.nan)
        alive = np.zeros(T, bool); sdx_len = L
        for t in range(L):
            cn[t] = env.control_step(r['target'][e, t]); q[t] = env.q(); pose[t] = env.pen_pose(); alive[t] = True
            zmin, zmax = env.pen_ends_z()
            if pose[t, 2] < RESET_Z or zmin < END_Z[0] or zmax > END_Z[1]:
                sdx_len = t
                break
        both = slice(0, min(L, sdx_len))
        ig_c = r['contact_norm'][e, both] >= CONTACT_THRESH
        sd_c = cn[both] >= CONTACT_THRESH
        pos_err = np.linalg.norm(pose[both, :3] - r['obj_pose'][e, both, :3], axis=-1)
        ig_head = np.array([shaft_heading(p[3:]) for p in r['obj_pose'][e, both]])
        sd_head = np.array([shaft_heading(p[3:]) for p in pose[both]])
        dh = np.abs(np.angle(np.exp(1j * (ig_head - sd_head))))
        k = both.stop
        eps.append({'episode': e, 'ig_steps': L, 'sdx_steps': int(sdx_len), 'sdx_survived_ig_length': bool(sdx_len >= L),
                    'q_rmse_rad': float(np.sqrt(np.mean((q[both] - r['q'][e, both]) ** 2))) if k else None,
                    'pen_pos_err_cm_at_1s_2s_end': [float(100 * pos_err[min(i, k - 1)]) if k else None for i in (19, 39, k - 1)],
                    'heading_err_rad_at_1s_2s_end': [float(dh[min(i, k - 1)]) if k else None for i in (19, 39, k - 1)],
                    'contact_f1_all_channels': f1(ig_c, sd_c) if k else None,
                    'contact_on_rate_ig_sdx': [float(ig_c.mean()) if k else None, float(sd_c.mean()) if k else None]})
        env.close()
        for key, v in (('q', q), ('obj_pose', pose), ('contact_norm', cn), ('alive', alive)):
            traces[key].append(v)
        print(json.dumps(eps[-1]), flush=True)
    np.savez_compressed(out_prefix + '.npz', **{k: np.stack(v) for k, v in traces.items()}, record=str(record))
    agg = lambda key: float(np.nanmean([x[key] for x in eps if x[key] is not None]))
    summary = {'record': str(record), 'pen_inertia': inertia, 'episodes': n, 'wall_s': round(time.time() - t0, 1),
               'ig_mean_steps': float(np.mean([x['ig_steps'] for x in eps])), 'sdx_mean_steps': float(np.mean([x['sdx_steps'] for x in eps])),
               'sdx_survived_ig_length_frac': float(np.mean([x['sdx_survived_ig_length'] for x in eps])),
               'q_rmse_rad_mean': agg('q_rmse_rad'), 'contact_f1_mean': agg('contact_f1_all_channels'),
               'pen_pos_err_cm_1s_median': float(np.nanmedian([x['pen_pos_err_cm_at_1s_2s_end'][0] for x in eps
                                                                if x['pen_pos_err_cm_at_1s_2s_end'][0] is not None] or [np.nan])),
               'contact_on_rate_ig_sdx': [float(np.mean([x['contact_on_rate_ig_sdx'][i] for x in eps
                                                         if x['contact_on_rate_ig_sdx'][i] is not None] or [np.nan])) for i in (0, 1)]}
    Path(out_prefix + '.json').write_text(json.dumps({'summary': summary, 'episodes': eps}, indent=1))
    print('REPLAY_SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != 'all' else None,
         sys.argv[4] if len(sys.argv) > 4 else 'mesh')
