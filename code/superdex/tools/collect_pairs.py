"""Paired Isaac Gym -> SuperDex contact data for the contact alignment model f.

For every valid step t of every episode in an Isaac Gym record (recorder >= 3f7183c, which stores per-step qd and
object velocities), two SuperDex contact readings are produced under the same joint target target[t]:
  resync: SuperDex is reset to the Isaac Gym state before step t (init state for t=0, else q/qd/pen pose/pen
          velocity after step t-1), then one control step is run. The pair (IG contact[t], SDX contact[t]) then
          differs only by the contact model, not by trajectory drift. One-step joint / pen errors are stored.
  open:   plain open-loop replay of target[0..t] from the initial state (the proposal's replay); its pen position
          error to Isaac Gym is stored so drifted steps can be masked.
The target domain is fixed: SuperDex friction mu (default 1.0), physical (mesh) pen inertia.

usage: collect_pairs.py <record.npz> <out.npz> [episodes=a:b] [friction=1.0]
"""
import json
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sdx_env import PenSpinSDX  # noqa: E402

URDF = str(Path(__file__).resolve().parent.parent / 'assets/allegro/allegro_superdex.urdf')
QUAT_ERR = lambda a, b: 2 * np.arccos(np.clip(np.abs((a * b).sum(-1)), 0, 1))  # rad, xyzw


def main(record, out, episodes=None, friction='1.0', penlen=None):
    r = dict(np.load(record))
    assert 'qd' in r and 'obj_vel' in r, 'record lacks per-step velocities (needs recorder >= 3f7183c)'
    names = [str(x) for x in r['dof_names']]
    n = len(r['lengths'])
    a, b = (int(x) for x in episodes.split(':')) if episodes else (0, n)
    mu = float(friction)
    scale = r['obj_scale'].astype(float); pc = r['obj_point_cloud_local']
    pen_lens = np.round((pc[..., 2].max(1) - pc[..., 2].min(1)) / scale, 2) * scale
    keys = ('resync_contact', 'open_contact', 'resync_q_err', 'resync_pen_pos_err', 'resync_pen_rot_err', 'open_pen_pos_err')
    res = {k: [] for k in keys}
    ep_ids, lens, t0 = [], [], time.time()
    for e in range(a, b):
        L = int(r['lengths'][e])
        if L < 1:
            continue
        env = PenSpinSDX(URDF, names, pen_scale=float(r['obj_scale'][e]), pen_mass=float(r['obj_mass'][e]), friction=mu,
                         pen_length=float(pen_lens[e]) if penlen == 'ig' else None)
        o = r['init_obj'][e]
        rc, rq, rp, rr = np.zeros((L, 32)), np.zeros(L), np.zeros(L), np.zeros(L)
        for t in range(L):
            if t == 0:
                env.reset(r['init_q'][e], o[:3], o[3:7], r['init_qd'][e], o[7:10], o[10:13])
            else:
                p, v = r['obj_pose'][e, t - 1], r['obj_vel'][e, t - 1]
                env.reset(r['q'][e, t - 1], p[:3], p[3:7], r['qd'][e, t - 1], v[:3], v[3:6])
            rc[t] = env.control_step(r['target'][e, t])
            pose = env.pen_pose()
            rq[t] = np.abs(env.q() - r['q'][e, t]).max()
            rp[t] = np.linalg.norm(pose[:3] - r['obj_pose'][e, t, :3])
            rr[t] = QUAT_ERR(pose[3:7], r['obj_pose'][e, t, 3:7])
        oc, op = np.zeros((L, 32)), np.zeros(L)
        env.reset(r['init_q'][e], o[:3], o[3:7], r['init_qd'][e], o[7:10], o[10:13])
        for t in range(L):
            oc[t] = env.control_step(r['target'][e, t])
            op[t] = np.linalg.norm(env.pen_pose()[:3] - r['obj_pose'][e, t, :3])
        env.close()
        for k, v in zip(keys, (rc, oc, rq, rp, rr, op)):
            res[k].append(v)
        ep_ids.append(e); lens.append(L)
        print(json.dumps({'episode': e, 'steps': L, 'resync_q_err_med': float(np.median(rq)),
                          'resync_pen_pos_err_med_cm': float(np.median(rp) * 100),
                          'on_ig': float((r['contact_norm'][e, :L] >= 0.05).mean()), 'on_resync': float((rc >= 0.05).mean()),
                          'on_open': float((oc >= 0.05).mean()), 'wall_s': round(time.time() - t0, 1)}), flush=True)
    np.savez_compressed(out, record=str(record), friction=mu, episodes=np.array(ep_ids), lengths=np.array(lens),
                        **{k: np.concatenate(v) for k, v in res.items()})
    print('PAIRS_DONE', out, 'episodes', len(ep_ids), 'steps', int(sum(lens)), 'wall_s', round(time.time() - t0, 1), flush=True)


if __name__ == '__main__':
    kv = dict(x.split('=', 1) for x in sys.argv[3:])
    main(sys.argv[1], sys.argv[2], **kv)
