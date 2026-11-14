"""Merge paired contact files into one dataset and report pairing fidelity and the identity baseline
(SuperDex contact predicted = Isaac Gym contact), which the alignment model f has to beat.

Splits (by Isaac Gym record): train pool = pairT seeds 100, 101 (budgets take its episodes in order: first 10/25/50/100);
val = pairT seed 102 (held out); shift = pairBT seed 103 (B+tactile student rollouts, policy-shift check).
usage: pair_stats.py <pairs_dir> <out_dataset.npz> <out_stats.json> [split=file_prefix ...]
  (split specs replace the default SPLITS; a split may be listed several times)
"""
import glob
import json
import os
import sys
import numpy as np

THRESH = 0.05
SPLITS = [('train', 'pairT_seed100_'), ('train', 'pairT_seed101_'), ('val', 'pairT_seed102_'), ('shift', 'pairBT_seed103_')]


def f1(pred, true):
    tp = (pred & true).sum(0); fp = (pred & ~true).sum(0); fn = (~pred & true).sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        per = 2 * tp / (2 * tp + fp + fn)
    micro = 2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1)
    return per, float(micro)


def main(pdir, out_ds, out_stats, *specs):
    splits = [tuple(x.split('=', 1)) for x in specs] if specs else SPLITS
    cols = {k: [] for k in ('split', 'episode', 't', 'ig_contact', 'ig_sensed', 'q', 'target', 'action',
                            'resync_contact', 'open_contact', 'resync_q_err', 'resync_pen_pos_err', 'resync_pen_rot_err',
                            'open_pen_pos_err')}
    gid = 0; records = {}
    for split, prefix in splits:
        files = sorted(glob.glob(os.path.join(pdir, prefix + '*.npz')), key=lambda p: int(p.rsplit('_', 1)[1].split('-')[0]))
        assert len(files) == 2, (prefix, files)
        for fpath in files:
            p = dict(np.load(fpath)); rec = str(p['record']); records[fpath] = rec
            r = dict(np.load(rec))
            off = 0
            for e, L in zip(p['episodes'], p['lengths']):
                e, L = int(e), int(L); sl = slice(off, off + L); off += L
                cols['split'].append(np.full(L, split)); cols['episode'].append(np.full(L, gid)); cols['t'].append(np.arange(L))
                cols['ig_contact'].append(r['contact_norm'][e, :L])
                # sensed IG tactile seen by the policy for the state after step t (latency/dropout applied): priv dims
                # 29:61 of the next step's input; NaN at the last valid step
                nxt = np.full((L, 32), np.nan); k = min(L, r['in_priv'].shape[1] - 1)
                nxt[:k] = r['in_priv'][e, 1:k + 1, 29:61]
                cols['ig_sensed'].append(nxt)
                for k2 in ('q', 'target', 'action'):
                    cols[k2].append(r[k2][e, :L])
                for k2 in ('resync_contact', 'open_contact', 'resync_q_err', 'resync_pen_pos_err', 'resync_pen_rot_err', 'open_pen_pos_err'):
                    cols[k2].append(p[k2][sl])
                gid += 1
            assert off == len(p['resync_q_err'])
    ds = {k: np.concatenate(v) for k, v in cols.items()}
    ds['split'] = ds['split'].astype('U16')
    np.savez_compressed(out_ds, **ds)
    stats = {'files': records, 'thresh': THRESH}
    names = list(dict.fromkeys(sp for sp, _ in splits))
    for split in names:
        m = ds['split'] == split
        ig = ds['ig_contact'][m] >= THRESH; rs = ds['resync_contact'][m] >= THRESH; op = ds['open_contact'][m] >= THRESH
        per, micro = f1(ig, rs)
        okm = ds['open_pen_pos_err'][m] < 0.01
        _, micro_open = f1(ig[okm], op[okm])
        stats[split] = {
            'episodes': int(len(np.unique(ds['episode'][m]))), 'steps': int(m.sum()),
            'resync_q_err_rad_median': float(np.median(ds['resync_q_err'][m])),
            'resync_q_err_rad_p95': float(np.percentile(ds['resync_q_err'][m], 95)),
            'resync_pen_pos_err_cm_median': float(np.median(ds['resync_pen_pos_err'][m]) * 100),
            'resync_pen_pos_err_cm_p95': float(np.percentile(ds['resync_pen_pos_err'][m], 95) * 100),
            'resync_pen_rot_err_deg_median': float(np.degrees(np.median(ds['resync_pen_rot_err'][m]))),
            'on_rate_ig': float(ig.mean()), 'on_rate_sdx_resync': float(rs.mean()), 'on_rate_sdx_open': float(op.mean()),
            'identity_f1_micro_resync': micro, 'identity_f1_micro_open_err_lt_1cm': micro_open,
            'open_steps_err_lt_1cm_frac': float(okm.mean()),
            'per_channel': {'on_ig': ig.mean(0).round(4).tolist(), 'on_sdx': rs.mean(0).round(4).tolist(),
                            'identity_f1': np.nan_to_num(per, nan=-1).round(3).tolist()}}
    json.dump(stats, open(out_stats, 'w'), indent=1)
    for split in names:
        s = {k: v for k, v in stats[split].items() if k != 'per_channel'}
        print(split, json.dumps(s))


if __name__ == '__main__':
    main(*sys.argv[1:])
