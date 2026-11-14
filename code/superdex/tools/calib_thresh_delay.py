"""Threshold/delay calibration baseline: SDX binary contact at t := |F_IG(t-k)| >= tau_ch, with per-channel tau fit on
the same training rows as f (first 100 'train' teacher episodes + extra student splits) for each k in {0,1,2}, and k
chosen by val micro F1 (excluding the pen-total channel). Rows before the episode start use |F| = 0.
usage: calib_thresh_delay.py <dataset.npz> <out.npz> <out.json> [budget=100] [extra=train_bt,train_p] [evals=val,shift,shift_p]"""
import json
import sys
import numpy as np
from train_f import THRESH, f1_scores, fit_thresholds


def lagged(d, k):
    """ig_contact at t-k of the same episode (0 before the episode start), row-aligned with d."""
    if k == 0:
        return d['ig_contact']
    out = np.zeros_like(d['ig_contact'])
    key = {(s, e, t): i for i, (s, e, t) in enumerate(zip(d['split'], d['episode'], d['t']))}
    for i, (s, e, t) in enumerate(zip(d['split'], d['episode'], d['t'])):
        j = key.get((s, e, t - k))
        if j is not None:
            out[i] = d['ig_contact'][j]
        else:
            assert t - k < 0, (s, e, t, k)
    return out


def main(ds, out_npz, out_json, budget='100', extra='train_bt,train_p', evals='val,shift,shift_p'):
    d = dict(np.load(ds)); y = d['resync_contact'] >= THRESH
    split, ep = d['split'], d['episode']
    extra = [x for x in extra.split(',') if x]; evals = evals.split(',')
    tr = np.nonzero(split == 'train')[0]
    train_eps = list(dict.fromkeys(ep[tr].tolist()))[:int(budget)]
    sel = np.concatenate([tr[np.isin(ep[tr], train_eps)]] + [np.nonzero(split == s)[0] for s in extra])
    rows = {s: np.nonzero(split == s)[0] for s in evals}
    res = {'dataset': ds, 'budget': int(budget), 'extra': extra, 'train_steps': int(len(sel)), 'lags': {}}
    for s in evals:
        res[f'identity_{s}'] = f1_scores(d['ig_contact'][rows[s]] >= THRESH, y[rows[s]])['micro_f1_ex_pen']
    best = None
    for k in (0, 1, 2):
        norm = lagged(d, k)
        taus = fit_thresholds(norm[sel], y[sel])
        sc = {s: f1_scores(norm[rows[s]] >= taus, y[rows[s]])['micro_f1_ex_pen'] for s in evals}
        on = float((norm[sel] >= taus)[:, np.arange(32) != 11].mean())
        res['lags'][k] = {'f1': sc, 'train_on_rate_ex_pen': on, 'taus': taus.tolist()}
        print('LAG', k, json.dumps({s: round(v, 4) for s, v in sc.items()}), 'on', round(on, 4), flush=True)
        if best is None or sc['val'] > res['lags'][best]['f1']['val']:
            best = k
    res['chosen_lag'] = best
    res['sdx_train_on_rate_ex_pen'] = float(y[sel][:, np.arange(32) != 11].mean())
    np.savez(out_npz, taus=np.array(res['lags'][best]['taus'], np.float32), lag=np.int64(best))
    json.dump(res, open(out_json, 'w'), indent=1)
    print('CHOSEN lag', best, 'taus', np.round(res['lags'][best]['taus'], 3).tolist())


if __name__ == '__main__':
    main(*sys.argv[1:])
