"""Contact alignment model f: Isaac Gym contact history + joint state + action -> SuperDex binary contact (32).

Data: runs/pairs_v2/dataset_mu1.npz (step-wise resync pairs, SuperDex mu=1.0). Budgets use the first N train-pool
episodes in dataset order; val = held-out teacher record (seed 102); shift = B+tactile student rollouts (seed 103).
Inputs at step t (all available inside Isaac Gym during student training):
  log1p(|F_IG| / 0.05) for steps t, t-1, t-2 (32 each; zero before the episode start), q_t, q_{t-1}, target_t, action_t.
Target: SuperDex |F| >= 0.05 after the same step. No model selection on val: fixed architecture and step count.
Baselines: identity (IG |F| >= 0.05) and per-channel threshold recalibration (tau_c fitted on the budget's train
episodes to maximise train F1). Metrics: per-channel F1 and micro F1 over the 31 channels other than channel 11
(the IG index -1 quirk = total pen force, always on in both simulators).

usage: train_f.py <dataset.npz> <out_dir> [budgets=10,25,50,100] [seeds=0,1,2] [steps=4000] [extra=] [evals=val,shift]
  extra: splits added in full to every budget's training set (e.g. student-rollout pairs); budget 0 = extra only.
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

THRESH, HIST, PEN_CH = 0.05, 3, 11
KEEP = [i for i in range(32) if i != PEN_CH]


def features(d, idx):
    """Feature rows for dataset rows idx (episodes are contiguous with t = 0..L-1)."""
    c = np.log1p(d['ig_contact'] / THRESH)
    t = d['t'][idx]
    parts = []
    for k in range(HIST):
        prev = np.where(t >= k, idx - k, idx)
        parts.append(np.where((t >= k)[:, None], c[prev], 0.0))
    qprev = np.where((t >= 1)[:, None], d['q'][np.where(t >= 1, idx - 1, idx)], d['q'][idx])
    parts += [d['q'][idx], qprev, d['target'][idx], d['action'][idx]]
    return np.concatenate(parts, 1).astype(np.float32)


def f1_scores(pred, true):
    tp = (pred & true).sum(0); fp = (pred & ~true).sum(0); fn = (~pred & true).sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        per = np.where(tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), np.nan)
    k = KEEP
    micro = 2 * tp[k].sum() / max(2 * tp[k].sum() + fp[k].sum() + fn[k].sum(), 1)
    return {'micro_f1_ex_pen': float(micro), 'per_channel_f1': [None if np.isnan(x) else round(float(x), 4) for x in per],
            'on_rate_pred_ex_pen': float(pred[:, k].mean()), 'on_rate_true_ex_pen': float(true[:, k].mean()),
            'precision_ex_pen': float(tp[k].sum() / max(tp[k].sum() + fp[k].sum(), 1)),
            'recall_ex_pen': float(tp[k].sum() / max(tp[k].sum() + fn[k].sum(), 1))}


class FNet(nn.Module):
    def __init__(self, din, mean, std, hidden=256):
        super().__init__()
        self.register_buffer('mean', torch.as_tensor(mean)); self.register_buffer('std', torch.as_tensor(std))
        self.mlp = nn.Sequential(nn.Linear(din, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, 32))

    def forward(self, x):
        return self.mlp((x - self.mean) / self.std)


def fit_thresholds(norm, true):
    grid = np.concatenate([[THRESH], np.geomspace(1e-3, 5.0, 60)])
    taus = np.full(32, THRESH)
    for ch in range(32):
        best = -1
        for tau in grid:
            p = norm[:, ch] >= tau; tp = (p & true[:, ch]).sum(); den = p.sum() + true[:, ch].sum()
            f = 2 * tp / den if den else 0.0
            if f > best + 1e-9:
                best, taus[ch] = f, tau
    return taus


def main(ds_path, out_dir, budgets='10,25,50,100', seeds='0,1,2', steps='4000', extra='', evals='val,shift'):
    torch.set_num_threads(4)
    d = dict(np.load(ds_path)); out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    split = d['split']; ep = d['episode']
    y_all = d['resync_contact'] >= THRESH
    extra = [x for x in extra.split(',') if x]; evals = evals.split(',')
    rows = {s: np.nonzero(split == s)[0] for s in ['train'] + extra + evals}
    assert all(len(r) for r in rows.values()), {s: len(r) for s, r in rows.items()}
    extra_rows = np.concatenate([rows[s] for s in extra]) if extra else np.zeros(0, int)
    X = {s: features(d, rows[s]) for s in evals}
    Y = {s: y_all[rows[s]] for s in evals}
    train_eps = list(dict.fromkeys(ep[rows['train']].tolist()))  # dataset order
    results = {'dataset': str(ds_path), 'features': 'log1p(|F|/0.05) x3 steps, q_t, q_{t-1}, target_t, action_t',
               'steps': int(steps), 'extra': extra, 'extra_steps': int(len(extra_rows)), 'evals': evals, 'budgets': {}}
    for s in evals:
        results[f'identity_{s}'] = f1_scores(d['ig_contact'][rows[s]] >= THRESH, Y[s])
    for B in [int(b) for b in budgets.split(',')]:
        sel = np.concatenate([rows['train'][np.isin(ep[rows['train']], train_eps[:B])], extra_rows])
        xtr, ytr = features(d, sel), y_all[sel]
        taus = fit_thresholds(d['ig_contact'][sel], ytr)
        rb = {'episodes': B, 'train_steps': int(len(sel)), 'tau': taus.round(4).tolist()}
        for s in evals:
            rb[f'threshold_{s}'] = f1_scores(d['ig_contact'][rows[s]] >= taus, Y[s])
        mean, std = xtr.mean(0), xtr.std(0) + 1e-3
        rb['f'] = []
        for seed in [int(x) for x in seeds.split(',')]:
            torch.manual_seed(seed); rng = np.random.default_rng(seed)
            net = FNet(xtr.shape[1], mean, std)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
            xt, yt = torch.from_numpy(xtr), torch.from_numpy(ytr.astype(np.float32))
            t0 = time.time()
            for it in range(int(steps)):
                b = torch.from_numpy(rng.integers(0, len(xt), 512))
                loss = nn.functional.binary_cross_entropy_with_logits(net(xt[b]), yt[b])
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                tr_loss = float(nn.functional.binary_cross_entropy_with_logits(net(xt), yt))
                r = {'seed': seed, 'train_bce': tr_loss, 'wall_s': round(time.time() - t0, 1)}
                for s in evals:
                    pred = (net(torch.from_numpy(X[s])) > 0).numpy()
                    r[s] = f1_scores(pred, Y[s])
            torch.save({'state_dict': net.state_dict(), 'din': xtr.shape[1], 'budget': B, 'seed': seed,
                        'features': results['features']}, out / f'f_b{B}_s{seed}.pt')
            rb['f'].append(r)
            print(json.dumps({'budget': B, 'seed': seed, 'train_bce': round(tr_loss, 4),
                              **{f'{s}_f1': round(r[s]['micro_f1_ex_pen'], 4) for s in evals},
                              **{f'thr_{s}_f1': round(rb[f'threshold_{s}']['micro_f1_ex_pen'], 4) for s in evals}}), flush=True)
        results['budgets'][B] = rb
    print('IDENTITY', json.dumps({s: round(results[f'identity_{s}']['micro_f1_ex_pen'], 4) for s in evals}))
    json.dump(results, open(out / 'f_results.json', 'w'), indent=1)


if __name__ == '__main__':
    kv = dict(a.split('=', 1) for a in sys.argv[3:])
    main(sys.argv[1], sys.argv[2], **kv)
