"""PenSpin-style fine-tuning (real/finetune_ppo.py, scripts/finetune_ppo.sh) on closed-loop oracle rollouts in SuperDex.

Budget = first N attempts in dataset order; attempts with total rotation >= thresh are kept. As official: L1 on the clamped
action, Adam lr 1e-3, batch 512, no weight decay, 3000 epochs, all parameters, normalizer in train mode, final epoch used.
Different from official: inputs end at the action's step (official windows look ahead of the label), the label is the oracle
action, validation holds out whole demos (only with >= 10 demos; it is not used for selection), and with fewer than 512 frames
batches are drawn with replacement.
usage: finetune_student.py <ckpt> <mode> <out_prefix> budget=<N> seed=<S> [thresh=6.283185] [data=runs/oracle] [epochs=3000]"""
import glob
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from student_policy import ActorCritic, NET, RunningMeanStd  # noqa: E402

BATCH, LR, VAL_RATIO = 512, 1e-3, 0.1


def shard_key(p):
    name = Path(p).stem  # pairT_seed100_777112_16-32
    return int(name.split('seed')[1].split('_')[0]), int(name.rsplit('_', 1)[1].split('-')[0])


def load_demos(data, budget, thresh):
    """Attempts in dataset order; returns the kept demos as dicts of arrays, plus the attempt counts."""
    attempts, demos = 0, []
    for f in sorted(glob.glob(f'{data}/*.npz'), key=shard_key):
        d = dict(np.load(f)); start = 0
        for n_frames, rot in zip(d['n_frames'], d['rot_rad']):
            sl = slice(start, start + int(n_frames)); start += int(n_frames)
            if attempts < budget and rot >= thresh:
                demos.append({k: d[k][sl] for k in ('obs', 'proprio_hist', 'tactile_hist', 'action')})
            attempts += 1
    return demos, min(attempts, budget)


def main(ckpt, mode, out_prefix, budget, seed, thresh='6.283185', data='runs/oracle', epochs='3000', lr='1e-3'):
    budget, seed, thresh, epochs = int(budget), int(seed), float(thresh), int(epochs)
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    demos, attempts = load_demos(data, budget, thresh)
    res = {'ckpt': ckpt, 'mode': mode, 'budget': budget, 'attempts': attempts, 'thresh': thresh, 'seed': seed, 'demos': len(demos),
           'device': device}
    if not demos:
        Path(out_prefix + '.json').write_text(json.dumps(res, indent=1)); print('NO_DEMOS', json.dumps(res), flush=True)
        return
    n_val = len(demos) - int(len(demos) * (1 - VAL_RATIO)) if len(demos) >= 10 else 0
    perm = rng.permutation(len(demos)); val_ids, train_ids = perm[:n_val], perm[n_val:]
    cat = lambda ids, k: torch.from_numpy(np.concatenate([demos[i][k] for i in ids])).to(device)
    tr = {k: cat(train_ids, k) for k in demos[0]}
    va = {k: cat(val_ids, k) for k in demos[0]} if n_val else None
    ck = torch.load(ckpt, map_location='cpu')
    model = ActorCritic(dict(NET, input_mode=mode)).to(device); model.load_state_dict(ck['model'], strict=True)
    rms = RunningMeanStd((96,)).to(device); rms.load_state_dict(ck['running_mean_std'], strict=True)
    opt = torch.optim.Adam(model.parameters(), float(lr), weight_decay=0)

    def l1(b, idx=None):
        take = (lambda x: x) if idx is None else (lambda x: x[idx])
        inp = {'obs': rms(take(b['obs'])), 'proprio_hist': take(b['proprio_hist'])}
        if mode != 'proprio':
            inp['obj_ends'] = torch.zeros(inp['obs'].shape[0], 3, 6, device=device)
        if 'tactile' in mode:
            inp['tactile_hist'] = take(b['tactile_hist'])
        return torch.nn.functional.l1_loss(model._actor_critic(inp)[0].clamp(-1, 1), take(b['action']))

    def evaluate():
        model.eval(); rms.eval()
        with torch.no_grad():
            out = (float(l1(tr)), float(l1(va)) if va else None)
        model.train(); rms.train()
        return out

    n = len(tr['action']); it_per_epoch = max(1, n // BATCH); t0 = time.time()
    res.update(train_demos=len(train_ids), val_demos=n_val, train_frames=n, it_per_epoch=it_per_epoch, updates=epochs * it_per_epoch,
               start_l1=evaluate(), curve=[])
    print('START', json.dumps(res), flush=True)
    for ep in range(epochs):
        for _ in range(it_per_epoch):
            idx = torch.randperm(n, device=device)[:BATCH] if n >= BATCH else torch.randint(0, n, (BATCH,), device=device)
            loss = l1(tr, idx)
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 100 == 0 or ep + 1 == epochs:
            tl, vl = evaluate(); res['curve'].append([ep + 1, tl, vl])
            print(json.dumps({'epoch': ep + 1, 'train_l1': tl, 'val_l1': vl, 'wall_s': round(time.time() - t0, 1)}), flush=True)
    torch.save({'model': model.state_dict(), 'running_mean_std': rms.state_dict()}, out_prefix + '.pth')
    res['wall_s'] = round(time.time() - t0, 1)
    Path(out_prefix + '.json').write_text(json.dumps(res, indent=1))
    print('FINETUNE_DONE', out_prefix, flush=True)


if __name__ == '__main__':
    kv = dict(a.split('=', 1) for a in sys.argv[4:])
    main(sys.argv[1], sys.argv[2], sys.argv[3], **kv)
