"""The env-side baseline tactile channels (AllegroHandHora._tactile_baseline/_baseline_contacts) on dataset episodes.
TACTILE_CALIB: binary output == calib_thresh_delay's |F_IG(t-k)| >= tau. TACTILE_RAND with forced identity parameters
(tau 0.05, lag 0, no miss/spurious) == |F_IG(t)| >= 0.05; sampled parameters: ranges and on-rate.
usage: test_tacbase.py <dataset.npz> <calib.npz> <superdex tools dir>"""
import os
import sys
import types
import numpy as np
import torch

ds, calib, tools = sys.argv[1:4]
sys.path.insert(0, tools)
from calib_thresh_delay import lagged  # noqa: E402

src = open('penspin/tasks/allegro_hand_hora.py').read()
mod = types.ModuleType('m'); mod.os = os; mod.torch = torch; mod.np = np
start = src.index('    def _tactile_baseline(self):'); end = src.index('    def pre_physics_step(self, actions):')
exec('class H:\n' + src[start:end], mod.__dict__)

d = dict(np.load(ds))
eps = list(dict.fromkeys(d['episode'][d['split'] == 'val'].tolist()))
rows = [r for r in (np.nonzero((d['episode'] == e) & (d['split'] == 'val'))[0] for e in eps) if len(r) >= 200][:16]
L = min(len(r) for r in rows)
c = np.load(calib); k = int(c['lag'])
ref_all = lagged(d, k) >= c['taus']


def run(mode, force_identity=False):
    for v in ('TACTILE_RAND', 'TACTILE_CALIB', 'TACTILE_ALIGN_F'):
        os.environ.pop(v, None)
    os.environ.update({'TACTILE_RAND': '1'} if mode == 'rand' else {'TACTILE_CALIB': calib})
    h = mod.H(); h.num_envs = len(rows); h.device = 'cpu'; h.latency = 0.2; h.sensor_noise = 0.1
    h.at_reset_buf = torch.ones(h.num_envs)
    h._tactile_baseline(); h._baseline_contacts(torch.rand(h.num_envs, 32) * 5)  # reset call with stale inputs
    h.at_reset_buf = torch.zeros(h.num_envs)
    tb = h._tbase
    if force_identity:
        tb['taus'][:] = 0.05; tb['lag'][:] = 0; tb['miss'][:] = 0; tb['spur'][:] = 0
    worst, on = 0, []
    for t in range(L):
        idx = np.array([r[t] for r in rows])
        h._baseline_contacts(torch.from_numpy(d['ig_contact'][idx]).float())
        ref = torch.from_numpy(ref_all[idx] if mode == 'calib' else d['ig_contact'][idx] >= 0.05).float()
        if mode == 'calib' or force_identity:
            worst = max(worst, int((h.baseline_raw_contacts != ref).sum()))
        on.append(float(h.baseline_raw_contacts[:, np.arange(32) != 11].mean()))
    return h, worst, np.mean(on)


h, w, on = run('calib'); print('CALIB lag', k, 'max mismatches per step', w, 'steps', L, 'envs', len(rows), 'on_ex_pen', round(on, 4))
h, w, on = run('rand', force_identity=True); print('RAND_IDENTITY max mismatches per step', w, 'on_ex_pen', round(on, 4))
h, w, on = run('rand'); tb = h._tbase
print('RAND sampled on_ex_pen', round(on, 4), 'taus', float(tb['taus'].min()), float(tb['taus'].max()),
      'miss', float(tb['miss'].max()), 'spur', float(tb['spur'].max()), 'lags', sorted(set(tb['lag'].tolist())))
print('IDENTITY on_ex_pen', round(float((d['ig_contact'][np.concatenate([r[:L] for r in rows])][:, np.arange(32) != 11] >= 0.05).mean()), 4))
