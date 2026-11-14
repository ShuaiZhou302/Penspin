"""The env-side f channel (AllegroHandHora._tactile_align/_aligned_contacts) reproduces train_f predictions.
Drives the two methods on a stand-in object with the dataset's per-step Isaac Gym values for several val episodes
(batched as envs; each episode is preceded by a reset call with stale inputs) and compares the binary f output with
FNet(train_f.features(...)) > 0.
usage: test_falign_port.py <dataset.npz> <f.pt> <superdex tools dir>"""
import os
import sys
import types
import numpy as np
import torch

ds, fpath, tools = sys.argv[1:4]
sys.path.insert(0, tools)
from train_f import FNet, features  # noqa: E402

src = open('penspin/tasks/allegro_hand_hora.py').read()
mod = types.ModuleType('m'); mod.os = os; mod.torch = torch
start = src.index('    def _tactile_align(self):'); end = src.index('    def pre_physics_step(self, actions):')
body = 'class H:\n' + src[start:end]
exec(body, mod.__dict__)
H = mod.H

d = dict(np.load(ds))
ck = torch.load(fpath, map_location='cpu')
net = FNet(ck['din'], np.zeros(ck['din'], np.float32), np.ones(ck['din'], np.float32)); net.load_state_dict(ck['state_dict'])
eps = list(dict.fromkeys(d['episode'][d['split'] == 'val'].tolist()))
rows = [r for r in (np.nonzero(d['episode'] == e)[0] for e in eps) if len(r) >= 200][:16]
eps = eps[:len(rows)]
L = min(len(r) for r in rows)
os.environ['TACTILE_ALIGN_F'] = fpath
h = H(); h.num_envs = len(eps); h.device = 'cpu'; h.latency = 0.2; h.sensor_noise = 0.1
h.at_reset_buf = torch.ones(h.num_envs)
h.allegro_hand_dof_pos = torch.randn(h.num_envs, 16); h.cur_targets = torch.randn(h.num_envs, 16); h.actions = torch.randn(h.num_envs, 16)
h._tactile_align(); h._aligned_contacts(torch.rand(h.num_envs, 32))  # reset call with stale inputs
h.at_reset_buf = torch.zeros(h.num_envs)
worst = 0; n = 0
with torch.no_grad():
    for t in range(L):
        idx = np.array([r[t] for r in rows])
        h.allegro_hand_dof_pos = torch.from_numpy(d['q'][idx]).float(); h.cur_targets = torch.from_numpy(d['target'][idx]).float()
        h.actions = torch.from_numpy(d['action'][idx]).float()
        h._aligned_contacts(torch.from_numpy(d['ig_contact'][idx]).float())
        ref = (net(torch.from_numpy(features(d, idx))) > 0).float()
        worst = max(worst, int((h.aligned_raw_contacts != ref).sum())); n += ref.numel()
print('FALIGN_PORT mismatching binary outputs (max per step):', worst, 'checked', n, 'steps', L, 'envs', len(eps))
print('FALIGN_ON_RATE', float(h.aligned_raw_contacts.mean()))
