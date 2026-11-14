"""Patch the falign student repo (97c4472) with two tactile-gap baselines for the student's tactile channel.

TACTILE_RAND=1: contact randomization. Per env and episode: per-channel threshold 0.05 x log-uniform[0.5, 4], miss
probability U[0, 0.5], spurious-contact probability U[0, 0.01], delay k uniform in {0, 1, 2} steps. Ranges are fixed a
priori (no SuperDex data).
TACTILE_CALIB=<npz with taus[32], lag>: per-channel threshold and delay fit on the SuperDex pairs
(tools/calib_thresh_delay.py in the superdex port dir).
Both feed the unchanged latency (0.2) / dropout (0.1) model with its own state; the teacher's privileged tactile stays
the raw Isaac Gym sensed contact. Unset = original behaviour. Exclusive with TACTILE_ALIGN_F.
"""
ENV = 'penspin/tasks/allegro_hand_hora.py'
s = open(ENV).read()
rep = [
("""            if self._tactile_align() is not None:
                student_contacts = self._aligned_contacts(contact_norm)""",
"""            if self._tactile_align() is not None:
                student_contacts = self._aligned_contacts(contact_norm)
            if self._tactile_baseline() is not None:
                student_contacts = self._baseline_contacts(contact_norm)"""),
("""    def pre_physics_step(self, actions):""",
"""    def _tactile_baseline(self):
        \"\"\"Student tactile baseline channel (TACTILE_RAND=1 or TACTILE_CALIB=<npz>), or None.\"\"\"
        if not hasattr(self, '_tbase'):
            self._tbase = None
            rand = os.environ.get('TACTILE_RAND') == '1'
            calib = os.environ.get('TACTILE_CALIB')
            if rand or calib:
                assert not (rand and calib) and not os.environ.get('TACTILE_ALIGN_F'), 'tactile channels are exclusive'
                n, dev = self.num_envs, self.device
                tb = {'rand': rand, 'hist': torch.zeros((n, 2, 32), device=dev),
                      'nhist': torch.zeros(n, dtype=torch.long, device=dev), 'last': torch.zeros((n, 32), device=dev)}
                if calib:
                    c = np.load(calib)
                    tb['taus'] = torch.tensor(c['taus'], dtype=torch.float, device=dev)[None].repeat(n, 1)
                    tb['lag'] = torch.full((n,), int(c['lag']), dtype=torch.long, device=dev)
                    print('TACTILE_CALIB', calib, 'lag', int(c['lag']), flush=True)
                else:
                    tb.update(taus=torch.zeros((n, 32), device=dev), lag=torch.zeros(n, dtype=torch.long, device=dev),
                              miss=torch.zeros((n, 1), device=dev), spur=torch.zeros((n, 1), device=dev))
                    self._tbase = tb
                    self._resample_tactile_rand(torch.ones(n, dtype=torch.bool, device=dev))
                    print('TACTILE_RAND thr x[0.5,4] miss[0,0.5] spur[0,0.01] lag{0,1,2}', flush=True)
                self._tbase = tb
        return self._tbase

    def _resample_tactile_rand(self, envs):
        tb = self._tbase
        m = int(envs.sum())
        if m:
            u = torch.rand((m, 32), device=self.device)
            tb['taus'][envs] = 0.05 * torch.exp(np.log(0.5) + u * np.log(8.0))
            tb['miss'][envs] = 0.5 * torch.rand((m, 1), device=self.device)
            tb['spur'][envs] = 0.01 * torch.rand((m, 1), device=self.device)
            tb['lag'][envs] = torch.randint(0, 3, (m,), device=self.device)

    def _baseline_contacts(self, contact_norm):
        tb = self._tbase
        at_reset = self.at_reset_buf.bool()  # same reset convention as _aligned_contacts
        tb['last'][at_reset] = 0.0
        k = tb['nhist'][:, None]
        h1 = torch.where(k >= 1, tb['hist'][:, 0], torch.zeros_like(contact_norm))
        h2 = torch.where(k >= 2, tb['hist'][:, 1], torch.zeros_like(contact_norm))
        lag = tb['lag'][:, None]
        norm = torch.where(lag == 0, contact_norm, torch.where(lag == 1, h1, h2))
        raw = (norm >= tb['taus']).float()
        if tb['rand']:
            raw = raw * (torch.rand_like(raw) >= tb['miss']).float()
            raw = torch.maximum(raw, (torch.rand_like(raw) < tb['spur']).float())
        tb['hist'] = torch.stack([contact_norm, h1], dim=1)
        tb['nhist'] = torch.clamp(tb['nhist'] + 1, max=2)
        tb['nhist'][at_reset] = 0  # the next step is step 0 of the new episode
        if tb['rand']:
            self._resample_tactile_rand(at_reset)
        latency = torch.where(torch.rand_like(raw) < self.latency, 1.0, 0.0)
        tb['last'] = tb['last'] * latency + raw * (1 - latency)
        mask = torch.where(torch.rand_like(raw) < self.sensor_noise, 0.0, 1.0)
        self.baseline_raw_contacts = raw
        return torch.where(tb['last'] > 0.1, mask * tb['last'], tb['last'])

    def pre_physics_step(self, actions):"""),
]
for a, b in rep:
    assert s.count(a) == 1, a
    s = s.replace(a, b)
assert '\nimport numpy as np' in s
open(ENV, 'w').write(s)
print('patched', ENV)
