"""Patch the objpose student repo (fa99bdc) with the contact-alignment tactile channel.

TACTILE_ALIGN_F=<f checkpoint> (tools/train_f.py format): the student's tactile history is built from f's prediction
of SuperDex binary contact (inputs: log1p(|F_IG|/0.05) at t, t-1, t-2, q_t, q_{t-1}, target_t, clamped action_t; the
same feature order as train_f.features), passed through the unchanged latency (0.2) / dropout (0.1) model with its own
state. The teacher's privileged tactile stays the raw Isaac Gym sensed contact. Unset = original behaviour.
"""
import sys

ENV = 'penspin/tasks/allegro_hand_hora.py'
s = open(ENV).read()
rep = [
("""            contacts = torch.norm(contacts, dim=-1)
            contacts = torch.where(contacts >= self.contact_thresh, 1.0, 0.0)""",
"""            contacts = torch.norm(contacts, dim=-1)
            contact_norm = contacts.clone()
            contacts = torch.where(contacts >= self.contact_thresh, 1.0, 0.0)"""),
("""            self.sensed_contacts = sensed_contacts
            if self.viewer:""",
"""            self.sensed_contacts = sensed_contacts
            student_contacts = sensed_contacts
            if self._tactile_align() is not None:
                student_contacts = self._aligned_contacts(contact_norm)
            if self.viewer:"""),
("""            cur_obs_buf = torch.cat([cur_obs_buf, sensed_contacts.unsqueeze(1)], dim=-1) # [1, 1, 32+12+32]""",
"""            cur_obs_buf = torch.cat([cur_obs_buf, student_contacts.unsqueeze(1)], dim=-1) # [1, 1, 32+12+32]"""),
("""    def pre_physics_step(self, actions):""",
"""    def _tactile_align(self):
        \"\"\"Lazily loaded contact alignment model f (TACTILE_ALIGN_F), or None.\"\"\"
        if not hasattr(self, '_falign'):
            self._falign = None
            path = os.environ.get('TACTILE_ALIGN_F')
            if path:
                ck = torch.load(path, map_location=self.device)
                sd = ck['state_dict']
                net = torch.nn.Sequential(torch.nn.Linear(ck['din'], 256), torch.nn.ELU(), torch.nn.Linear(256, 256),
                                          torch.nn.ELU(), torch.nn.Linear(256, 32)).to(self.device)
                net.load_state_dict({k[len('mlp.'):]: v for k, v in sd.items() if k.startswith('mlp.')})
                net.eval()
                assert ck['din'] == 3 * 32 + 4 * 16, ck['din']
                n = self.num_envs
                self._falign = {'net': net, 'mean': sd['mean'].to(self.device), 'std': sd['std'].to(self.device),
                                'chist': torch.zeros((n, 2, 32), device=self.device),
                                'qprev': torch.zeros((n, 16), device=self.device),
                                'nprev': torch.zeros(n, dtype=torch.long, device=self.device),
                                'last': torch.zeros((n, 32), device=self.device)}
                print('TACTILE_ALIGN_F', path, 'budget', ck.get('budget'), 'seed', ck.get('seed'), flush=True)
        return self._falign

    def _aligned_contacts(self, contact_norm):
        fa = self._falign
        at_reset = self.at_reset_buf.bool()  # envs reset in this step's post_physics_step (mirrors last_contacts reset)
        fa['last'][at_reset] = 0.0
        c = torch.log1p(contact_norm / 0.05)
        k = fa['nprev'][:, None]
        h1 = torch.where(k >= 1, fa['chist'][:, 0], torch.zeros_like(c))
        h2 = torch.where(k >= 2, fa['chist'][:, 1], torch.zeros_like(c))
        q = self.allegro_hand_dof_pos
        qp = torch.where(k >= 1, fa['qprev'], q)
        act = getattr(self, 'actions', None)
        act = torch.zeros_like(q) if act is None else act.clamp(-1.0, 1.0)
        x = torch.cat([c, h1, h2, q, qp, self.cur_targets, act], dim=-1)
        with torch.no_grad():
            pred = (fa['net']((x - fa['mean']) / fa['std']) > 0).float()
        fa['chist'] = torch.stack([c, h1], dim=1)
        fa['qprev'] = q.clone()
        fa['nprev'] = torch.clamp(fa['nprev'] + 1, max=2)
        fa['nprev'][at_reset] = 0  # the next step is step 0 of the new episode
        latency = torch.where(torch.rand_like(pred) < self.latency, 1.0, 0.0)
        fa['last'] = fa['last'] * latency + pred * (1 - latency)
        mask = torch.where(torch.rand_like(pred) < self.sensor_noise, 0.0, 1.0)
        self.aligned_raw_contacts = pred
        return torch.where(fa['last'] > 0.1, mask * fa['last'], fa['last'])

    def pre_physics_step(self, actions):"""),
]
for a, b in rep:
    assert s.count(a) == 1, a
    s = s.replace(a, b)
open(ENV, 'w').write(s)
print('patched', ENV)
