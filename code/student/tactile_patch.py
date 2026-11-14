"""Add proprio_tactile_ends_est: endpoint estimator additionally reads the env's binary tactile history (30x32, official
latency/dropout sensor model) through its own TemporalTransformer; tactile reaches the policy ONLY via the 18-d endpoint
estimate. Policy input dim unchanged (154). Existing modes unchanged."""
import sys
from pathlib import Path
T = Path(sys.argv[1])

def patch(rel, pairs):
    p = T / rel; s = p.read_text()
    for old, new in pairs:
        assert s.count(old) == 1, (rel, old[:80], s.count(old))
        s = s.replace(old, new)
    p.write_text(s)

patch('penspin/algo/models/models.py', [
    ("        self.proprio_family = self.input_mode in ('proprio', 'proprio_ends_gt', 'proprio_ends_est')\n",
     "        self.proprio_family = self.input_mode in ('proprio', 'proprio_ends_gt', 'proprio_ends_est', 'proprio_tactile_ends_est')\n"),
    ("        if self.input_mode == 'proprio_ends_est' and self.student:\n"
     "            # Estimates the 3-frame endpoint history from the proprio transformer feature only.\n"
     "            self.ends_head = MLP(units=[64, 18], input_size=32, with_last_activation=False)\n",
     "        if self.input_mode == 'proprio_ends_est' and self.student:\n"
     "            # Estimates the 3-frame endpoint history from the proprio transformer feature only.\n"
     "            self.ends_head = MLP(units=[64, 18], input_size=32, with_last_activation=False)\n"
     "        if self.input_mode == 'proprio_tactile_ends_est' and self.student:\n"
     "            # Estimator also reads binary tactile history; tactile reaches the policy only through the estimate.\n"
     "            self.tactile_tconv = TemporalTransformer(32, 2, 2, 32, use_pe=True)\n"
     "            self.ends_head = MLP(units=[64, 18], input_size=64, with_last_activation=False)\n"),
    ("            if self.input_mode == 'proprio_ends_est':\n"
     "                ends_pred = self.ends_head(proprio_hidden).view(obs.shape[0], 3, 6)\n",
     "            if self.input_mode == 'proprio_ends_est':\n"
     "                ends_pred = self.ends_head(proprio_hidden).view(obs.shape[0], 3, 6)\n"
     "            elif self.input_mode == 'proprio_tactile_ends_est':\n"
     "                est_hidden = torch.cat([proprio_hidden, self.tactile_tconv(obs_dict['tactile_hist'])], dim=-1)\n"
     "                ends_pred = self.ends_head(est_hidden).view(obs.shape[0], 3, 6)\n"),
    ("            ends = ends_pred if self.input_mode == 'proprio_ends_est' else obs_dict['obj_ends']\n",
     "            ends = ends_pred if ends_pred is not None else obs_dict['obj_ends']\n"),
])

patch('dagger_train.py', [
    ("    assert input_mode in ('proprio', 'proprio_ends_gt', 'proprio_ends_est'), input_mode\n",
     "    assert input_mode in ('proprio', 'proprio_ends_gt', 'proprio_ends_est', 'proprio_tactile_ends_est'), input_mode\n"),
    ("    estimate = input_mode == 'proprio_ends_est'\n",
     "    estimate = input_mode in ('proprio_ends_est', 'proprio_tactile_ends_est')\n"
     "    use_tactile = input_mode == 'proprio_tactile_ends_est'\n"),
    ("        buffer['ends'] = torch.empty((bsize, 3, 6), device=agent.device)\n",
     "        buffer['ends'] = torch.empty((bsize, 3, 6), device=agent.device)\n"
     "    if use_tactile:\n"
     "        assert tuple(obs['tactile_hist'].shape) == (n, agent.proprio_len, 32), obs['tactile_hist'].shape\n"
     "        buffer['tactile'] = torch.empty((bsize, agent.proprio_len, 32), device=agent.device)\n"),
    ("                    student_in['obj_ends'] = add_ends_noise(obs['obj_ends'])\n",
     "                    student_in['obj_ends'] = add_ends_noise(obs['obj_ends'])\n"
     "                if use_tactile:\n"
     "                    student_in['tactile_hist'] = obs['tactile_hist']\n"),
    ("                    print('OBJ_ENDS_DEPENDENCY', json.dumps({'input_mode': input_mode, 'max_action_delta': delta}), flush=True)\n",
     "                    print('OBJ_ENDS_DEPENDENCY', json.dumps({'input_mode': input_mode, 'max_action_delta': delta}), flush=True)\n"
     "                    if use_tactile:\n"
     "                        flipped = dict(student_in); flipped['tactile_hist'] = 1 - obs['tactile_hist']\n"
     "                        tdelta = float((agent.model.act_inference(flipped)[0] - student).abs().max())\n"
     "                        assert tdelta > 0., 'tactile input unused'\n"
     "                        print('TACTILE_DEPENDENCY', json.dumps({'max_action_delta': tdelta,\n"
     "                              'tactile_on_fraction': float(obs['tactile_hist'].mean())}), flush=True)\n"),
    ("                    buffer['ends'][sl].copy_(obs['obj_ends'])\n",
     "                    buffer['ends'][sl].copy_(obs['obj_ends'])\n"
     "                if use_tactile:\n"
     "                    buffer['tactile'][sl].copy_(obs['tactile_hist'])\n"),
    ("                batch_in['obj_ends'] = add_ends_noise(batch['ends'])\n",
     "                batch_in['obj_ends'] = add_ends_noise(batch['ends'])\n"
     "            if use_tactile:\n"
     "                batch_in['tactile_hist'] = batch['tactile']\n"),
])

patch('protocol_student_diagnostic.py', [
    ('                inputs["obj_ends"] = _ends_noise(obs["obj_ends"])\n',
     '                inputs["obj_ends"] = _ends_noise(obs["obj_ends"])\n'
     '                if agent.input_mode == "proprio_tactile_ends_est":\n'
     '                    inputs["tactile_hist"] = obs["tactile_hist"]\n'),
    ('                    assert (d == 0.) == (agent.input_mode == "proprio_ends_est"), (agent.input_mode, d)\n',
     '                    assert (d == 0.) == agent.input_mode.endswith("ends_est"), (agent.input_mode, d)\n'),
])
print('TACTILE_PATCHED')
