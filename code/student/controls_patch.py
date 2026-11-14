"""Two controls for proprio_tactile_ends_est:
proprio_tactile_ends_est_sg: identical architecture, but the endpoint estimate is detached before the policy, so the
  estimator (ends_head + tactile_tconv) is trained only by the pose loss; tactile can inform the policy only as a pose estimate.
proprio_extra_ends_est: B plus an extra TemporalTransformer(32,2,2,32) reading proprio_hist (instead of tactile) into the
  same MLP[64,18] head -> identical parameter count to the tactile variant, no tactile input.
Existing modes unchanged."""
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
    ("        self.proprio_family = self.input_mode in ('proprio', 'proprio_ends_gt', 'proprio_ends_est', 'proprio_tactile_ends_est')\n",
     "        self.proprio_family = self.input_mode in ('proprio', 'proprio_ends_gt', 'proprio_ends_est', 'proprio_tactile_ends_est',\n"
     "                                                  'proprio_tactile_ends_est_sg', 'proprio_extra_ends_est')\n"),
    ("        if self.input_mode == 'proprio_tactile_ends_est' and self.student:\n",
     "        if self.input_mode in ('proprio_tactile_ends_est', 'proprio_tactile_ends_est_sg') and self.student:\n"),
    ("            self.ends_head = MLP(units=[64, 18], input_size=64, with_last_activation=False)\n",
     "            self.ends_head = MLP(units=[64, 18], input_size=64, with_last_activation=False)\n"
     "        if self.input_mode == 'proprio_extra_ends_est' and self.student:\n"
     "            # Equal-capacity control: second proprio transformer in place of the tactile encoder.\n"
     "            self.extra_tconv = TemporalTransformer(32, 2, 2, 32, use_pe=True)\n"
     "            self.ends_head = MLP(units=[64, 18], input_size=64, with_last_activation=False)\n"),
    ("            elif self.input_mode == 'proprio_tactile_ends_est':\n"
     "                est_hidden = torch.cat([proprio_hidden, self.tactile_tconv(obs_dict['tactile_hist'])], dim=-1)\n"
     "                ends_pred = self.ends_head(est_hidden).view(obs.shape[0], 3, 6)\n",
     "            elif self.input_mode in ('proprio_tactile_ends_est', 'proprio_tactile_ends_est_sg'):\n"
     "                est_hidden = torch.cat([proprio_hidden, self.tactile_tconv(obs_dict['tactile_hist'])], dim=-1)\n"
     "                ends_pred = self.ends_head(est_hidden).view(obs.shape[0], 3, 6)\n"
     "            elif self.input_mode == 'proprio_extra_ends_est':\n"
     "                est_hidden = torch.cat([proprio_hidden, self.extra_tconv(obs_dict['proprio_hist'])], dim=-1)\n"
     "                ends_pred = self.ends_head(est_hidden).view(obs.shape[0], 3, 6)\n"),
    ("            ends = ends_pred if ends_pred is not None else obs_dict['obj_ends']\n",
     "            if ends_pred is None:\n"
     "                ends = obs_dict['obj_ends']\n"
     "            else:\n"
     "                ends = ends_pred.detach() if self.input_mode.endswith('_sg') else ends_pred\n"),
])

patch('dagger_train.py', [
    ("    assert input_mode in ('proprio', 'proprio_ends_gt', 'proprio_ends_est', 'proprio_tactile_ends_est'), input_mode\n",
     "    assert input_mode in ('proprio', 'proprio_ends_gt', 'proprio_ends_est', 'proprio_tactile_ends_est',\n"
     "                          'proprio_tactile_ends_est_sg', 'proprio_extra_ends_est'), input_mode\n"),
    ("    estimate = input_mode in ('proprio_ends_est', 'proprio_tactile_ends_est')\n"
     "    use_tactile = input_mode == 'proprio_tactile_ends_est'\n",
     "    estimate = input_mode in ('proprio_ends_est', 'proprio_tactile_ends_est', 'proprio_tactile_ends_est_sg', 'proprio_extra_ends_est')\n"
     "    use_tactile = input_mode in ('proprio_tactile_ends_est', 'proprio_tactile_ends_est_sg')\n"),
    ("                loss = bc_loss + pose_weight * pose_loss\n",
     "                loss = bc_loss + pose_weight * pose_loss\n"
     "                if input_mode.endswith('_sg') and iteration == start_iteration+1 and not pose_losses:\n"
     "                    head = list(agent.model.ends_head.parameters()) + list(agent.model.tactile_tconv.parameters())\n"
     "                    g = torch.autograd.grad(bc_loss, head, retain_graph=True, allow_unused=True)\n"
     "                    gn = sum(float(x.abs().sum()) for x in g if x is not None)\n"
     "                    assert gn == 0., ('stop-gradient leak', gn)\n"
     "                    gp = torch.autograd.grad(pose_loss, head, retain_graph=True, allow_unused=True)\n"
     "                    gpn = sum(float(x.abs().sum()) for x in gp if x is not None)\n"
     "                    assert gpn > 0.\n"
     "                    print('SG_GRADIENT_CHECK', json.dumps({'bc_to_estimator': gn, 'pose_to_estimator': gpn}), flush=True)\n"),
])

patch('protocol_student_diagnostic.py', [
    ('                if agent.input_mode == "proprio_tactile_ends_est":\n',
     '                if "tactile" in agent.input_mode:\n'),
    ('                    assert (d == 0.) == agent.input_mode.endswith("ends_est"), (agent.input_mode, d)\n',
     '                    assert (d == 0.) == ("ends_est" in agent.input_mode), (agent.input_mode, d)\n'),
])
print('CONTROLS_PATCHED')
