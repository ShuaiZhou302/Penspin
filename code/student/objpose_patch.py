"""Patch a fresh copy of the strict pure-DAgger source to add object-endpoint student variants.
proprio_ends_gt : student actor additionally gets simulator pen endpoints (official obj_ends, +/-radius noise, 3 frames).
proprio_ends_est: student predicts those endpoints from its own proprio history (aux L1 loss) and feeds the prediction;
                  it never reads obj_ends as input. 'proprio' path is unchanged."""
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
    ("        self.student = kwargs.get('student', False)\n",
     "        self.student = kwargs.get('student', False)\n"
     "        self.proprio_family = self.input_mode in ('proprio', 'proprio_ends_gt', 'proprio_ends_est')\n"),
    ("            self.end_feat_extractor = MLP(units=[6,6,6], input_size=3, with_last_activation=False)\n",
     "            self.end_feat_extractor = MLP(units=[6,6,6], input_size=3, with_last_activation=False)\n"
     "        if self.input_mode == 'proprio_ends_est' and self.student:\n"
     "            # Estimates the 3-frame endpoint history from the proprio transformer feature only.\n"
     "            self.ends_head = MLP(units=[64, 18], input_size=32, with_last_activation=False)\n"),
    ("        if self.student and self.input_mode != 'proprio':\n",
     "        if self.student and not self.proprio_family:\n"),
    ("        if self.proprio_mode and self.student:\n"
     "            proprio_feat = self.adapt_tconv(obs_dict['proprio_hist'])\n"
     "            proprio_feat = self.all_fuse(proprio_feat)\n",
     "        ends_pred = None\n"
     "        if self.proprio_mode and self.student:\n"
     "            proprio_hidden = self.adapt_tconv(obs_dict['proprio_hist'])\n"
     "            if self.input_mode == 'proprio_ends_est':\n"
     "                ends_pred = self.ends_head(proprio_hidden).view(obs.shape[0], 3, 6)\n"
     "            proprio_feat = self.all_fuse(proprio_hidden)\n"),
    ("            point_feat_1 = self.end_feat_extractor(obs_dict['obj_ends'][..., :3]).unsqueeze(-1)\n"
     "            point_feat_2 = self.end_feat_extractor(obs_dict['obj_ends'][..., 3:]).unsqueeze(-1)\n",
     "            ends = ends_pred if self.input_mode == 'proprio_ends_est' else obs_dict['obj_ends']\n"
     "            point_feat_1 = self.end_feat_extractor(ends[..., :3]).unsqueeze(-1)\n"
     "            point_feat_2 = self.end_feat_extractor(ends[..., 3:]).unsqueeze(-1)\n"),
    ("        elif self.input_mode == 'proprio':\n",
     "        elif self.proprio_family:\n"),
    ("            extrin_gt = extrin.detach()\n",
     "            extrin_gt = ends_pred.reshape(obs.shape[0], -1) if ends_pred is not None else extrin.detach()\n"),
])

patch('dagger_train.py', [
    ('import torch\n\n\nclass Reservoir:',
     'import torch\n\n\n'
     'def ends_l1_cm(pred, target):\n'
     '    """Mean per-coordinate L1 (cm) over 3 frames, symmetric in endpoint order."""\n'
     '    pred, target = pred.reshape(-1, 3, 6), target.reshape(-1, 3, 6)\n'
     '    a = (pred - target).abs().mean(-1)\n'
     '    b = (pred - torch.cat([target[..., 3:], target[..., :3]], -1)).abs().mean(-1)\n'
     '    return 100 * torch.minimum(a, b).mean()\n\n\n'
     'class Reservoir:'),
    ("    assert mode in ('dagger', 'bc')\n",
     "    assert mode in ('dagger', 'bc')\n"
     "    input_mode = agent.input_mode\n"
     "    assert input_mode in ('proprio', 'proprio_ends_gt', 'proprio_ends_est'), input_mode\n"
     "    use_ends = input_mode != 'proprio'\n"
     "    estimate = input_mode == 'proprio_ends_est'\n"
     "    pose_weight = float(os.environ.get('POSE_LOSS_WEIGHT', '1.0'))\n"
     "    print('OBJPOSE_CONFIG', json.dumps({'input_mode': input_mode, 'pose_loss_weight': pose_weight if estimate else None}), flush=True)\n"),
    ("        'target': torch.empty((bsize, agent.actions_num), device=agent.device),\n    }\n",
     "        'target': torch.empty((bsize, agent.actions_num), device=agent.device),\n    }\n"
     "    if use_ends:\n"
     "        assert tuple(obs['obj_ends'].shape) == (n, 3, 6), obs['obj_ends'].shape\n"
     "        buffer['ends'] = torch.empty((bsize, 3, 6), device=agent.device)\n"),
    ("        rollout_error, action_delta = 0., 0.\n",
     "        rollout_error, action_delta = 0., 0.\n        rollout_pose = 0.\n"),
    ("                student_in = {'obs': agent.running_mean_std(obs['obs']), 'proprio_hist': history}\n"
     "                student, student_extrin, _ = agent.model.act_inference(student_in)\n",
     "                student_in = {'obs': agent.running_mean_std(obs['obs']), 'proprio_hist': history}\n"
     "                if use_ends:\n"
     "                    student_in['obj_ends'] = obs['obj_ends']\n"
     "                student, student_extrin, student_aux = agent.model.act_inference(student_in)\n"
     "                if iteration == start_iteration+1 and t == 0 and use_ends:\n"
     "                    moved = dict(student_in); moved['obj_ends'] = obs['obj_ends'] + 0.05\n"
     "                    delta = float((agent.model.act_inference(moved)[0] - student).abs().max())\n"
     "                    if estimate:\n"
     "                        assert delta == 0., ('estimator variant read simulator obj_ends', delta)\n"
     "                    else:\n"
     "                        assert delta > 0., 'obj_ends input unused'\n"
     "                    print('OBJ_ENDS_DEPENDENCY', json.dumps({'input_mode': input_mode, 'max_action_delta': delta}), flush=True)\n"
     "                if estimate:\n"
     "                    rollout_pose += float(ends_l1_cm(student_aux, obs['obj_ends']))\n"),
    ("                buffer['target'][sl].copy_(teacher)\n",
     "                buffer['target'][sl].copy_(teacher)\n"
     "                if use_ends:\n"
     "                    buffer['ends'][sl].copy_(obs['obj_ends'])\n"),
    ("        losses = []\n",
     "        losses, pose_losses = [], []\n"),
    ("            mu, _, _, _, _ = agent.model._actor_critic({\n"
     "                'obs': agent.running_mean_std(batch['obs']), 'proprio_hist': batch['proprio']})\n"
     "            loss = (mu.clamp(-1,1)-batch['target']).abs().sum(-1).mean()\n",
     "            batch_in = {'obs': agent.running_mean_std(batch['obs']), 'proprio_hist': batch['proprio']}\n"
     "            if use_ends:\n"
     "                batch_in['obj_ends'] = batch['ends']\n"
     "            mu, _, _, _, aux = agent.model._actor_critic(batch_in)\n"
     "            bc_loss = (mu.clamp(-1,1)-batch['target']).abs().sum(-1).mean()\n"
     "            loss = bc_loss\n"
     "            if estimate:\n"
     "                pose_loss = ends_l1_cm(aux, batch['ends'])\n"
     "                loss = bc_loss + pose_weight * pose_loss\n"
     "                pose_losses.append(float(pose_loss.detach()))\n"),
    ("            losses.append(float(loss.detach()))\n",
     "            losses.append(float(bc_loss.detach()))\n"),
    ("               'loss_l1_sum':sum(losses)/len(losses), ",
     "               'loss_l1_sum':sum(losses)/len(losses), 'input_mode':input_mode,\n"
     "               'pose_l1_cm':sum(pose_losses)/len(pose_losses) if pose_losses else None,\n"
     "               'rollout_pose_l1_cm':rollout_pose/horizon if estimate else None, "),
])

patch('protocol_student_diagnostic.py', [
    ('            action, extrin, _ = agent.model.act_inference(inputs)\n',
     '            if agent.input_mode != "proprio":\n'
     '                inputs["obj_ends"] = obs["obj_ends"]\n'
     '                if step == 0:\n'
     '                    moved = dict(inputs); moved["obj_ends"] = obs["obj_ends"] + 0.05\n'
     '                    d = float((agent.model.act_inference(moved)[0] - agent.model.act_inference(inputs)[0]).abs().max())\n'
     '                    assert (d == 0.) == (agent.input_mode == "proprio_ends_est"), (agent.input_mode, d)\n'
     '                    print("EVAL_OBJ_ENDS_DEPENDENCY", agent.input_mode, d, flush=True)\n'
     '            action, extrin, _ = agent.model.act_inference(inputs)\n'),
])
print('PATCHED')
