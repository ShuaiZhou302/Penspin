"""Add ENDS_NOISE_CM option (proprio_ends_gt only): i.i.d. Gaussian noise on all 18 endpoint-history inputs, resampled every
control step and every minibatch, scaled so the mean absolute per-coordinate error equals ENDS_NOISE_CM (comparable to B's
pose_l1_cm metric). Uses dedicated generators so global CPU/CUDA RNG streams (env randomness) are untouched. Default 0 = unchanged."""
import sys
from pathlib import Path
T = Path(sys.argv[1])

def patch(rel, pairs):
    p = T / rel; s = p.read_text()
    for old, new in pairs:
        assert s.count(old) == 1, (rel, old[:80], s.count(old))
        s = s.replace(old, new)
    p.write_text(s)

patch('dagger_train.py', [
    ('import json\nimport hashlib\n', 'import json\nimport hashlib\nimport math\n'),
    ("    pose_weight = float(os.environ.get('POSE_LOSS_WEIGHT', '1.0'))\n",
     "    pose_weight = float(os.environ.get('POSE_LOSS_WEIGHT', '1.0'))\n"
     "    ends_noise_cm = float(os.environ.get('ENDS_NOISE_CM', '0'))\n"
     "    assert ends_noise_cm >= 0 and (ends_noise_cm == 0 or input_mode == 'proprio_ends_gt'), ends_noise_cm\n"
     "    noise_sigma = ends_noise_cm / 100 / math.sqrt(2 / math.pi)\n"
     "    noise_gen = torch.Generator(device=agent.device)\n"
     "    noise_gen.manual_seed(24001)\n"
     "    def add_ends_noise(x):\n"
     "        if noise_sigma == 0:\n"
     "            return x\n"
     "        return x + noise_sigma * torch.randn(x.shape, generator=noise_gen, device=x.device)\n"
     "    if ends_noise_cm > 0:\n"
     "        probe = add_ends_noise(torch.zeros(4096, 3, 6, device=agent.device))\n"
     "        print('ENDS_NOISE', json.dumps({'target_mean_abs_cm': ends_noise_cm, 'sigma_m': noise_sigma,\n"
     "              'measured_mean_abs_cm': float(100 * probe.abs().mean())}), flush=True)\n"),
    ("                    student_in['obj_ends'] = obs['obj_ends']\n",
     "                    student_in['obj_ends'] = add_ends_noise(obs['obj_ends'])\n"),
    ("                batch_in['obj_ends'] = batch['ends']\n",
     "                batch_in['obj_ends'] = add_ends_noise(batch['ends'])\n"),
])
patch('protocol_student_diagnostic.py', [
    ('\n\ndef run(agent):\n',
     '\n\n_ENDS_NOISE = {}\n\n\n'
     'def _ends_noise(x):\n'
     '    """ENDS_NOISE_CM evaluation noise (same definition as dagger_train); dedicated generator, global RNG untouched."""\n'
     '    import math\n'
     '    cm = float(os.environ.get("ENDS_NOISE_CM", "0"))\n'
     '    if cm == 0:\n'
     '        return x\n'
     '    if "g" not in _ENDS_NOISE:\n'
     '        _ENDS_NOISE["g"] = torch.Generator(device=x.device)\n'
     '        _ENDS_NOISE["g"].manual_seed(24002)\n'
     '        print("EVAL_ENDS_NOISE_CM", cm, flush=True)\n'
     '    return x + cm / 100 / math.sqrt(2 / math.pi) * torch.randn(x.shape, generator=_ENDS_NOISE["g"], device=x.device)\n\n\n'
     'def run(agent):\n'),
    ('                inputs["obj_ends"] = obs["obj_ends"]\n',
     '                inputs["obj_ends"] = _ends_noise(obs["obj_ends"])\n'),
])
print('NOISE_PATCHED')
