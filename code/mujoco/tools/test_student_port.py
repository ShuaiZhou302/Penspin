"""Port checks for a student-policy Isaac Gym record (obs noise on):
1) StudentPolicy on the recorded raw inputs reproduces the recorded student actions;
2) history mechanics of TeacherObs match Isaac Gym: initial frames, one-frame shift, newest frame = [noisy q, target],
   newest tactile frame = priv tactile, and the noise residual has the configured per-episode + per-step scales."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from student_policy import StudentPolicy  # noqa: E402

r = dict(np.load(sys.argv[1])); pol = StudentPolicy(sys.argv[2], sys.argv[3])
n, T = r['action'].shape[:2]; L = r['lengths']
worst, checked = 0.0, 0
for e in range(0, n, 4):
    for t in range(0, max(int(L[e]), 1), 7):
        a = pol.act(r['in_obs'][e, t], r['in_proprio'][e, t], r['in_tactile'][e, t])
        worst = max(worst, float(np.abs(a - r['action'][e, t]).max())); checked += 1
print('STUDENT_ACTION_MAX_ABS_DIFF', worst, 'over', checked, 'steps')
P, Tc = r['in_proprio'], r['in_tactile']
init = np.concatenate([P[:, 0, 0, :16], P[:, 0, 0, :16]], -1)
print('INIT_PROPRIO_TILED', float(np.abs(P[:, 0] - init[:, None]).max()), 'INIT_TACTILE_ZERO', float(np.abs(Tc[:, 0]).max()))
print('INIT_POSE_EQ_OBS1', float(np.abs(P[:, 0, 0, :16] - r['in_obs'][:, 1, :16]).max()))
shift = target = tac = 0.0; resid = []
for e in range(n):
    k = int(L[e])
    for t in range(k - 1):
        shift = max(shift, float(np.abs(P[e, t + 1, :-1] - P[e, t, 1:]).max()), float(np.abs(Tc[e, t + 1, :-1] - Tc[e, t, 1:]).max()))
        target = max(target, float(np.abs(P[e, t + 1, -1, 16:] - r['target'][e, t]).max()))
        tac = max(tac, float(np.abs(Tc[e, t + 1, -1] - r['in_priv'][e, t + 1, 29:61]).max()))
    if k > 5:
        resid.append(P[e, 1:k, -1, :16] - r['q'][e, :k - 1])
print('SHIFT_MAX', shift, 'NEWEST_TARGET_MAX', target, 'NEWEST_TACTILE_EQ_PRIV_MAX', tac)
per_ep = np.array([x.mean(0) for x in resid]); per_step = np.concatenate([x - x.mean(0) for x in resid])
print('NOISE_STD per-episode %.4f (cfg 0.02)  per-step %.4f (cfg 0.01)' % (per_ep.std(), per_step.std()))
