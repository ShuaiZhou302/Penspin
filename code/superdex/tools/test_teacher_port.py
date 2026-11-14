"""Port checks on a no-observation-noise Isaac Gym record:
1) TeacherNP on the recorded raw inputs reproduces the recorded teacher actions;
2) TeacherObs driven by the recorded Isaac Gym states reproduces the recorded next-step raw inputs (obs96, the
   non-tactile priv dims, point cloud). Tactile goes through random latency/dropout and is checked separately."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from teacher_np import TeacherNP, TeacherObs  # noqa: E402

r = dict(np.load(sys.argv[1])); teacher = TeacherNP(sys.argv[2])
n, T = r['action'].shape[:2]; L = r['lengths']
# 1) network
worst = 0.0
for e in range(n):
    k = max(int(L[e]), 1)
    a = teacher.act(r['in_obs'][e, :k], r['in_priv'][e, :k], r['in_pc'][e, :k])
    worst = max(worst, float(np.abs(a - r['action'][e, :k]).max()))
print('TEACHER_ACTION_MAX_ABS_DIFF', worst)
# 2) observation builder
NONTAC = slice(0, 29)
errs = {'obs': 0.0, 'priv_nontactile': 0.0, 'pc': 0.0}
init_ok = 0.0
for e in range(n):
    k = int(L[e])
    if k < 3:
        continue
    init_pose = r['in_obs'][e, 1, :16]
    init_ok = max(init_ok, float(np.abs(r['in_obs'][e, 1] - np.tile(np.concatenate([init_pose, init_pose]), 3)).max()))
    ob = TeacherObs(init_pose, r['in_obs'][e, 0], r['in_priv'][e, 0], r['obj_point_cloud_local'][e],
                    float(r['obj_scale'][e]), float(r['obj_mass'][e]), float(r['obj_friction'][e]), float(r['obj_restitution'][e]),
                    np.random.default_rng(0), obs_noise_t=0.0, obs_noise_e=0.0)
    prev_quat = r['init_obj'][e, 3:7]
    for t in range(k - 1):
        pose = r['obj_pose'][e, t]
        o, p, pc, _ = ob.update(r['q'][e, t], r['target'][e, t], r['contact_norm'][e, t], r['ft_pos'][e, t].reshape(4, 3),
                                pose[:3], pose[3:7], prev_quat, noise_t=np.zeros(16))
        prev_quat = pose[3:7]
        errs['obs'] = max(errs['obs'], float(np.abs(o - r['in_obs'][e, t + 1]).max()))
        errs['priv_nontactile'] = max(errs['priv_nontactile'], float(np.abs(p[NONTAC] - r['in_priv'][e, t + 1, NONTAC]).max()))
        errs['pc'] = max(errs['pc'], float(np.abs(pc - r['in_pc'][e, t + 1]).max()))
print('INIT_HISTORY_FRAMES_MAX_ABS_DIFF', init_ok)
print('OBS_BUILDER_MAX_ABS_DIFF', errs)
# tactile statistics: recorded sensed vs binary raw contacts
raw = (r['contact_norm'] >= 0.05).astype(float)
sensed = r['in_priv'][:, 1:, 29:61]
valid = np.arange(T - 1)[None, :] < (L[:, None] - 1)
print('TACTILE_ON_RAW_vs_SENSED', float(raw[:, :-1][valid].mean()), float(sensed[valid].mean()))
