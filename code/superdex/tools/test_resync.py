"""Checks needed by collect_pairs.py resync mode:
1) PenSpinSDX.reset interprets pen velocities in the world frame (as Isaac Gym root states are);
2) a reset fully determines the next control step: an env reused after a long rollout gives the same contacts,
   joints and pen pose as a fresh env reset to the same state."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sdx_env import PenSpinSDX  # noqa: E402
from teacher_np import quat_mul, quat_conj, quat_to_axis_angle  # noqa: E402

URDF = str(Path(__file__).resolve().parent.parent / 'assets/allegro/allegro_superdex.urdf')
r = dict(np.load(sys.argv[1])); names = [str(x) for x in r['dof_names']]
e = 0
env = PenSpinSDX(URDF, names, pen_scale=float(r['obj_scale'][e]), pen_mass=float(r['obj_mass'][e]), friction=1.0)
# 1) free pen far above the hand, long axis along world x (Ry(+90 deg)); world angvel (3,0,0) and linvel (0,1,0)
s = np.sin(np.pi / 4); qy = np.array([0, s, 0, s])
env.reset(r['init_q'][e], [0, 0, 2.0], qy, None, [0, 1.0, 0], [3.0, 0, 0])
p0 = env.pen_pose(); env.control_step(r['init_q'][e]); p1 = env.pen_pose()
rot = quat_to_axis_angle(quat_mul(p1[3:], quat_conj(p0[3:])))
print('FREE_PEN world rotation over 0.05 s', np.round(rot, 4), 'expected (0.15, 0, 0) if world frame')
print('FREE_PEN displacement', np.round(p1[:3] - p0[:3], 4), 'expected (0, 0.05, -0.0123)')
# 2) reset determinism
o = r['init_obj'][e]
env.reset(r['init_q'][e], o[:3], o[3:7], r['init_qd'][e], o[7:10], o[10:13])
for t in range(30):
    env.control_step(r['target'][e, t])
state = (env.q(), env.pen_pose())
qd_dummy = np.full(16, 0.3); lin, ang = [0.01, -0.02, 0.0], [0.5, -1.0, 2.0]
def probe(en):
    en.reset(state[0], state[1][:3], state[1][3:], qd_dummy, lin, ang)
    c = en.control_step(r['target'][e, 30])
    return c, en.q(), en.pen_pose()
c1, q1, p1 = probe(env)
fresh = PenSpinSDX(URDF, names, pen_scale=float(r['obj_scale'][e]), pen_mass=float(r['obj_mass'][e]), friction=1.0)
c2, q2, p2 = probe(fresh)
c3, q3, p3 = probe(env)
print('REUSED_vs_FRESH contact', float(np.abs(c1 - c2).max()), 'q', float(np.abs(q1 - q2).max()), 'pen', float(np.abs(p1 - p2).max()))
print('REUSED_twice contact', float(np.abs(c1 - c3).max()), 'q', float(np.abs(q1 - q3).max()), 'pen', float(np.abs(p1 - p3).max()))
print('contacts on', int((c1 >= 0.05).sum()))
