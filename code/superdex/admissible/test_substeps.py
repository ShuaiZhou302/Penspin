"""Free-hand check of SDX_SUBSTEPS: pen parked far from the hand, same targets; prints the joint trajectory so runs with
different SDX_SUBSTEPS can be compared. If the external torque accumulated over substeps, trajectories would scale with n.
usage: SDX_SUBSTEPS=n test_substeps.py <record.npz> <out.npy>"""
import sys
import numpy as np
sys.path.insert(0, "tools")
from sdx_env import PenSpinSDX, IG_LOWER, IG_UPPER, SUBSTEPS  # noqa: E402
from closed_loop import URDF  # noqa: E402

r = dict(np.load(sys.argv[1])); names = [str(x) for x in r['dof_names']]
env = PenSpinSDX(URDF, names, pen_scale=float(r['obj_scale'][0]), pen_mass=float(r['obj_mass'][0]), friction=1.0)
q0 = r['init_q'][0]
env.reset(q0, [0.5, 0.5, 0.05], [0, 0, 0, 1], np.zeros(16))
lo, hi = np.asarray(IG_LOWER), np.asarray(IG_UPPER)
traj, taus = [], []
for t in range(40):
    target = np.clip(q0 + 0.3 * np.sin(0.2 * t + np.arange(16)), lo, hi)
    env.q_prev = env.q()
    for _ in range(10):
        taus.append(env.physics_step(target))
    traj.append(env.q())
traj = np.array(traj); np.save(sys.argv[2], traj)
print('SUBSTEPS', SUBSTEPS, 'q[10]', np.round(traj[10, :6], 5), 'q[39]', np.round(traj[39, :6], 5), 'tau_absmax', np.abs(taus).max())
