"""Port check 1: model compiles; after reset to the recorded initial state, fingertip positions match Isaac Gym's recorded
fingertip positions (priv[16:28] of step-0 observation and ft_pos), pen pose round-trips, and q round-trips.
usage: test_mj_kin.py <record.npz> [n_eps]"""
import sys
import numpy as np
sys.path.insert(0, 'tools')
from mj_env import PenSpinMJ  # noqa: E402

r = dict(np.load(sys.argv[1])); n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
names = [str(x) for x in r['dof_names']]
pc = r['obj_point_cloud_local']; scale = r['obj_scale'].astype(float)
lens = np.round((pc[..., 2].max(1) - pc[..., 2].min(1)) / scale, 2) * scale
errs, qerr, perr = [], [], []
for e in range(n):
    env = PenSpinMJ(None, names, pen_scale=float(scale[e]), pen_mass=float(r['obj_mass'][e]), friction=1.0, pen_length=float(lens[e]))
    o = r['init_obj'][e]
    env.reset(r['init_q'][e], o[:3], o[3:7], r['init_qd'][e], o[7:10], o[10:13])
    ft = env.fingertips().ravel()
    errs.append(np.abs(ft - r['in_priv'][e, 0, 16:28]).max())
    qerr.append(np.abs(env.q() - r['init_q'][e]).max()); perr.append(np.abs(env.pen_pose()[:3] - o[:3]).max())
    if e == 0:
        m = env.model
        print('nbody', m.nbody, 'ngeom', m.ngeom, 'njnt', m.njnt, 'nmesh', m.nmesh, 'timestep', m.opt.timestep, 'pen mass', m.body_mass[env.pen_body])
        print('mj ft', np.round(ft, 4)); print('ig ft', np.round(r['in_priv'][e, 0, 16:28], 4))
        print('contacts at reset', env.data.ncon, 'contact norms', np.round(env.contact_norms(), 3))
print('fingertip max err m per ep', np.round(errs, 5)); print('q err', max(qerr), 'pen pos err', max(perr))
