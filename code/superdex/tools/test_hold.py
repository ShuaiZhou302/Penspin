"""Static-hold check: grasp-cache states (scale 0.29) held with target = initial q for 2 s in SuperDex."""
import sys, time, json, numpy as np
sys.path.insert(0, sys.argv[1])
from sdx_env import PenSpinSDX
IG = [f'joint_{i}.0' for i in (0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 8, 9, 10, 11)]
cache = np.load('<PROJECT_ROOT>/penspin-student-objpose-dagger-20261022/cache/allegro_round_tip_thin/pencil/s029_50k.npy')
env = PenSpinSDX(sys.argv[1] + '/../assets/allegro/allegro_superdex.urdf', IG, pen_scale=0.29, pen_mass=0.015)
for row in range(int(sys.argv[2])):
    c = cache[row]; q0, pos, quat = c[:16], c[16:19], c[19:23]
    env.reset(q0, pos, quat); p0 = env.pen_pose(); t = time.time(); nz = []
    for k in range(40):
        f = env.control_step(q0); nz.append(int((f >= 0.05).sum()))
    p1 = env.pen_pose(); zmin, zmax = env.pen_ends_z()
    print(json.dumps({'row': row, 'pen_disp_cm': round(100 * float(np.linalg.norm(p1[:3] - p0[:3])), 3),
                      'q_err_max': round(float(np.abs(env.q() - q0).max()), 4), 'ends_z': [round(zmin, 4), round(zmax, 4)],
                      'fell': bool(zmin < 0.56 or zmax > 0.71), 'sensors_on_first_last': [nz[0], nz[-1]], 'wall_s': round(time.time() - t, 2)}), flush=True)
