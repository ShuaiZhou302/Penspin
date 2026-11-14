"""Which SuperDex actors does the pen touch at resync steps where Isaac Gym reports fingertip-FSR contact?
Lists pen contact points grouped by the other actor's name (force norm, point count) and the fsr links'
own parent/joint setup in the prefab."""
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sdx_env import PenSpinSDX, SENSOR_NAMES  # noqa: E402
import superdex.robotics as sdr  # noqa: E402

URDF = str(Path(__file__).resolve().parent.parent / 'assets/allegro/allegro_superdex.urdf')
r = dict(np.load(sys.argv[1])); names = [str(x) for x in r['dof_names']]
pf = sdr.load_bot_prefab_from_urdf_file(URDF)
for i in range(len(pf.joints)):
    j = pf.joints[i]
    if 'fsr' in j.name or 'tip' in j.name:
        print('JOINT', j.name, str(j.type))
e = 0
env = PenSpinSDX(URDF, names, pen_scale=float(r['obj_scale'][e]), pen_mass=float(r['obj_mass'][e]), friction=1.0)
names_by_handle = {}
for h in env.hand.get_nested_link_actors():
    a = env.scene.get_actor(h); names_by_handle[h.value] = a.get_name()
print('NESTED_ACTORS', len(names_by_handle), sorted(n.split('/')[-1] for n in names_by_handle.values())[:60])
tips = [i for i, n in enumerate(SENSOR_NAMES) if 'tip_fsr' in n]
L = int(r['lengths'][e]); shown = 0
for t in range(1, L):
    ig_on = [SENSOR_NAMES[i] for i in tips if r['contact_norm'][e, t, i] >= 0.05]
    if not ig_on:
        continue
    p, v = r['obj_pose'][e, t - 1], r['obj_vel'][e, t - 1]
    env.reset(r['q'][e, t - 1], p[:3], p[3:7], r['qd'][e, t - 1], v[:3], v[3:6])
    env.control_step(r['target'][e, t])
    by = defaultdict(lambda: [0, np.zeros(3)])
    for c in env.pen.get_contact_points_world():
        a, b = c.actor_a.value, c.actor_b.value
        other = b if a == env.pen_key else a
        nm = names_by_handle.get(other, "nonhand:%d" % other)
        by[nm][0] += 1; by[nm][1] += [c.force[0], c.force[1], c.force[2]]
    print('t', t, 'IG_on', ig_on)
    print('   SDX', {k.split('/')[-1]: (n, round(float(np.linalg.norm(f)), 3)) for k, (n, f) in by.items()})
    shown += 1
    if shown >= 6:
        break
    print('   SDX contact_norms on', [SENSOR_NAMES[i] for i in np.nonzero(env.contact_norms() >= 0.05)[0]])
