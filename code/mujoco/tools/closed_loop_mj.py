"""[MuJoCo copy: closed_loop_mj.py, env = mj_env.PenSpinMJ; sdx_* keys hold the MuJoCo values.]
Closed-loop teacher in SuperDex from the initial states of an Isaac Gym record, compared with the teacher's own
Isaac Gym episodes in that record.

usage: closed_loop.py <record.npz> <teacher.npz> <out_prefix> [max_eps|all] [k=v ...]
  options: inertia=mesh|ig  friction=<mu or 'rec'>  noise_t=<float> noise_e=<float>  seed=<int>
           policy=teacher|student  ckpt=<student checkpoint>  mode=<student input_mode>
Rotation metric = sum over steps of clip(angvel . z, -0.5, 0.5) (the Isaac Gym rotation reward; the recorded
'rot_rad' is angvel . z * 0.05). Termination uses the Isaac Gym rules. The episode cap is the record's T (400).
"""
import json
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mj_env import PenSpinMJ as PenSpinSDX, IG_LOWER, IG_UPPER, CONTACT_THRESH, CONTACT_OVERRIDE, JOINT_OVERRIDE, SUBSTEPS, OPT  # noqa: E402
from teacher_np import TeacherNP, TeacherObs, ACTION_SCALE, CONTROL_DT, FINGERTIPS  # noqa: E402

URDF = None  # mj_env uses the Isaac Gym URDF
RESET_Z, END_Z = 0.605, (0.56, 0.71)
KEEP = [i for i in range(32) if i != 11]  # contact channels except the IG index -1 pen-total quirk


def main(record, teacher_npz, out_prefix, max_eps='all', **opt):
    r = dict(np.load(record)); teacher = TeacherNP(teacher_npz)
    names = [str(x) for x in r['dof_names']]
    n = len(r['lengths']) if max_eps == 'all' else min(int(max_eps), len(r['lengths']))
    T = r['target'].shape[1]
    inertia = opt.get('inertia', 'mesh'); fr = opt.get('friction', 'rec')
    rng = np.random.default_rng(int(opt.get('seed', 0)))
    student = None
    if opt.get('policy', 'teacher') == 'student':
        from student_policy import StudentPolicy
        student = StudentPolicy(opt['ckpt'], opt['mode'])
    pen = {}
    if opt.get('penlen') == 'ig':
        # per-env asset length from the record point cloud; termination uses the Isaac Gym scalar (last asset length x last env scale)
        pc = r['obj_point_cloud_local']
        scale = r['obj_scale'].astype(float); ratio = np.round((pc[..., 2].max(1) - pc[..., 2].min(1)) / scale, 2)
        assert set(ratio.tolist()) <= {0.4, 0.44, 0.48, 0.52}, set(ratio.tolist())
        lens = ratio * scale; term = 0.52 * scale[-1]
    eps, t0 = [], time.time()
    for e in range(n):
        friction = float(r['obj_friction'][e]) if fr == 'rec' else float(fr)
        if opt.get('penlen') == 'ig':
            pen = dict(pen_length=float(lens[e]), term_length=term)
        env = PenSpinSDX(URDF, names, pen_scale=float(r['obj_scale'][e]), pen_mass=float(r['obj_mass'][e]), friction=friction,
                         pen_inertia=r['obj_inertia'][e] if inertia == 'ig' else None, **pen)
        o = r['init_obj'][e]
        env.reset(r['init_q'][e], o[:3], o[3:7], r['init_qd'][e], o[7:10], o[10:13])
        init_pose = r['in_obs'][e, 1, :16] if r['lengths'][e] > 1 else r['init_q'][e]
        ob = TeacherObs(init_pose, r['in_obs'][e, 0], r['in_priv'][e, 0], r['obj_point_cloud_local'][e],
                        float(r['obj_scale'][e]), float(r['obj_mass'][e]), friction, float(r['obj_restitution'][e]), rng,
                        obs_noise_t=float(opt.get('noise_t', 0.0)), obs_noise_e=float(opt.get('noise_e', 0.0)))
        obs, priv, pc = r['in_obs'][e, 0], r['in_priv'][e, 0], r['in_pc'][e, 0]
        target = r['init_target'][e].astype(float)
        prev_quat = o[3:7]; rot_reward = 0.0; rot_rad = 0.0; steps = T
        jex = -np.inf; on = 0; ns = 0
        for t in range(T):
            if student is None:
                a = teacher.act(obs[None], priv[None], pc[None])[0]
            else:
                a = student.act(obs, ob.proprio_hist, ob.tactile_hist)
            target = np.clip(target + ACTION_SCALE * a, IG_LOWER, IG_UPPER)
            cn = env.control_step(target)
            qn = env.q(); jex = max(jex, float(np.max(qn - np.asarray(IG_UPPER))), float(np.max(np.asarray(IG_LOWER) - qn)))
            on += int((np.asarray(cn)[KEEP] >= CONTACT_THRESH).sum()); ns += 1
            pose = env.pen_pose()
            ft = env.fingertips()
            obs, priv, pc, angvel = ob.update(env.q(), target, cn, ft, pose[:3], pose[3:7], prev_quat)
            prev_quat = pose[3:7]
            vz = float(angvel[2]); rot_reward += float(np.clip(vz, -0.5, 0.5)); rot_rad += vz * CONTROL_DT
            zmin, zmax = env.pen_ends_z()
            if pose[2] < RESET_Z or zmin < END_Z[0] or zmax > END_Z[1]:
                steps = t
                break
        env.close()
        L = int(r['lengths'][e]); ig_vz = r['rot_rad'][e, :L] / CONTROL_DT
        eps.append({'episode': e, 'sdx_steps': steps, 'sdx_rot_reward': rot_reward, 'sdx_rot_rad': rot_rad,
                    'ig_steps': L, 'ig_rot_reward': float(np.clip(ig_vz, -0.5, 0.5).sum()), 'ig_rot_rad': float(r['rot_rad'][e, :L].sum()),
                    'min_contact_dist': env.min_dist, 'min_contact_dist_late': env.min_dist_late, 'max_joint_excess': max(jex, env.max_jex), 'contact_on': on, 'contact_steps': ns})
        print(json.dumps(eps[-1]), flush=True)
    m = lambda k: float(np.mean([x[k] for x in eps]))
    summary = {'record': str(record), 'episodes': n, 'options': {'inertia': inertia, 'friction': fr, **opt}, 'wall_s': round(time.time() - t0, 1),
               'sdx_rot_reward_mean': m('sdx_rot_reward'), 'ig_rot_reward_mean': m('ig_rot_reward'),
               'sdx_duration_s_mean': m('sdx_steps') * CONTROL_DT, 'ig_duration_s_mean': m('ig_steps') * CONTROL_DT,
               'min_contact_dist': float(min(x['min_contact_dist'] for x in eps)),
               'max_joint_excess': float(max(x['max_joint_excess'] for x in eps)),
               'min_contact_dist_late': float(min(x['min_contact_dist_late'] for x in eps)),
               'finite': bool(np.all(np.isfinite([x['sdx_rot_reward'] for x in eps]))),
               'contact_on_rate': float(sum(x['contact_on'] for x in eps) / max(1, 31 * sum(x['contact_steps'] for x in eps))),
               'sdx_contact': dict(CONTACT_OVERRIDE),
               'sdx_joint': dict(JOINT_OVERRIDE), 'sdx_substeps': SUBSTEPS, 'mj_opt': OPT,
               'sdx_full_episodes': int(sum(x['sdx_steps'] >= T for x in eps)), 'ig_full_episodes': int(sum(x['ig_steps'] >= T - 1 for x in eps)),
               'sdx_rot_rad_mean': m('sdx_rot_rad'), 'ig_rot_rad_mean': m('ig_rot_rad')}
    Path(out_prefix + '.json').write_text(json.dumps({'summary': summary, 'episodes': eps}, indent=1))
    print('CLOSED_LOOP_SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    kv = dict(a.split('=', 1) for a in sys.argv[5:])
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else 'all', **kv)
