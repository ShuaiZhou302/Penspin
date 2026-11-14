"""Where does the joint-limit excess come from? Wraps PenSpinSDX to log, per episode, the worst (excess, joint, control
step, substep) and the initial-state excess. usage: diag_jex.py <record> <n_eps> (env vars as for closed_loop)."""
import sys
import json
import numpy as np
sys.path.insert(0, "tools")
import sdx_env  # noqa: E402
import closed_loop  # noqa: E402

LO, HI = np.asarray(sdx_env.IG_LOWER), np.asarray(sdx_env.IG_UPPER)
LOG = []


class Env(sdx_env.PenSpinSDX):
    def reset(self, q_ig, *a, **k):
        out = super().reset(q_ig, *a, **k)
        q0 = np.asarray(q_ig, float); self.k = 0
        ex0 = np.maximum(q0 - HI, LO - q0)
        qs = self.q(); exs = np.maximum(qs - HI, LO - qs)
        LOG.append({'init_excess_record': float(ex0.max()), 'init_joint': int(ex0.argmax()), 'init_excess_sdx': float(exs.max()), 'worst': None})
        return out

    def physics_step(self, target_ig):
        tau = super().physics_step(target_ig)
        q = self.q(); ex = np.maximum(q - HI, LO - q); j = int(ex.argmax()); self.k += 1
        w = LOG[-1]['worst']
        if w is None or ex[j] > w[0]:
            LOG[-1]['worst'] = (float(ex[j]), j, self.k // 10, self.k % 10, float(q[j]), float(LO[j]), float(HI[j]), float(target_ig[j]))
        return tau


closed_loop.PenSpinSDX = Env
closed_loop.main(sys.argv[1], 'assets/teacher_705613.npz', '/tmp/diag_jex_hl', sys.argv[2], inertia='mesh', friction='1.0',
                 noise_t='0.01', noise_e='0.02', seed='42', penlen='ig')
for i, x in enumerate(LOG):
    print('EP', i, json.dumps(x))
