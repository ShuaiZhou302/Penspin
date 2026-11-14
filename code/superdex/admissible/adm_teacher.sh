#!/bin/bash
# Section 7: teacher at A(n), n in 10/20/40, eval seeds 42-44 x 64, mu=1.0. No resubmission.
cd <PROJECT_ROOT>/penspin-superdex-admissible-20261107
L=adm_teacher.log; date > $L
sha256sum -c CODE_SHA256 >> $L 2>&1 || { echo "code changed" > adm_teacher.FAILED; exit 1; }
JS=jobs_adm_teacher.txt; : > $JS
for n in 10 20 40; do for s in 42 43 44; do o=runs/teacher_A${n}_s$s
  echo "SDX_JOINT=limit_stiffness=1e6,limit_damping=0 SDX_CONTACT=penalty_coefficient=1e11,penalty_smoothing_half_distance=0.0002 SDX_SUBSTEPS=$n OMP_NUM_THREADS=1 .venv/bin/python tools/closed_loop.py isaac/runs/record_evalT_seed${s}_777022.npz assets/teacher_705613.npz $o all inertia=mesh friction=1.0 noise_t=0.01 noise_e=0.02 seed=$s penlen=ig > $o.log 2> $o.err" >> $JS
done; done
xargs -a $JS -P 9 -I{} sh -c "{}" >> $L 2>&1
n=$(ls runs/teacher_A*_s4?.json 2>/dev/null | wc -l); date >> $L; echo "json $n / 9" >> $L
[ "$n" = 9 ] && touch adm_teacher.DONE || echo "incomplete $n/9" > adm_teacher.FAILED
