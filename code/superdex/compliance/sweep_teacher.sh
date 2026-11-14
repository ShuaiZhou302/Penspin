#!/bin/bash
# E1 stage 1: teacher closed loop in SDX mu=1.0 under 6 contact configs x eval seeds 42-44 x 64. No resubmission.
cd <PROJECT_ROOT>/penspin-superdex-compliance-20261107
L=sweep_teacher.log; date > $L
sha256sum -c CODE_SHA256 >> $L 2>&1 || { echo "code changed" > sweep_teacher.FAILED; exit 1; }
JS=jobs_teacher.txt; : > $JS
while read c ov; do [ "$ov" = none ] && ov=""
  for s in 42 43 44; do o=runs/teacher_${c}_s$s
    echo "SDX_CONTACT=$ov OMP_NUM_THREADS=1 .venv/bin/python tools/closed_loop.py isaac/runs/record_evalT_seed${s}_777022.npz assets/teacher_705613.npz $o all inertia=mesh friction=1.0 noise_t=0.01 noise_e=0.02 seed=$s penlen=ig > $o.log 2> $o.err" >> $JS
  done
done < configs.txt
xargs -a $JS -P 9 -I{} sh -c "{}" >> $L 2>&1
n=$(ls runs/teacher_C?_s4?.json 2>/dev/null | wc -l); date >> $L; echo "json $n / 18" >> $L
[ "$n" = 18 ] && touch sweep_teacher.DONE || echo "incomplete $n/18" > sweep_teacher.FAILED
