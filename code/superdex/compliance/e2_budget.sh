#!/bin/bash
# E2a: equal-attempt budget sweep for f and calibration on the v5 dataset (frozen tools copy). No resubmission.
cd <PROJECT_ROOT>/penspin-superdex-compliance-20261107
L=e2_budget.log; date > $L; D=runs/pairs_v5/dataset_mu1_v5.npz; E=val,shift,shift_p,shift_f
fail() { echo "$1" > e2_budget.FAILED; date >> $L; exit 1; }
sha256sum -c CODE_SHA256 >> $L 2>&1 || fail "code changed"
OMP_NUM_THREADS=4 .venv/bin/python tools/train_f.py $D runs/e2/f_teacher budgets=10,25,50,100 seeds=0,1,2 steps=4000 extra= evals=$E >> $L 2>&1 || fail "f teacher-only"
OMP_NUM_THREADS=4 .venv/bin/python tools/train_f.py $D runs/e2/f_extra budgets=10,25,50,100 seeds=0,1,2 steps=4000 extra=train_bt,train_p evals=$E >> $L 2>&1 || fail "f extra"
for b in 10 25 50 100; do
  OMP_NUM_THREADS=2 .venv/bin/python tools/calib_thresh_delay.py $D runs/e2/calib_teacher_b$b.npz runs/e2/calib_teacher_b$b.json $b "" $E >> $L 2>&1 || fail "calib teacher b$b"
  OMP_NUM_THREADS=2 .venv/bin/python tools/calib_thresh_delay.py $D runs/e2/calib_extra_b$b.npz runs/e2/calib_extra_b$b.json $b train_bt,train_p $E >> $L 2>&1 || fail "calib extra b$b"
done
date >> $L; touch e2_budget.DONE
