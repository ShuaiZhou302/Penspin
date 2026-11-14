#!/bin/bash
# E3: f trained on open-loop SDX replay labels (deployable pairing), all rows or rows with open-loop pen error < 1 cm.
cd <PROJECT_ROOT>/penspin-superdex-compliance-20261107
until [ -e e2_budget.DONE ] || [ -e e2_budget.FAILED ]; do sleep 60; done
L=e3_open.log; date > $L; D=runs/pairs_v5/dataset_mu1_v5.npz; E=val,shift,shift_p,shift_f
fail() { echo "$1" > e3_open.FAILED; date >> $L; exit 1; }
sha256sum -c CODE_SHA256 CODE_SHA256_E3 >> $L 2>&1 || fail "code changed"
for om in inf 0.01; do for ex in none train_bt,train_p; do x=$ex; [ "$ex" = none ] && x=""
  o=runs/e3/open_${om}_$( [ "$ex" = none ] && echo teacher || echo extra )
  OMP_NUM_THREADS=4 .venv/bin/python tools/train_f_open.py $D $o budgets=10,25,50,100 seeds=0,1,2 steps=4000 extra=$x evals=$E open_max=$om >> $L 2>&1 || fail "$o"
done; done
date >> $L; touch e3_open.DONE
