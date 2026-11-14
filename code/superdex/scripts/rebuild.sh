#!/bin/bash
# Refit the contact models on pairs with the correct pen length, then the friction sweep of the policies that need no retraining.
cd <PROJECT_ROOT>/penspin-superdex-penlen-20261031 || exit 1
L=rebuild.log; fail(){ date >> $L; echo "$1" | tee -a $L > REBUILD.FAILED; exit 1; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date >> $L; xargs -a jobs_pairs_v4.txt -P 8 -I{} sh -c "{}" >> $L 2>&1
n=$(ls runs/pairs_v4/*.npz | wc -l); date >> $L; echo "pairs $n / 18" >> $L; [ "$n" = 18 ] || fail "pairs $n/18"
.venv/bin/python tools/pair_stats.py runs/pairs_v4 runs/pairs_v4/dataset_mu1_v4.npz runs/pairs_v4/stats_mu1_v4.json train=pairT_seed100_ train=pairT_seed101_ val=pairT_seed102_ shift=pairBT_seed103_ train_bt=pairBT_seed104_ train_bt=pairBT_seed105_ train_p=pairP_seed106_ train_p=pairP_seed107_ shift_p=pairP_seed108_ >> $L 2>&1 || fail "pair_stats"
OMP_NUM_THREADS=4 .venv/bin/python tools/train_f.py runs/pairs_v4/dataset_mu1_v4.npz runs/f_v4 budgets=100 seeds=0,1,2 steps=4000 extra=train_bt,train_p evals=val,shift,shift_p >> $L 2>&1 || fail "train_f"
OMP_NUM_THREADS=2 .venv/bin/python tools/calib_thresh_delay.py runs/pairs_v4/dataset_mu1_v4.npz runs/calib/calib_v4.npz runs/calib/calib_v4.json 100 train_bt,train_p val,shift,shift_p >> $L 2>&1 || fail "calib"
date >> $L; touch REBUILD_DONE
xargs -a jobs_sweep.txt -P 8 -I{} sh -c "{}" >> $L 2>&1
date >> $L; touch SWEEP_DONE
