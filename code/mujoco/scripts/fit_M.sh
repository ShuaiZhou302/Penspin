#!/bin/bash
# f_M and calib_M on the MuJoCo pairs, same settings as f v4 / calib v4 (budget 100 + train_bt, train_p). No resubmission.
cd <PROJECT_ROOT>/penspin-mujoco-20261107
L=fit_M.log; date > $L; DS=runs/pairs_M/dataset_M.npz
sha256sum -c CODE_SHA256 --quiet >> $L 2>&1 || { echo code > FIT_M.FAILED; exit 1; }
OMP_NUM_THREADS=4 .venv/bin/python tools/train_f.py $DS runs/f_M budgets=100 seeds=0,1,2 steps=4000 extra=train_bt,train_p evals=val,shift,shift_p >> $L 2>&1 || { echo f > FIT_M.FAILED; exit 1; }
OMP_NUM_THREADS=2 .venv/bin/python tools/calib_thresh_delay.py $DS runs/calib_M/calib_M.npz runs/calib_M/calib_M.json 100 train_bt,train_p val,shift,shift_p >> $L 2>&1 || { echo calib > FIT_M.FAILED; exit 1; }
date >> $L; touch FIT_M.DONE
