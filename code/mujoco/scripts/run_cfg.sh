#!/bin/bash
# usage: run_cfg.sh <out> <record> <n_eps> <seed> <tc> <d> <substeps>   (contact and limit share solref/solimp)
cd <PROJECT_ROOT>/penspin-mujoco-20261107
export MJ_CONTACT="solref=$5,1;solimp=$6,$6,0.001" MJ_LIMIT="solreflimit=$5,1;solimplimit=$6,$6,0.001" MJ_SUBSTEPS=$7 OMP_NUM_THREADS=1
[ "$5" = default ] && unset MJ_CONTACT MJ_LIMIT
exec .venv/bin/python tools/closed_loop_mj.py $2 assets/teacher_705613.npz $1 $3 inertia=mesh friction=1.0 noise_t=0.01 noise_e=0.02 seed=$4 penlen=ig > $1.log 2> $1.err
