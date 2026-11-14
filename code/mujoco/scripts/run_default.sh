#!/bin/bash
# usage: run_default.sh <out> <record> <seed> <substeps> [k=v ...]   MuJoCo default contact/limits (target M), all episodes
cd <PROJECT_ROOT>/penspin-mujoco-20261107
unset MJ_CONTACT MJ_LIMIT MJ_OPT; export MJ_SUBSTEPS=$4 OMP_NUM_THREADS=1
o=$1; rec=$2; s=$3; shift 4
exec .venv/bin/python tools/closed_loop_mj.py $rec assets/teacher_705613.npz $o all inertia=mesh friction=1.0 noise_t=0.01 noise_e=0.02 seed=$s penlen=ig "$@" > $o.log 2> $o.err
