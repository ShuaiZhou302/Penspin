#!/bin/bash
# Paired contact collection at SuperDex mu=1.0: each record split into two 32-episode halves, 8 processes total.
# usage: run_pairs.sh <teacher_jobid> <student_jobid>
set -euo pipefail
D=<PROJECT_ROOT>/penspin-superdex-port-20261028
cd "$D"
mkdir -p runs/pairs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
T=$1; B=$2
for spec in "pairT_seed100_$T" "pairT_seed101_$T" "pairT_seed102_$T" "pairBT_seed103_$B"; do
  rec=isaac/runs/record_${spec}.npz
  test -s "$rec"
  for half in 0:32 32:64; do
    tag=${spec}_${half/:/-}
    nohup .venv/bin/python tools/collect_pairs.py "$rec" runs/pairs/${tag}.npz episodes=$half friction=1.0 \
      > runs/pairs/${tag}.log 2> runs/pairs/${tag}.err &
  done
done
sha256sum tools/collect_pairs.py tools/sdx_env.py > runs/pairs/CODE_SHA256SUMS
wait
grep -h PAIRS_DONE runs/pairs/*.log | wc -l
