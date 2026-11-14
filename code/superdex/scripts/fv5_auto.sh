#!/bin/bash
# Runs detached: f v5 = f v4 data + f-student pairs (record 803304, seeds 109/110 train_f, 111 shift_f). No resubmission; any failure -> FV5.FAILED.
cd <PROJECT_ROOT>/penspin-superdex-penlen-20261031 || exit 1
L=fv5_auto.log; fail(){ date >> $L; echo "$1" | tee -a $L > FV5.FAILED; exit 1; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R=803304; end=$(( $(date +%s) + 6*3600 ))
while :; do
  s=$(sacct -n -P -j $R -o JobID,State | grep -v '\.')
  echo "$s" | grep -qE "FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL" && fail "record: $s"
  [ -n "$s" ] && ! echo "$s" | grep -q "\[" && ! echo "$s" | cut -d"|" -f2 | grep -qv "^COMPLETED$" && break
  [ "$(date +%s)" -gt "$end" ] && fail "record wait expired"
  sleep 60
done
date >> $L; echo "records completed" >> $L
for i in 0 1 2; do grep -h RECORD_SUMMARY isaac/runs/record_${R}_$i.out | grep -q '"policy": "student"' || fail "record $i policy not student"; done
sha256sum -c step7_code.sha256 >> $L 2>&1 || fail "code changed"
mkdir -p runs/pairs_v5; : > jobs_pairs_v5.txt
for f in runs/pairs_v4/pair*.npz; do ln -sf ../pairs_v4/$(basename $f) runs/pairs_v5/; done
for s in 109 110 111; do for h in 0-32 32-64; do
  rec=isaac/runs/record_pairFB_seed${s}_$R.npz; [ -s $rec ] || fail "missing $rec"; o=runs/pairs_v5/pairFB_seed${s}_${R}_$h
  echo ".venv/bin/python tools/collect_pairs.py $rec $o.npz episodes=${h/-/:} friction=1.0 penlen=ig > $o.log 2> $o.err" >> jobs_pairs_v5.txt
done; done
xargs -a jobs_pairs_v5.txt -P 6 -I{} sh -c "{}" >> $L 2>&1
n=$(ls runs/pairs_v5/pairFB_*.npz | wc -l); date >> $L; echo "new pairs $n / 6" >> $L; [ "$n" = 6 ] || fail "pairs $n/6"
.venv/bin/python tools/pair_stats.py runs/pairs_v5 runs/pairs_v5/dataset_mu1_v5.npz runs/pairs_v5/stats_mu1_v5.json train=pairT_seed100_ train=pairT_seed101_ val=pairT_seed102_ shift=pairBT_seed103_ train_bt=pairBT_seed104_ train_bt=pairBT_seed105_ train_p=pairP_seed106_ train_p=pairP_seed107_ shift_p=pairP_seed108_ train_f=pairFB_seed109_ train_f=pairFB_seed110_ shift_f=pairFB_seed111_ >> $L 2>&1 || fail "pair_stats"
.venv/bin/python tools_diag/check_v5_superset.py >> $L 2>&1 || fail "v5 rows differ from v4"
OMP_NUM_THREADS=4 .venv/bin/python tools/train_f.py runs/pairs_v5/dataset_mu1_v5.npz runs/f_v5 budgets=100 seeds=0,1,2 steps=4000 extra=train_bt,train_p,train_f evals=val,shift,shift_p,shift_f >> $L 2>&1 || fail "train_f"
OMP_NUM_THREADS=4 .venv/bin/python tools/train_f.py runs/pairs_v5/dataset_mu1_v5.npz runs/f_v4_on_v5 budgets=100 seeds=0 steps=4000 extra=train_bt,train_p evals=val,shift,shift_p,shift_f >> $L 2>&1 || fail "train_f v4 control"
OMP_NUM_THREADS=2 .venv/bin/python tools_diag/diag_own.py runs/f_v5/f_b100_s0.pt runs/diag/own_state_f1_v5.json >> $L 2>&1 || fail "diag_own"
date >> $L; touch FV5_DONE
