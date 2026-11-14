#!/bin/bash
# Runs detached: wait for the retrained f and calib students, check the records are student rollouts, then SDX zero-shot at mu 1.0/0.5/1.5 (3 eval seeds).
# Any job not COMPLETED, or any record not policy=student: write step7_auto.FAILED with the reason and stop (no resubmission).
cd <PROJECT_ROOT>/penspin-superdex-penlen-20261031 || exit 1
J="783805,783806,783807,783808,783809,783810"
R=<PROJECT_ROOT>/penspin-student-penlen-20261031/outputs/AllegroHandHora
L=step7_auto.log
end=$(( $(date +%s) + 120*3600 ))
while :; do
  s=$(sacct -X -n -P -j $J -o JobID,State)
  if echo "$s" | grep -qE "FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL"; then
    echo "$s" > step7_auto.FAILED; date >> $L; echo "jobs failed, stopping" >> $L; exit 1
  fi
  if [ -n "$s" ] && ! echo "$s" | cut -d"|" -f2 | grep -qv "^COMPLETED"; then break; fi
  if [ "$(date +%s)" -gt "$end" ]; then echo "$s" > step7_auto.FAILED; echo "wait expired" >> $L; exit 1; fi
  sleep 300
done
date >> $L; echo "jobs completed" >> $L
sha256sum -c step7_code.sha256 >> $L 2>&1 || { echo "code changed" > step7_auto.FAILED; exit 1; }
for j in 783809 783810; do for i in 0 1 2; do
  grep -q "RECORD_SUMMARY.*\"policy\": \"student\"" isaac/runs/record_${j}_$i.out || { echo "record ${j}_$i is not a student rollout" > step7_auto.FAILED; exit 1; }
done; done
JF=jobs_step7.txt; : > $JF
for mu in 1.0 0.5 1.5; do for s in 42 43 44; do
  for p in "fv4 783805 evalFB_seed${s}_783809" "cv4 783806 evalTD_seed${s}_783810"; do
    set -- $p; rec=isaac/runs/record_$3.npz; ck=$R/objpose_proprio_tactile_ends_est_$2/stage1_nn/final.pth
    [ -s "$rec" ] && [ -s "$ck" ] || { echo "missing $rec or $ck" > step7_auto.FAILED; exit 1; }
    o=runs/zs_mu${mu}_$1_s$s
    echo ".venv/bin/python tools/closed_loop.py $rec assets/teacher_705613.npz $o all inertia=mesh friction=$mu noise_t=0.01 noise_e=0.02 seed=$s penlen=ig policy=student mode=proprio_tactile_ends_est ckpt=$ck > $o.log 2> $o.err" >> $JF
  done
done; done
OMP_NUM_THREADS=1 xargs -a $JF -P 8 -I{} sh -c "{}" >> $L 2>&1
n=$(ls runs/zs_mu*_fv4_s4?.json runs/zs_mu*_cv4_s4?.json 2>/dev/null | wc -l)
date >> $L; echo "sdx json $n / 18" >> $L
if [ "$n" = 18 ]; then touch step7_auto.DONE; else echo "sdx incomplete $n/18" > step7_auto.FAILED; exit 1; fi
