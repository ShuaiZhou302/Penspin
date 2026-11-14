#!/bin/bash
# Runs detached: wait for training seeds 1,2 of f v4, calib v4, rand and their IG evals, then SDX zero-shot at mu 1.0/0.5/1.5 (3 eval seeds), with B+tactile seeds 1,2.
# Initial states come from each policy family seed-0 record (identical across records for an eval seed). Any job not COMPLETED: write step8_auto.FAILED and stop (no resubmission).
cd <PROJECT_ROOT>/penspin-superdex-penlen-20261031 || exit 1
J=$(awk "{print \$3\",\"\$4}" jobs_ids_seeds.txt | paste -sd,)
A=<PROJECT_ROOT>/penspin-student-penlen-20261031/outputs/AllegroHandHora/objpose_proprio_tactile_ends_est
B=<PROJECT_ROOT>/penspin-student-objpose-dagger-20261022/outputs/AllegroHandHora/objpose_proprio_tactile_ends_est
L=step8_auto.log
end=$(( $(date +%s) + 120*3600 ))
while :; do
  s=$(sacct -X -n -P -j $J -o JobID,State)
  if echo "$s" | grep -qE "FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL"; then
    echo "$s" > step8_auto.FAILED; date >> $L; echo "jobs failed, stopping" >> $L; exit 1
  fi
  if [ -n "$s" ] && ! echo "$s" | cut -d"|" -f2 | grep -qv "^COMPLETED"; then break; fi
  if [ "$(date +%s)" -gt "$end" ]; then echo "$s" > step8_auto.FAILED; echo "wait expired" >> $L; exit 1; fi
  sleep 300
done
date >> $L; echo "jobs completed" >> $L
sha256sum -c step7_code.sha256 >> $L 2>&1 || { echo "code changed" > step8_auto.FAILED; exit 1; }
JF=jobs_step8.txt; : > $JF
add(){ # name train_seed ckpt record_template
  for mu in 1.0 0.5 1.5; do for s in 42 43 44; do
    rec=isaac/runs/record_$(echo $4 | sed "s/SS/$s/").npz; [ -s "$rec" ] && [ -s "$3" ] || { echo "missing $rec or $3" > step8_auto.FAILED; exit 1; }
    o=runs/zs_mu${mu}_$1_t$2_s$s
    echo ".venv/bin/python tools/closed_loop.py $rec assets/teacher_705613.npz $o all inertia=mesh friction=$mu noise_t=0.01 noise_e=0.02 seed=$s penlen=ig policy=student mode=proprio_tactile_ends_est ckpt=$3 > $o.log 2> $o.err" >> $JF
  done; done; }
while read k ts t e; do
  case $k in f) add fv4 $ts ${A}_$t/stage1_nn/final.pth evalFB_seedSS_783809;; c) add cv4 $ts ${A}_$t/stage1_nn/final.pth evalTD_seedSS_783810;; r) add trand $ts ${A}_$t/stage1_nn/final.pth evalTR_seedSS_778933;; esac
done < jobs_ids_seeds.txt
add btactile 1 ${B}_772991/stage1_nn/final.pth evalBT_seedSS_777024
add btactile 2 ${B}_772994/stage1_nn/final.pth evalBT_seedSS_777024
OMP_NUM_THREADS=1 xargs -a $JF -P 8 -I{} sh -c "{}" >> $L 2>&1
n=$(ls runs/zs_mu*_t[12]_s4?.json 2>/dev/null | wc -l)
date >> $L; echo "sdx json $n / 72" >> $L
if [ "$n" = 72 ]; then touch step8_auto.DONE; else echo "sdx incomplete $n/72" > step8_auto.FAILED; exit 1; fi
