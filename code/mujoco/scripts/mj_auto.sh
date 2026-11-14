#!/bin/bash
# Runs detached: wait for the MuJoCo-aligned student trainings + IG evals (jobs_mj.txt), moving pending IG eval arrays to debug
# as slots free up, then MuJoCo (target M, n=1) zero-shots runs/zsM_{fM,cM}_t{0,1,2}_s{42,43,44}. Moves only, never
# resubmits. Failure -> MJ_AUTO.FAILED.
cd <PROJECT_ROOT>/penspin-mujoco-20261107 || exit 1
N=<PROJECT_ROOT>/penspin-student-mjalign-20261107; JF=$N/runs/jobs_mj.txt
J=$(awk '{print $3","$4}' $JF | paste -sd,); EV=$(awk '{print $4}' $JF); L=mj_auto.log
end=$(( $(date +%s) + 120*3600 ))
date > $L
while :; do
  s=$(sacct -n -P -j $J -o JobID,State | grep -v '\.')
  if echo "$s" | grep -qE "FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL"; then echo "$s" > MJ_AUTO.FAILED; date >> $L; exit 1; fi
  if [ -n "$s" ] && ! echo "$s" | grep -q "\[" && ! echo "$s" | cut -d"|" -f2 | grep -qv "^COMPLETED$"; then break; fi
  for e in $EV; do
    r=$(squeue -j $e -h -t PD -o "%P %R" 2>/dev/null | head -1)
    case "$r" in ${MAIN_PARTITION}*Dependency*|"") ;; ${MAIN_PARTITION}*) scontrol update jobid=$e partition=${DEBUG_PARTITION} >/dev/null 2>&1 && { date >> $L; echo "moved $e" >> $L; };; esac
  done
  [ "$(date +%s)" -gt "$end" ] && { echo "wait expired" > MJ_AUTO.FAILED; exit 1; }
  sleep 60
done
date >> $L; echo "jobs completed" >> $L
sha256sum -c CODE_SHA256 --quiet >> $L 2>&1 || { echo "code changed" > MJ_AUTO.FAILED; exit 1; }
JS=jobs_zsM_aligned.txt; : > $JS
while read arm t tr ev; do
  ck=$N/outputs/AllegroHandHora/objpose_proprio_tactile_ends_est_$tr/stage1_nn/final.pth
  [ -s "$ck" ] || { echo "missing $ck" > MJ_AUTO.FAILED; exit 1; }
  for s in 42 43 44; do
    echo "./run_default.sh runs/zsM_${arm}_t${t}_s$s isaac/runs/record_evalFB_seed${s}_783809.npz $s 1 policy=student mode=proprio_tactile_ends_est ckpt=$ck" >> $JS
  done
done < $JF
./runjobs.sh $JS 18 ZSM_ALIGNED.RUN >> $L 2>&1
n=$(ls runs/zsM_fM_t?_s4?.json runs/zsM_cM_t?_s4?.json 2>/dev/null | wc -l)
date >> $L; echo "json $n / 18" >> $L
if [ "$n" = 18 ]; then touch MJ_AUTO.DONE; else echo "incomplete $n/18" > MJ_AUTO.FAILED; exit 1; fi
