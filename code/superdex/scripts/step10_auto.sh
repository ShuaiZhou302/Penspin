#!/bin/bash
# Runs detached: wait for the f v5 student trainings + IG evals (jobs_fv5.txt), moving pending IG eval arrays to debug as slots free up,
# then 27 SDX zero-shots runs/zs_mu{1.0,0.5,1.5}_fv5_t{0,1,2}_s{42,43,44}. Moves only, never resubmits. Failure -> step10_auto.FAILED.
cd <PROJECT_ROOT>/penspin-superdex-penlen-20261031 || exit 1
N=<PROJECT_ROOT>/penspin-student-fv5-20261106; JF=$N/runs/jobs_fv5.txt
J=$(awk '{print $2","$3}' $JF | paste -sd,); EV=$(awk '{print $3}' $JF); L=step10_auto.log
end=$(( $(date +%s) + 72*3600 ))
while :; do
  s=$(sacct -n -P -j $J -o JobID,State | grep -v '\.')
  if echo "$s" | grep -qE "FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL"; then echo "$s" > step10_auto.FAILED; date >> $L; exit 1; fi
  if [ -n "$s" ] && ! echo "$s" | grep -q "\[" && ! echo "$s" | cut -d"|" -f2 | grep -qv "^COMPLETED$"; then break; fi
  for e in $EV; do
    r=$(squeue -j $e -h -t PD -o "%P %R" 2>/dev/null | head -1)
    case "$r" in ${MAIN_PARTITION}*Dependency*|"") ;; ${MAIN_PARTITION}*) scontrol update jobid=$e partition=${DEBUG_PARTITION} >/dev/null 2>&1 && { date >> $L; echo "moved $e" >> $L; };; esac
  done
  [ "$(date +%s)" -gt "$end" ] && { echo "wait expired" > step10_auto.FAILED; exit 1; }
  sleep 60
done
date >> $L; echo "jobs completed" >> $L
sha256sum -c step7_code.sha256 >> $L 2>&1 || { echo "code changed" > step10_auto.FAILED; exit 1; }
JS=jobs_step10.txt; : > $JS
while read ts t e; do
  ck=$N/outputs/AllegroHandHora/objpose_proprio_tactile_ends_est_$t/stage1_nn/final.pth
  for mu in 1.0 0.5 1.5; do for s in 42 43 44; do
    rec=isaac/runs/record_evalFB_seed${s}_783809.npz; [ -s "$rec" ] && [ -s "$ck" ] || { echo "missing $rec or $ck" > step10_auto.FAILED; exit 1; }
    o=runs/zs_mu${mu}_fv5_t${ts}_s$s
    echo ".venv/bin/python tools/closed_loop.py $rec assets/teacher_705613.npz $o all inertia=mesh friction=$mu noise_t=0.01 noise_e=0.02 seed=$s penlen=ig policy=student mode=proprio_tactile_ends_est ckpt=$ck > $o.log 2> $o.err" >> $JS
  done; done
done < $JF
OMP_NUM_THREADS=1 xargs -a $JS -P 8 -I{} sh -c "{}" >> $L 2>&1
n=$(ls runs/zs_mu*_fv5_t?_s4?.json 2>/dev/null | wc -l)
date >> $L; echo "sdx json $n / 27" >> $L
if [ "$n" = 27 ]; then touch step10_auto.DONE; else echo "sdx incomplete $n/27" > step10_auto.FAILED; exit 1; fi
