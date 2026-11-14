#!/bin/bash
# Runs detached: wait for the fine-tuning array, then SDX evaluation of the fine-tuned proprio students: 2pi filter at mu 1.0/0.5/1.5, pi filter at mu 1.0 (3 eval seeds).
# Any task not COMPLETED: write step9_auto.FAILED and stop (no resubmission).
cd <PROJECT_ROOT>/penspin-superdex-penlen-20261031 || exit 1
J=$(cat jobs_ft.txt); L=step9_auto.log
end=$(( $(date +%s) + 120*3600 ))
while :; do
  s=$(sacct -X -n -P -j $J -o JobID,State)
  if echo "$s" | grep -qE "FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL"; then
    echo "$s" > step9_auto.FAILED; date >> $L; echo "jobs failed, stopping" >> $L; exit 1
  fi
  if [ "$(echo "$s" | grep -c "|COMPLETED")" = 24 ]; then break; fi
  if [ "$(date +%s)" -gt "$end" ]; then echo "$s" > step9_auto.FAILED; echo "wait expired" >> $L; exit 1; fi
  sleep 120
done
date >> $L; echo "fine-tuning completed" >> $L
sha256sum -c step7_code.sha256 >> $L 2>&1 || { echo "code changed" > step9_auto.FAILED; exit 1; }
JF=jobs_step9.txt; : > $JF
add(){ for s in 42 43 44; do
  rec=isaac/runs/record_evalP_seed${s}_777023.npz; [ -s "$rec" ] && [ -s "$2" ] || { echo "missing $rec or $2" > step9_auto.FAILED; exit 1; }
  o=runs/zs_mu$3_$1_s$s
  echo ".venv/bin/python tools/closed_loop.py $rec assets/teacher_705613.npz $o all inertia=mesh friction=$3 noise_t=0.01 noise_e=0.02 seed=$s penlen=ig policy=student mode=proprio ckpt=$2 > $o.log 2> $o.err" >> $JF
done; }
for b in 10 25 50 100; do for r in 0 1 2; do
  for mu in 1.0 0.5 1.5; do add ft2pib${b}_r$r runs/ft/ft2pib${b}_r$r.pth $mu; done
  add ftpib${b}_r$r runs/ft/ftpib${b}_r$r.pth 1.0
done; done
OMP_NUM_THREADS=1 xargs -a $JF -P 8 -I{} sh -c "{}" >> $L 2>&1
n=$(ls runs/zs_mu*_ft*b*_r?_s4?.json 2>/dev/null | wc -l)
date >> $L; echo "sdx json $n / 144" >> $L
if [ "$n" = 144 ]; then touch step9_auto.DONE; else echo "sdx incomplete $n/144" > step9_auto.FAILED; exit 1; fi
