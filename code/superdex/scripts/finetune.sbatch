#!/bin/bash -l
#SBATCH -J penspin-ft
#SBATCH -p <PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=0-23%8
#SBATCH -o <PROJECT_ROOT>/penspin-superdex-penlen-20261031/runs/ft/slurm_%A_%a.out
#SBATCH -e <PROJECT_ROOT>/penspin-superdex-penlen-20261031/runs/ft/slurm_%A_%a.err
set -euo pipefail
cd <PROJECT_ROOT>/penspin-superdex-penlen-20261031
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
I=${SLURM_ARRAY_TASK_ID}
TH=(6.283185 3.141593); NM=(2pi pi); BU=(10 25 50 100)
T=$((I / 12)); J=$((I % 12)); B=${BU[$((J / 3))]}; S=$((J % 3))
BASE=<PROJECT_ROOT>/penspin-16831-dagger-scratch-20261012/outputs/AllegroHandHora/scratch_dagger_713476/stage1_nn/iter_400_newsteps_9830400.pth
.venv/bin/python tools/finetune_student.py $BASE proprio runs/ft/ft${NM[$T]}b${B}_r$S budget=$B seed=$S thresh=${TH[$T]}
