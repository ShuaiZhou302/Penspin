#!/bin/bash -l
#SBATCH -J penspin-ftd
#SBATCH -p <PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --array=0-5%6
#SBATCH -o <PROJECT_ROOT>/penspin-superdex-penlen-20261031/runs/ft_diag/slurm_%A_%a.out
#SBATCH -e <PROJECT_ROOT>/penspin-superdex-penlen-20261031/runs/ft_diag/slurm_%A_%a.err
set -euo pipefail
cd <PROJECT_ROOT>/penspin-superdex-penlen-20261031
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
I=${SLURM_ARRAY_TASK_ID}; S=$((I % 3)); V=$((I / 3))
LR=(1e-3 1e-4); NM=(e400lr3 e400lr4)
BASE=<PROJECT_ROOT>/penspin-16831-dagger-scratch-20261012/outputs/AllegroHandHora/scratch_dagger_713476/stage1_nn/iter_400_newsteps_9830400.pth
.venv/bin/python tools_diag/finetune_student.py $BASE proprio runs/ft_diag/${NM[$V]}_r$S budget=100 seed=$S epochs=400 lr=${LR[$V]}
