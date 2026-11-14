#!/bin/bash -l
#SBATCH -J penspin-student-reset-audit
#SBATCH -p <PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --array=0-2%3
#SBATCH -o <PROJECT_ROOT>/penspin-student-mjalign-20261107/runs/student_%A_%a.out
#SBATCH -e <PROJECT_ROOT>/penspin-student-mjalign-20261107/runs/student_%A_%a.err
set -euo pipefail
REPO=<PROJECT_ROOT>/penspin-student-mjalign-20261107
cd "$REPO"
sha256sum -c SOURCE_SHA256SUMS --quiet
module load gcc/11.5
export PATH="$REPO/.conda/bin:$PATH"
export LD_LIBRARY_PATH="$REPO/.conda/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 PENSPIN_PROTOCOL_AUDIT=1
export TORCH_EXTENSIONS_DIR="$REPO/runs/torch_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
ITEM=${SLURM_ARRAY_TASK_ID}
SEED=$((42 + ITEM % 3))
MODE="objpose_${INPUT_MODE:?}"
export AUDIT_MODE=baseline
export AUDIT_LABEL="${MODE}_pilot${PILOT_ID}_seed${SEED}_${SLURM_ARRAY_JOB_ID}"
TEACHER=<PROJECT_ROOT>/penspin-16831/outputs/AllegroHandHora/teacher_paper500m_705613/stage1_nn/best_reward_86.48.pth
STUDENT_CHECKPOINT="${STUDENT_CHECKPOINT:-$REPO/outputs/AllegroHandHora/objpose_${INPUT_MODE}_${PILOT_ID}/stage1_nn/final.pth}"
test -s "$STUDENT_CHECKPOINT"
python train.py task=AllegroHandHora headless=True seed="$SEED" test=True \
  train.algo=DemonTrain train.demon_path="$TEACHER" train.load_path="$STUDENT_CHECKPOINT" \
  train.ppo.output_name="AllegroHandHora/protocol_${AUDIT_LABEL}" \
  train.ppo.distill=True train.ppo.use_l1=True train.ppo.enable_latent_loss=False train.ppo.priv_info=False \
  train.ppo.proprio_mode=True train.ppo.proprio_len=30 train.ppo.input_mode="${INPUT_MODE:?}" train.ppo.is_demon=True \
  task.env.numEnvs=64 task.env.numObservations=276 \
  task.env.randomForceProbScalar=0.25 task.env.rotation_axis=+z \
  task.env.reward.pencil_z_dist_penalty_scale=-1.0 task.env.forceScale=2 \
  task.env.randomization.obs_noise_t_scale=0.01 task.env.randomization.obs_noise_e_scale=0.02 \
  task.env.object.type=cylinder_pencil-5-7 task.env.genGraspCategory=pencil \
  task.env.privInfo.enable_obj_orientation=True task.env.privInfo.enable_ft_pos=True \
  task.env.privInfo.enable_obj_angvel=True task.env.privInfo.enable_tactile=True \
  task.env.randomization.randomizeScaleList=[0.29] task.env.grasp_cache_name=allegro_round_tip_thin \
  task.env.hora.point_cloud_sampled_dim=100 task.env.initPoseMode=low task.env.reset_height_threshold=0.605 \
  task.env.reward.angvelClipMax=0.5 task.env.enable_obj_ends=True wandb_activate=False
