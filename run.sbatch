#!/bin/bash
#SBATCH --job-name=bush4-sweep-3dgs-hq
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=/home/zs292/slurm_outputs/som/bush4-sweep-3dgs-hq-%j.out
#SBATCH --error=/home/zs292/slurm_outputs/som/bush4-sweep-3dgs-hq-%j.err

set -euo pipefail

source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate som

cd /home/zs292/shape-of-motion

MODAL_DIR=/home/zs292/outputs_modal/bush4
SWEEP_DATA_DIR=$MODAL_DIR/sweep_colmap_dataset
WORK_DIR=/home/zs292/outputs/bush4_sweep_static_3dgs_hq_v1

PYTHONPATH=. python -u run_training.py \
  --work-dir $WORK_DIR \
  --trajectory-type static \
  --num-fg 50000 \
  --num-bg 50000 \
  --num-epochs 3000 \
  --batch-size 4 \
  --num-dl-workers 8 \
  --loss.w-rgb 1.0 \
  --loss.w-mask 1.0 \
  --loss.w-depth-reg 0.0 \
  --loss.w-depth-grad 0.0 \
  --loss.w-depth-const 0.0 \
  --loss.w-track 0.0 \
  --loss.w-smooth-tracks 0.0 \
  --loss.w-z-accel 0.0 \
  --loss.w-dct-coef 0.0 \
  data:custom \
  --data.data-dir $SWEEP_DATA_DIR \
  --data.camera-type colmap \
  --data.load-from-cache