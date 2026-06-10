#!/bin/bash
#SBATCH --job-name=som-train-bush1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/zs292/slurm_outputs/som/som-train-bush1-%j.out
#SBATCH --error=/home/zs292/slurm_outputs/som/som-train-bush1-%j.err

set -euo pipefail

source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate som

cd /home/zs292/shape-of-motion

python -u run_training.py \
  --work-dir /home/zs292/outputs/bush1_radii_fix_v1 \
  data:custom \
  --data.data-dir /home/zs292/datasets/bush1 \
  --data.camera-type droid_recon