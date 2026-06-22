#!/bin/bash
#SBATCH --job-name=som-train-bush3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=/home/zs292/slurm_outputs/som/som-train-bush3-%j.out
#SBATCH --error=/home/zs292/slurm_outputs/som/som-train-bush3-%j.err

set -euo pipefail

source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate som

cd /home/zs292/shape-of-motion

python -u run_training.py \
  --work-dir /home/zs292/outputs/dct619/bush3_no_2d_prior \
  --trajectory-type dct_center \
  data:custom \
  --data.data-dir /home/zs292/datasets/custom_train/bush3_test \
  --data.camera-type megasam