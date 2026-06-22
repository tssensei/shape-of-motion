#!/bin/bash
#SBATCH --job-name=som-preproc-bush3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/zs292/slurm_outputs/som/som-preproc-bush3-%j.out
#SBATCH --error=/home/zs292/slurm_outputs/som/som-preproc-bush3-%j.err

set -euo pipefail

source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate som

cd /home/zs292/shape-of-motion/preproc

CUDA_VISIBLE_DEVICES=0 python -u compute_tracks_torch.py \
  --image_dir /home/zs292/datasets/custom/images/bush3_test \
  --mask_dir /home/zs292/datasets/custom/masks/bush3_test \
  --out_dir /home/zs292/datasets/custom/bootstapir/bush3_test \
  --model_type bootstapir \
  --ckpt_dir /home/zs292/shape-of-motion/preproc/checkpoints
