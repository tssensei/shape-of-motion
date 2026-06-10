#!/bin/bash
#SBATCH --job-name=som-preproc-bush1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/zs292/slurm_outputs/som/som-preproc-bush1-%j.out
#SBATCH --error=/home/zs292/slurm_outputs/som/som-preproc-bush1-%j.err

set -euo pipefail

source /share/apps/software/anaconda3/etc/profile.d/conda.sh

SEQ=/home/zs292/datasets/bush1
cd /home/zs292/shape-of-motion/preproc

conda activate som_unidepth
CUDA_VISIBLE_DEVICES=0 python -u compute_metric_depth.py \
  --img-dir $SEQ/images \
  --depth-dir $SEQ/unidepth_disp \
  --intrins-file $SEQ/unidepth_intrins.json

conda activate som
CUDA_VISIBLE_DEVICES=0 python -u compute_depth.py \
  --img_dir $SEQ/images \
  --out_raw_dir $SEQ/depth_anything \
  --out_aligned_dir $SEQ/aligned_depth_anything \
  --model depth-anything \
  --metric_dir $SEQ/unidepth_disp

CUDA_VISIBLE_DEVICES=0 python -u compute_tracks_torch.py \
  --image_dir $SEQ/images \
  --mask_dir $SEQ/masks \
  --out_dir $SEQ/bootstapir \
  --model_type bootstapir

CUDA_VISIBLE_DEVICES=0 python -u recon_with_depth.py \
  --img_dir $SEQ/images \
  --calib $SEQ/unidepth_intrins.json \
  --depth_dir $SEQ/aligned_depth_anything \
  --out_path $SEQ/droid_recon
