#!/bin/bash
#SBATCH --job-name=som-preproc-bush1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=/home/zs292/outputs/som-preproc-bush1-%j.out
#SBATCH --error=/home/zs292/outputs/som-preproc-bush1-%j.err

source ~/.bashrc
conda activate som

cd /home/zs292/shape-of-motion/preproc

python -u process_custom.py --img-dirs /home/zs292/datasets/custom/images/bush1 --gpus 0
