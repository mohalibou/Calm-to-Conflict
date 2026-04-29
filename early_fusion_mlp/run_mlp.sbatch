#!/bin/bash
#SBATCH --account=msoleyma_1026
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

module load conda
source activate calm_conflict

python early_fusion_mlp.py
