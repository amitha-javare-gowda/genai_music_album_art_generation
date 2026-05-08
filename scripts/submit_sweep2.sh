#!/bin/bash
#SBATCH --account=p32506
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --output=/projects/p32506/spectrogen/logs/sweep2_%j.log

module purge
source /software/anaconda3/2018.12/etc/profile.d/conda.sh
conda activate genai

cd /projects/p32506/spectrogen/scripts
wandb agent amithajavaregowda2026-northwestern-university/spectrogen/i1prfn00
