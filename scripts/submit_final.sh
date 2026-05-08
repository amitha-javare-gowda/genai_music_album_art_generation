#!/bin/bash
#SBATCH --account=p32506
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --output=/projects/p32506/spectrogen/logs/final_%j.log

module purge
source /software/anaconda3/2018.12/etc/profile.d/conda.sh
conda activate genai

cd /projects/p32506/spectrogen/scripts

python train.py \
    --covers_dir    /projects/p32506/spectrogen/data/covers_128 \
    --spectro_dir   /projects/p32506/spectrogen/data/spectrograms \
    --batch_size    32 \
    --epochs        500 \
    --lr            2e-4 \
    --T             500 \
    --schedule      linear \
    --embed_dim     256 \
    --base_channels 64 \
    --num_workers   8 \
    --ckpt_dir      /projects/p32506/spectrogen/checkpoints \
    --log_every     100 \
    --sample_every  2000 \
    --save_every    50 \
    --run_name      spectrogen_final \
    --wandb_project spectrogen
