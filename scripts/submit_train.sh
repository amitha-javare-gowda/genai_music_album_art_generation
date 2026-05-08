#!/bin/bash
#SBATCH --account=p32506
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --output=/projects/p32506/spectrogen/logs/train_%j.log

echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"

module purge
source /software/anaconda3/2018.12/etc/profile.d/conda.sh
conda activate genai

nvidia-smi

PROJECT_DIR=/projects/p32506/spectrogen
cd $PROJECT_DIR/scripts

python train.py \
    --covers_dir    $PROJECT_DIR/data/covers_128 \
    --spectro_dir   $PROJECT_DIR/data/spectrograms \
    --batch_size    32 \
    --epochs        300 \
    --lr            2e-4 \
    --T             500 \
    --schedule      cosine \
    --embed_dim     256 \
    --base_channels 64 \
    --num_workers   8 \
    --ckpt_dir      $PROJECT_DIR/checkpoints \
    --log_every     100 \
    --sample_every  1000 \
    --save_every    5 \
    --run_name      spectrogen_baseline \
    --wandb_project spectrogen

echo "Finished: $(date)"
