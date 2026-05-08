# SpectroGen — Music Album Art Generation via Spectrogram Diffusion

Cross-modal image generation system that synthesizes album cover artwork 
conditioned on the acoustic properties of a music track.

## Architecture
- **Audio Encoder**: CNN mel-spectrogram → 256-dim embedding
- **Diffusion U-Net**: DDPM with cross-attention audio conditioning
- **Sampler**: DDIM (50 steps for fast inference)

## Dataset
- **Audio**: FMA Small/Medium (~5,461 paired samples)
- **Images**: MusicBrainz Cover Art Archive (128×128)
- **Genres**: 8 (Rock, Electronic, Hip-Hop, Folk, Pop, Experimental, Instrumental, International)

## Results
- Best val_loss: 0.0317 (linear schedule, T=500, embed_dim=256)
- Trained for 300+ epochs on Northwestern Quest HPC (H100 GPU)
- Hyperparameter grid search across 8 configurations

## Extra Criteria
- **Hyperparameter Tuning**: Grid search over schedule × T × embed_dim (8 configs)
- **W&B Logging**: Loss curves, sample images, sweep comparison
- **Streamlit GUI**: Upload audio → generate album art

## How to Run

### Training
```bash
python scripts/train.py \
    --covers_dir data/covers_128 \
    --spectro_dir data/spectrograms \
    --schedule linear --T 500 --embed_dim 256 \
    --epochs 500 --run_name spectrogen_final
```

### GUI Demo
```bash
streamlit run scripts/app.py
```

## W&B Dashboard
https://wandb.ai/amithajavaregowda2026-northwestern-university/spectrogen
