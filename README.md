# SpectroGen: Music Album Art Generation via Spectrogram Diffusion

## Live Demo
https://genaimusicalbumartgeneration-amitha.streamlit.app

## W&B Project
https://wandb.ai/amithajavaregowda2026-northwestern-university/spectrogen

---

## Project Overview

SpectroGen is a cross-modal image generation system that synthesizes album cover artwork conditioned on the acoustic properties of a music track. Given a mel-spectrogram of a 30-second audio clip, the model learns to generate a visual representation of that track's artistic identity including mood, genre, and energy.

The conditioning signal is audio rather than text or class labels. The model must discover audio-visual correlations entirely from paired data. No pretrained weights are used at any stage. Everything is trained from scratch.

---

## Model Architecture

### Audio Encoder
A lightweight CNN that maps log-mel spectrograms to a fixed-length embedding vector.

- Input: log-mel spectrogram, shape (64 mel bins x 1292 time frames)
- Architecture: 5 strided Conv2D layers with BatchNorm and GELU activations
- Global average pooling followed by a linear projection to 256 dimensions
- Trained jointly with the diffusion U-Net

### Diffusion U-Net
The denoising backbone trained to predict added noise given a noisy image, timestep, and audio embedding.

- 3-stage encoder with channels [64, 128, 256] and residual blocks
- Bottleneck with self-attention (4 heads) at 16x16 feature map
- 3-stage decoder with skip connections from encoder
- Time conditioning: sinusoidal timestep embedding injected via Adaptive Group Normalization (AdaGN) at every residual block
- Audio conditioning: 256-dim audio embedding injected into the decoder at 32x32 via cross-attention (image features = query, audio = key and value)

### Noise Schedule and Sampling
- Training: linear noise schedule, T=500 timesteps, epsilon-prediction MSE loss
- Optimizer: AdamW with lr=2e-4, weight decay=1e-4, linear warmup, mixed precision fp16
- Inference: DDIM sampler at 100-200 steps (10-20x faster than full DDPM)

---

## Dataset

| Property | Details |
|---|---|
| Audio | FMA Medium, 25,000 tracks, 30-second MP3 clips at 128kbps |
| Images | MusicBrainz Cover Art Archive, CC-licensed artwork |
| Matching | MusicBrainz release IDs embedded in FMA metadata |
| Valid pairs | 5,461 clean pairs (~22% match rate) |
| Resolution | 128 x 128 px |
| Genres | Rock, Electronic, Hip-Hop, Folk, Pop, Experimental, Instrumental, International |
| Split | 80% train / 10% val / 10% test |

Augmentation applied to training set only: random horizontal flip (p=0.5), color jitter (brightness/contrast/saturation +/-20%), random rotation (+/-10 degrees), random crop with reflection padding.

### How to Download the Data

**FMA Medium (audio):**
- GitHub and download scripts: https://github.com/mdeff/fma
- Direct download (22GB): https://os.unil.cloud.switch.ch/fma/fma_medium.zip
- Metadata CSV (342MB): https://os.unil.cloud.switch.ch/fma/fma_metadata.zip

**MusicBrainz Cover Art Archive (images):**
- API documentation: https://musicbrainz.org/doc/Cover_Art_Archive/API
- MusicBrainz main site: https://musicbrainz.org
- Image endpoint: https://coverartarchive.org/release/{mbid}/front-250

---

## Extra Criteria Pursued

### 1. Hyperparameter Tuning via Grid Search (W&B Sweeps)

A full 2x2x2 grid search across 8 configurations:

| Parameter | Values |
|---|---|
| Noise schedule | cosine, linear |
| Timesteps T | 250, 500 |
| Audio embedding dim | 128, 256 |

**Final results:**

| Config | T | embed_dim | Val Loss |
|---|---|---|---|
| linear_T500_d256 | 500 | 256 | 0.0317 (best) |
| cosine_T500_d256 | 500 | 256 | 0.0394 |
| linear_T500_d128 | 500 | 128 | 0.0365 |
| cosine_T500_d128 | 500 | 128 | 0.0366 |
| linear_T250_d256 | 250 | 256 | 0.0430 |
| cosine_T250_d256 | 250 | 256 | 0.0421 |
| linear_T250_d128 | 250 | 128 | 0.0436 |
| cosine_T250_d128 | 250 | 128 | 0.0415 |

The winning config (linear, T=500, d=256) was used to train the final model for 500 epochs, achieving val_loss of 0.024, a 45% improvement over the cosine baseline of 0.044.

W&B Sweep: https://wandb.ai/amithajavaregowda2026-northwestern-university/spectrogen/sweeps/ok1e67ve

### 2. Interactive GUI (Streamlit)

A Streamlit web app allows interactive demo:
- Upload any MP3 or WAV file
- Visualize the extracted mel-spectrogram
- Generate album art via DDIM inference with adjustable steps via sidebar slider
- Download the generated image as PNG

Run locally:
```bash
pip install streamlit gdown librosa torch torchvision
streamlit run scripts/app.py
```

---

## Difficulties Faced

### The Gap Between Loss and Visual Quality

The loss curves looked healthy. Both train and val loss dropped sharply from around 0.8 in the first few thousand steps and settled into a slow decline. But the actual generated images told a different story. Even at 300 epochs the outputs were abstract blobs of color with no compositional structure.

This is a real tension with diffusion models on small datasets. The MSE loss on predicted noise can be low even when images lack structure, because predicting average noise statistics across 5,461 examples is easier than learning the full visual distribution. The model learned what colors album art tends to use but not how album art is composed.

What would genuinely fix this going forward is more data. FMA Large has 106k tracks and is partially downloaded on Quest. With around 60k valid pairs the model would encounter far more visual diversity and likely produce more structured outputs. Adding a perceptual loss like LPIPS alongside MSE would also push the model to preserve edges and structure. Longer term, running diffusion in a pretrained latent space rather than pixel space would make the learning task much more tractable at this data scale.

### U-Net Channel Dimension Bug

After inserting cross-attention at the 32x32 stage, the next upsampling block crashed with a shape mismatch. It expected 128 input channels but received 256 from the cross-attention output. This was confusing at first because cross-attention preserves the input channel count, so the error appeared one stage later than expected.

The fix required manually tracing every tensor shape through the decoder. The upsampling block was redefined to accept 256 channels, which after concatenation with the 128-channel skip connection gives 384 channels into the next ResBlock, which then reduces to 128.

### Hyperparameter Sweep Checkpoint Conflicts

When the grid search ran configs with different T and embed_dim values, each run tried to resume from the previous checkpoint. A checkpoint saved with T=250 has noise schedule buffers of shape (250,) which cannot be loaded into a model initialized with T=500. This caused the sweep to crash repeatedly.

The fix was two things: give each sweep run a unique checkpoint name based on its config (e.g. sweep_linear_T500_d256) so runs never share checkpoints, and wrap the checkpoint loading in a try/except so architecture mismatches restart from scratch instead of crashing the whole sweep.

### Image Generation Quality

The generated images are abstract color compositions rather than recognizable album art. Looking at the W&B sample comparisons, the real covers have strong compositional structure, bold graphic design, and clear visual intent. The generated outputs have texture and distinct color regions but no coherent structure.

The model is doing something meaningful. Different audio inputs produce different outputs. Electronic FMA tracks produced reds and oranges, and folk tracks produced warmer earthier colors. The audio conditioning is influencing the generation. But the model has not learned how to compose an image, only what color palette loosely corresponds to a genre.

This comes down to scale. With 5,461 training pairs the model sees each image roughly 500 times across training. That is enough to learn color statistics but nowhere near enough to learn compositional structure. Album art is also an extremely diverse visual category with no consistent structure even within a genre, which makes the learning problem harder than something like faces or bedrooms where layouts follow predictable patterns.

For the images to actually look like album art the project would need at minimum 50,000 to 100,000 pairs, a perceptual loss to encourage structural coherence, and likely a move to latent diffusion rather than pixel space. These are real research-scale requirements, not something solvable with more training epochs on the current dataset.

---

## Training Details

| Run | Schedule | T | embed_dim | Epochs | Val Loss | Hardware |
|---|---|---|---|---|---|---|
| Baseline | cosine | 500 | 256 | 300 | 0.044 | H100 |
| Grid sweep x8 | various | 250/500 | 128/256 | 50 each | 0.032 to 0.044 | A100 |
| Final | linear | 500 | 256 | 500 | 0.024 | H100 |

Total compute: approximately 6 hours on Northwestern Quest HPC (H100 + A100 GPUs).

---

## Model Checkpoint

The trained model checkpoint (214MB) is available here:
https://drive.google.com/file/d/1KA3D6XeJI5K_hYnG1fV9mACKW-TxZ_RD/view?usp=share_link

The Streamlit app downloads this automatically on first launch via gdown.

---

## How to Run

**Install dependencies:**
```bash
pip install torch torchvision torchaudio librosa streamlit gdown matplotlib Pillow numpy wandb
```

**Run the GUI:**
```bash
streamlit run scripts/app.py
```

**Run training:**
```bash
python scripts/train.py \
    --covers_dir data/covers_128 \
    --spectro_dir data/spectrograms \
    --schedule linear \
    --T 500 \
    --embed_dim 256 \
    --epochs 500 \
    --run_name spectrogen_final
```

**Run hyperparameter sweep:**
```bash
wandb sweep scripts/sweep_config.yaml
wandb agent USERNAME/spectrogen/SWEEP_ID
```


