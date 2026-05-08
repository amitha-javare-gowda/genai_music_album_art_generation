"""
generate_samples.py — Generate and save sample images from trained model
"""
import torch
import numpy as np
from PIL import Image
import os, sys
sys.path.insert(0, '.')

from encoder   import AudioEncoder
from unet      import UNet
from diffusion import DDPM

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Load model
encoder = AudioEncoder(embed_dim=256).to(device)
unet    = UNet(in_channels=3, base_channels=64, audio_dim=256).to(device)
ddpm    = DDPM(unet, encoder, T=500, schedule='cosine', device=device).to(device)

ckpt_path = '/projects/p32506/spectrogen/checkpoints/spectrogen_baseline_best.pt'
ckpt      = torch.load(ckpt_path, map_location=device)
ddpm.load_state_dict(ckpt['model'])
ddpm.eval()
print(f"Loaded checkpoint from epoch {ckpt['epoch']+1}")

# Load some real spectrograms
spectro_dir = '/projects/p32506/spectrogen/data/spectrograms'
spec_files  = os.listdir(spectro_dir)[:8]

os.makedirs('/projects/p32506/spectrogen/samples', exist_ok=True)

print("Generating samples with 200 DDIM steps...")
with torch.no_grad():
    for i, fname in enumerate(spec_files):
        # Load spectrogram
        mel  = np.load(os.path.join(spectro_dir, fname))
        mel  = torch.tensor(mel).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 64, T)

        # Generate with MORE steps for better quality
        sample = ddpm.sample_ddim(mel, steps=200)  # 200 steps vs 50
        sample = (sample.clamp(-1, 1) + 1) / 2    # → [0, 1]
        sample = (sample[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        img = Image.fromarray(sample)
        img = img.resize((256, 256), Image.LANCZOS)  # upscale for display
        img.save(f'/projects/p32506/spectrogen/samples/sample_{i:02d}.png')
        print(f"Saved sample_{i:02d}.png")

print("✅ Done! Check /projects/p32506/spectrogen/samples/")
