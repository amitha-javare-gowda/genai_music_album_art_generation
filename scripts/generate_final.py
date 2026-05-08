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

encoder = AudioEncoder(embed_dim=256).to(device)
unet    = UNet(in_channels=3, base_channels=64, audio_dim=256).to(device)
ddpm    = DDPM(unet, encoder, T=500, schedule='linear', device=device).to(device)

ckpt = torch.load('/projects/p32506/spectrogen/checkpoints/spectrogen_final_best.pt',
                  map_location=device)
ddpm.load_state_dict(ckpt['model'])
ddpm.eval()
print(f"Loaded epoch {ckpt['epoch']+1} | val_loss: {ckpt['loss']:.4f}")

spectro_dir = '/projects/p32506/spectrogen/data/spectrograms'
spec_files  = sorted(os.listdir(spectro_dir))[:8]
os.makedirs('/projects/p32506/spectrogen/samples_final', exist_ok=True)

print("Generating with 200 DDIM steps...")
with torch.no_grad():
    for i, fname in enumerate(spec_files):
        mel    = np.load(os.path.join(spectro_dir, fname))
        mel    = torch.tensor(mel).unsqueeze(0).unsqueeze(0).to(device)
        sample = ddpm.sample_ddim(mel, steps=200)
        sample = sample[0].cpu()
        sample = (sample - sample.min()) / (sample.max() - sample.min() + 1e-8)
        sample = (sample.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img    = Image.fromarray(sample).resize((256, 256), Image.LANCZOS)
        img.save(f'/projects/p32506/spectrogen/samples_final/sample_{i:02d}.png')
        print(f"  sample_{i:02d}.png")

print("Done!")
