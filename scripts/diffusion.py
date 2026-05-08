"""
diffusion.py — SpectroGen DDPM
Cosine noise schedule, forward process, and epsilon-prediction loss.
Also includes DDIM sampler for fast inference.
"""

import torch
import torch.nn as nn
import numpy as np


def cosine_beta_schedule(T, s=0.008):
    """
    Cosine noise schedule from 'Improved DDPM' (Nichol & Dhariwal, 2021).
    Smoother than linear — avoids too much noise at the end.
    """
    steps     = T + 1
    x         = torch.linspace(0, T, steps)
    alphas_cp = torch.cos(((x / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cp = alphas_cp / alphas_cp[0]
    betas     = 1 - (alphas_cp[1:] / alphas_cp[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    """Linear noise schedule from original DDPM (Ho et al., 2020)."""
    return torch.linspace(beta_start, beta_end, T)


class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model.

    Wraps the U-Net and audio encoder with:
        - Noise schedule (cosine by default)
        - Forward noising process q(x_t | x_0)
        - Training loss (epsilon prediction)
        - DDPM reverse sampling
        - DDIM fast sampling
    """
    def __init__(self, unet, encoder, T=500, schedule='cosine', device='cuda'):
        super().__init__()
        self.unet    = unet
        self.encoder = encoder
        self.T       = T
        self.device  = device

        # ── Noise schedule ────────────────────────────────────────────
        if schedule == 'cosine':
            betas = cosine_beta_schedule(T)
        else:
            betas = linear_beta_schedule(T)

        alphas      = 1.0 - betas
        alphas_cp   = torch.cumprod(alphas, dim=0)          # ᾱ_t
        alphas_cp_prev = torch.cat([torch.tensor([1.0]), alphas_cp[:-1]])

        # Register as buffers (moved to device with model)
        self.register_buffer('betas',           betas)
        self.register_buffer('alphas',          alphas)
        self.register_buffer('alphas_cp',       alphas_cp)
        self.register_buffer('alphas_cp_prev',  alphas_cp_prev)
        self.register_buffer('sqrt_alphas_cp',         alphas_cp.sqrt())
        self.register_buffer('sqrt_one_minus_alphas_cp', (1 - alphas_cp).sqrt())
        self.register_buffer('log_one_minus_alphas_cp', (1 - alphas_cp).log())
        self.register_buffer('sqrt_recip_alphas_cp',   (1.0 / alphas_cp).sqrt())
        self.register_buffer('sqrt_recip_alphas_cp_m1', (1.0 / alphas_cp - 1).sqrt())

        # Posterior variance q(x_{t-1} | x_t, x_0)
        posterior_var = betas * (1.0 - alphas_cp_prev) / (1.0 - alphas_cp)
        self.register_buffer('posterior_var',      posterior_var)
        self.register_buffer('posterior_log_var',  torch.log(posterior_var.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_c1',
            betas * alphas_cp_prev.sqrt() / (1.0 - alphas_cp))
        self.register_buffer('posterior_mean_c2',
            (1.0 - alphas_cp_prev) * alphas.sqrt() / (1.0 - alphas_cp))

    # ── Forward Process ───────────────────────────────────────────────────────

    def q_sample(self, x0, t, noise=None):
        """
        Forward noising: sample x_t ~ q(x_t | x_0)
        x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε

        Args:
            x0:    (B, C, H, W) clean images in [-1, 1]
            t:     (B,) timesteps
            noise: (B, C, H, W) optional pre-sampled noise
        Returns:
            x_t:   (B, C, H, W) noisy images
            noise: (B, C, H, W) the noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_acp     = self.sqrt_alphas_cp[t][:, None, None, None]
        sqrt_1m_acp  = self.sqrt_one_minus_alphas_cp[t][:, None, None, None]

        x_t = sqrt_acp * x0 + sqrt_1m_acp * noise
        return x_t, noise

    # ── Training Loss ─────────────────────────────────────────────────────────

    def loss(self, x0, mel):
        """
        Compute DDPM training loss (MSE on epsilon prediction).

        Args:
            x0:  (B, 3, H, W) clean images in [-1, 1]
            mel: (B, 1, 64, T) mel-spectrograms in [0, 1]
        Returns:
            loss: scalar
        """
        B = x0.shape[0]

        # Sample random timesteps
        t     = torch.randint(0, self.T, (B,), device=x0.device)

        # Forward process
        noise = torch.randn_like(x0)
        x_t, noise = self.q_sample(x0, t, noise)

        # Audio conditioning
        audio_emb = self.encoder(mel)   # (B, embed_dim)

        # Predict noise
        pred_noise = self.unet(x_t, t, audio_emb)

        # MSE loss
        loss = nn.functional.mse_loss(pred_noise, noise)
        return loss

    # ── DDPM Reverse Sampling ─────────────────────────────────────────────────

    @torch.no_grad()
    def p_sample(self, x_t, t, audio_emb):
        """One step of DDPM reverse process."""
        t_batch    = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        pred_noise = self.unet(x_t, t_batch, audio_emb)

        # Predict x_0
        x0_pred    = (self.sqrt_recip_alphas_cp[t] * x_t
                      - self.sqrt_recip_alphas_cp_m1[t] * pred_noise)
        x0_pred    = x0_pred.clamp(-1, 1)

        # Posterior mean
        mean = (self.posterior_mean_c1[t] * x0_pred
               + self.posterior_mean_c2[t] * x_t)

        # Add noise (except at t=0)
        if t > 0:
            noise = torch.randn_like(x_t)
            var   = (0.5 * self.posterior_log_var[t]).exp()
            mean  = mean + var * noise

        return mean

    @torch.no_grad()
    def sample_ddpm(self, mel, shape=None):
        """
        Full DDPM sampling (T steps — slow but exact).
        Args:
            mel:   (B, 1, 64, T) conditioning mel-spectrograms
            shape: output image shape (default: (B, 3, 128, 128))
        """
        B         = mel.shape[0]
        shape     = shape or (B, 3, 128, 128)
        audio_emb = self.encoder(mel)

        x = torch.randn(shape, device=self.device)
        for t in reversed(range(self.T)):
            x = self.p_sample(x, t, audio_emb)
        return x

    # ── DDIM Fast Sampling ────────────────────────────────────────────────────

    @torch.no_grad()
    def sample_ddim(self, mel, steps=50, eta=0.0, shape=None):
        """
        DDIM sampling (fast, deterministic when eta=0).
        steps: number of denoising steps (50 gives good quality)
        eta:   0 = deterministic, 1 = stochastic (= DDPM)

        Args:
            mel:   (B, 1, 64, T) conditioning mel-spectrograms
            steps: number of DDIM steps
            eta:   stochasticity (0 = deterministic)
        Returns:
            (B, 3, 128, 128) generated images in [-1, 1]
        """
        B         = mel.shape[0]
        shape     = shape or (B, 3, 128, 128)
        audio_emb = self.encoder(mel)

        # Subsample timesteps
        timesteps  = torch.linspace(self.T - 1, 0, steps, dtype=torch.long)
        x          = torch.randn(shape, device=self.device)

        for i, t in enumerate(timesteps):
            t_val      = t.item()
            t_batch    = torch.full((B,), t_val, device=self.device, dtype=torch.long)
            pred_noise = self.unet(x, t_batch, audio_emb)

            acp        = self.alphas_cp[t_val]
            acp_prev   = self.alphas_cp[timesteps[i + 1].item()] if i < len(timesteps) - 1 \
                         else torch.tensor(1.0)

            # Predict x_0
            x0_pred    = (x - (1 - acp).sqrt() * pred_noise) / acp.sqrt()
            x0_pred    = x0_pred.clamp(-1, 1)

            # DDIM update
            sigma      = eta * ((1 - acp_prev) / (1 - acp) * (1 - acp / acp_prev)).sqrt()
            noise      = torch.randn_like(x) if eta > 0 else 0

            x = acp_prev.sqrt() * x0_pred \
                + (1 - acp_prev - sigma ** 2).clamp(min=0).sqrt() * pred_noise \
                + sigma * noise

        return x


if __name__ == '__main__':
    # Quick test — run: python diffusion.py
    import sys
    sys.path.insert(0, '.')
    from unet    import UNet
    from encoder import AudioEncoder

    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    B, T_audio = 2, 1292
    unet       = UNet(in_channels=3, base_channels=64, audio_dim=256).to(device)
    encoder    = AudioEncoder(embed_dim=256).to(device)
    ddpm       = DDPM(unet, encoder, T=500, schedule='cosine', device=device).to(device)

    # Test forward loss
    x0  = torch.randn(B, 3, 128, 128).to(device)
    mel = torch.rand(B, 1, 64, T_audio).to(device)

    loss = ddpm.loss(x0, mel)
    print(f"Training loss:   {loss.item():.4f}")

    # Test DDIM sampling (just 5 steps for speed)
    samples = ddpm.sample_ddim(mel, steps=5)
    print(f"DDIM output:     {samples.shape}")    # (2, 3, 128, 128)
    print(f"Value range:     [{samples.min():.2f}, {samples.max():.2f}]")

    total = sum(p.numel() for p in ddpm.parameters())
    print(f"Total params:    {total:,}")
    print("✅ DDPM test passed!")
