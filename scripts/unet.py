"""
unet.py — SpectroGen Diffusion U-Net
U-Net denoising network with:
  - Sinusoidal time embeddings via AdaGN
  - Cross-attention for audio conditioning
  - Self-attention at bottleneck
Built entirely from scratch.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Sinusoidal Time Embedding ─────────────────────────────────────────────────

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal timestep embedding (from DDPM / Attention is All You Need)."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        Args:
            t: (B,) integer timesteps
        Returns:
            emb: (B, dim)
        """
        device    = t.device
        half_dim  = self.dim // 2
        emb       = math.log(10000) / (half_dim - 1)
        emb       = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb       = t[:, None].float() * emb[None, :]
        emb       = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class TimeEmbedding(nn.Module):
    """Projects sinusoidal embedding to model dimension."""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim * 4)
        )

    def forward(self, t):
        return self.net(t)   # (B, dim*4)


# ── Adaptive Group Norm (AdaGN) ───────────────────────────────────────────────

class AdaGN(nn.Module):
    """
    Adaptive Group Normalization.
    Injects time (and optionally audio) conditioning via scale+shift.
    """
    def __init__(self, num_channels, cond_dim, num_groups=32):
        super().__init__()
        self.gn    = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.scale = nn.Linear(cond_dim, num_channels)
        self.shift = nn.Linear(cond_dim, num_channels)

    def forward(self, x, cond):
        """
        Args:
            x:    (B, C, H, W)
            cond: (B, cond_dim) — time or time+audio embedding
        """
        x     = self.gn(x)
        scale = self.scale(cond)[:, :, None, None]  # (B, C, 1, 1)
        shift = self.shift(cond)[:, :, None, None]  # (B, C, 1, 1)
        return x * (1 + scale) + shift


# ── Residual Block ────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Residual block with AdaGN time conditioning."""
    def __init__(self, in_channels, out_channels, cond_dim, num_groups=32, dropout=0.1):
        super().__init__()

        self.norm1 = AdaGN(in_channels,  cond_dim, num_groups)
        self.conv1 = nn.Conv2d(in_channels,  out_channels, 3, padding=1)

        self.norm2 = AdaGN(out_channels, cond_dim, num_groups)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.dropout = nn.Dropout(dropout)
        self.act     = nn.SiLU()

        # Skip connection
        self.skip = nn.Conv2d(in_channels, out_channels, 1) \
                    if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        h = self.act(self.norm1(x, cond))
        h = self.conv1(h)
        h = self.dropout(self.act(self.norm2(h, cond)))
        h = self.conv2(h)
        return h + self.skip(x)


# ── Self-Attention ────────────────────────────────────────────────────────────

class SelfAttention(nn.Module):
    """Multi-head self-attention for spatial feature maps."""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm   = nn.GroupNorm(32, channels)
        self.attn   = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h          = self.norm(x)
        h          = h.view(B, C, H * W).transpose(1, 2)   # (B, H*W, C)
        h, _       = self.attn(h, h, h)
        h          = h.transpose(1, 2).view(B, C, H, W)    # (B, C, H, W)
        return x + h


# ── Cross-Attention (Audio Conditioning) ──────────────────────────────────────

class CrossAttention(nn.Module):
    """
    Cross-attention that injects audio embedding into image features.
    Query = image features, Key/Value = audio embedding
    """
    def __init__(self, channels, audio_dim, num_heads=4):
        super().__init__()
        self.norm_x     = nn.GroupNorm(32, channels)
        self.norm_audio = nn.LayerNorm(audio_dim)

        self.q_proj  = nn.Linear(channels,  channels)
        self.k_proj  = nn.Linear(audio_dim, channels)
        self.v_proj  = nn.Linear(audio_dim, channels)
        self.out_proj = nn.Linear(channels, channels)

        self.num_heads  = num_heads
        self.head_dim   = channels // num_heads
        self.scale      = self.head_dim ** -0.5

    def forward(self, x, audio_emb):
        """
        Args:
            x:         (B, C, H, W) image features
            audio_emb: (B, audio_dim) audio embedding
        Returns:
            (B, C, H, W) conditioned features
        """
        B, C, H, W = x.shape

        # Normalize
        h     = self.norm_x(x).view(B, C, H * W).transpose(1, 2)  # (B, HW, C)
        a     = self.norm_audio(audio_emb).unsqueeze(1)             # (B, 1, audio_dim)

        # Project Q, K, V
        q     = self.q_proj(h)   # (B, HW, C)
        k     = self.k_proj(a)   # (B, 1,  C)
        v     = self.v_proj(a)   # (B, 1,  C)

        # Reshape for multi-head attention
        def reshape(t, seq): return t.view(B, seq, self.num_heads, self.head_dim).transpose(1, 2)
        q     = reshape(q, H * W)   # (B, heads, HW, head_dim)
        k     = reshape(k, 1)       # (B, heads, 1,  head_dim)
        v     = reshape(v, 1)       # (B, heads, 1,  head_dim)

        # Attention
        attn  = (q @ k.transpose(-2, -1)) * self.scale   # (B, heads, HW, 1)
        attn  = attn.softmax(dim=-1)
        out   = (attn @ v)                                # (B, heads, HW, head_dim)
        out   = out.transpose(1, 2).contiguous().view(B, H * W, C)
        out   = self.out_proj(out)                        # (B, HW, C)
        out   = out.transpose(1, 2).view(B, C, H, W)     # (B, C, H, W)

        return x + out


# ── Downsample / Upsample ─────────────────────────────────────────────────────

class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


# ── U-Net ─────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    Diffusion U-Net for SpectroGen.

    Architecture:
        Input: (B, 3, 128, 128) noisy image
        Encoder: 3 downsampling stages [64, 128, 256]
        Bottleneck: ResBlock + SelfAttention + ResBlock
        Decoder: 3 upsampling stages [256, 128, 64]
                 + CrossAttention at 32×32 for audio conditioning
        Output: (B, 3, 128, 128) predicted noise

    Conditioning:
        Time:  sinusoidal embedding → AdaGN at every ResBlock
        Audio: cross-attention at decoder 32×32 feature map
    """
    def __init__(
        self,
        in_channels  = 3,
        base_channels= 64,
        channel_mults= (1, 2, 4),     # → [64, 128, 256]
        time_dim     = 128,
        audio_dim    = 256,
        num_heads    = 4,
        num_groups   = 32,
        dropout      = 0.1,
    ):
        super().__init__()

        self.time_dim = time_dim
        channels      = [base_channels * m for m in channel_mults]  # [64, 128, 256]
        cond_dim      = time_dim * 4   # dimension after time embedding projection

        # ── Time embedding ────────────────────────────────────────────
        self.time_emb = TimeEmbedding(time_dim)

        # ── Input projection ──────────────────────────────────────────
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # ── Encoder ───────────────────────────────────────────────────
        # Stage 1: 128×128 → 64×64,  channels: 64  → 64
        self.enc1      = nn.ModuleList([
            ResBlock(channels[0], channels[0], cond_dim, num_groups, dropout),
            ResBlock(channels[0], channels[0], cond_dim, num_groups, dropout),
        ])
        self.down1     = Downsample(channels[0])

        # Stage 2: 64×64 → 32×32,  channels: 64 → 128
        self.enc2      = nn.ModuleList([
            ResBlock(channels[0], channels[1], cond_dim, num_groups, dropout),
            ResBlock(channels[1], channels[1], cond_dim, num_groups, dropout),
        ])
        self.down2     = Downsample(channels[1])

        # Stage 3: 32×32 → 16×16,  channels: 128 → 256
        self.enc3      = nn.ModuleList([
            ResBlock(channels[1], channels[2], cond_dim, num_groups, dropout),
            ResBlock(channels[2], channels[2], cond_dim, num_groups, dropout),
        ])
        self.down3     = Downsample(channels[2])

        # ── Bottleneck ────────────────────────────────────────────────
        # At 16×16, channels: 256
        self.mid1      = ResBlock(channels[2], channels[2], cond_dim, num_groups, dropout)
        self.mid_attn  = SelfAttention(channels[2], num_heads)
        self.mid2      = ResBlock(channels[2], channels[2], cond_dim, num_groups, dropout)

        # ── Decoder ───────────────────────────────────────────────────
        # Stage 3 up: 16×16 → 32×32,  channels: 256+256 → 256
        self.up3       = Upsample(channels[2])
        self.dec3      = nn.ModuleList([
            ResBlock(channels[2] * 2, channels[2], cond_dim, num_groups, dropout),
            ResBlock(channels[2],     channels[2], cond_dim, num_groups, dropout),
        ])
        # Cross-attention at 32×32 for audio conditioning
        self.cross_attn = CrossAttention(channels[2], audio_dim, num_heads)

        # Stage 2 up: 32×32 → 64×64
        # Input: 256 (from cross_attn) → upsample → 256, cat skip2 (128) → 384
        self.up2       = Upsample(channels[2])          # 256 → 256
        self.dec2      = nn.ModuleList([
            ResBlock(channels[2] + channels[1], channels[1], cond_dim, num_groups, dropout),  # 384 → 128
            ResBlock(channels[1],               channels[1], cond_dim, num_groups, dropout),  # 128 → 128
        ])

        # Stage 1 up: 64×64 → 128×128
        # Input: 128 → upsample → 128, cat skip1 (64) → 192
        self.up1       = Upsample(channels[1])          # 128 → 128
        self.dec1      = nn.ModuleList([
            ResBlock(channels[1] + channels[0], channels[0], cond_dim, num_groups, dropout),  # 192 → 64
            ResBlock(channels[0],               channels[0], cond_dim, num_groups, dropout),  # 64  → 64
        ])

        # ── Output projection ─────────────────────────────────────────
        self.output_norm = nn.GroupNorm(num_groups, base_channels)
        self.output_conv = nn.Conv2d(base_channels, in_channels, 1)

        # Initialize output to zero (common DDPM trick for stable training)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, x, t, audio_emb):
        """
        Args:
            x:         (B, 3, 128, 128) noisy image at timestep t
            t:         (B,) integer timesteps
            audio_emb: (B, audio_dim) audio conditioning embedding
        Returns:
            (B, 3, 128, 128) predicted noise
        """
        # Time conditioning
        t_emb = self.time_emb(t)   # (B, time_dim*4)

        # Input
        x = self.input_conv(x)     # (B, 64, 128, 128)

        # Encoder
        x = self.enc1[0](x, t_emb); x = self.enc1[1](x, t_emb)
        skip1 = x                                                   # (B, 64, 128, 128)
        x     = self.down1(x)                                       # (B, 64,  64,  64)

        x = self.enc2[0](x, t_emb); x = self.enc2[1](x, t_emb)
        skip2 = x                                                   # (B, 128, 64, 64)
        x     = self.down2(x)                                       # (B, 128, 32, 32)

        x = self.enc3[0](x, t_emb); x = self.enc3[1](x, t_emb)
        skip3 = x                                                   # (B, 256, 32, 32)
        x     = self.down3(x)                                       # (B, 256, 16, 16)

        # Bottleneck
        x = self.mid1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid2(x, t_emb)                                    # (B, 256, 16, 16)

        # Decoder
        x = self.up3(x)                                             # (B, 256, 32, 32)
        x = torch.cat([x, skip3], dim=1)                           # (B, 512, 32, 32)
        x = self.dec3[0](x, t_emb); x = self.dec3[1](x, t_emb)   # (B, 256, 32, 32)
        x = self.cross_attn(x, audio_emb)                          # audio conditioning ← 32×32

        x = self.up2(x)                                             # (B, 256, 64, 64)
        x = torch.cat([x, skip2], dim=1)                           # (B, 384, 64, 64)
        x = self.dec2[0](x, t_emb); x = self.dec2[1](x, t_emb)   # (B, 128, 64, 64)

        x = self.up1(x)                                             # (B, 128, 128, 128)
        x = torch.cat([x, skip1], dim=1)                           # (B, 192, 128, 128)
        x = self.dec1[0](x, t_emb); x = self.dec1[1](x, t_emb)   # (B, 64,  128, 128)

        # Output
        x = F.silu(self.output_norm(x))
        x = self.output_conv(x)                                    # (B, 3, 128, 128)
        return x


if __name__ == '__main__':
    # Quick test — run: python unet.py
    B = 2
    x         = torch.randn(B, 3, 128, 128)   # noisy images
    t         = torch.randint(0, 500, (B,))   # timesteps
    audio_emb = torch.randn(B, 256)           # audio embeddings

    model  = UNet(in_channels=3, base_channels=64, audio_dim=256)
    out    = model(x, t, audio_emb)

    print(f"Input shape:       {x.shape}")          # (2, 3, 128, 128)
    print(f"Output shape:      {out.shape}")         # (2, 3, 128, 128)
    assert out.shape == x.shape, "Shape mismatch!"

    total  = sum(p.numel() for p in model.parameters())
    print(f"Total params:      {total:,}")           # ~50-60M
    print("✅ U-Net test passed!")
