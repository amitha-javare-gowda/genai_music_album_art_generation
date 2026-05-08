"""
encoder.py — SpectroGen Audio Encoder
CNN encoder that maps mel-spectrograms to a fixed-dim embedding vector.
Trained jointly with the diffusion U-Net.
"""

import torch
import torch.nn as nn


class AudioEncoder(nn.Module):
    """
    Lightweight CNN encoder for mel-spectrograms.
    Input:  (B, 1, 64, T)  — batch of log-mel spectrograms
    Output: (B, embed_dim) — fixed-length audio embedding
    """
    def __init__(self, embed_dim=256, in_channels=1):
        super().__init__()

        self.encoder = nn.Sequential(
            # Block 1: (B, 1,   64, T) → (B, 32,  32, T//2)
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),

            # Block 2: (B, 32,  32, T//2) → (B, 64,  16, T//4)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),

            # Block 3: (B, 64,  16, T//4) → (B, 128, 8,  T//8)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),

            # Block 4: (B, 128, 8,  T//8) → (B, 256, 4,  T//16)
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),

            # Block 5: (B, 256, 4,  T//16) → (B, 256, 2,  T//32)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        # Global average pooling → (B, 256)
        self.pool    = nn.AdaptiveAvgPool2d((1, 1))

        # Project to embed_dim
        self.project = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, mel):
        """
        Args:
            mel: (B, 1, 64, T) mel-spectrogram in [0, 1]
        Returns:
            embedding: (B, embed_dim)
        """
        x = self.encoder(mel)       # (B, 256, H', W')
        x = self.pool(x)            # (B, 256, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 256)
        x = self.project(x)         # (B, embed_dim)
        return x


class AudioEncoderWithClassifier(nn.Module):
    """
    AudioEncoder with an auxiliary genre classification head.
    Used for warm-up training before coupling to diffusion.

    Training strategy:
        Phase 1 (warm-up): train with genre_loss only (5k steps)
        Phase 2 (joint):   freeze classifier, train with diffusion loss
    """
    def __init__(self, embed_dim=256, num_genres=16):
        super().__init__()
        self.encoder    = AudioEncoder(embed_dim=embed_dim)
        self.classifier = nn.Linear(embed_dim, num_genres)

    def forward(self, mel):
        embedding = self.encoder(mel)
        logits    = self.classifier(embedding)
        return embedding, logits

    def get_embedding(self, mel):
        """Return just the embedding (used during diffusion training)."""
        return self.encoder(mel)


if __name__ == '__main__':
    # Quick test — run: python encoder.py
    B, T = 4, 1292  # batch=4, time=1292 frames (30s at hop=512)

    mel      = torch.randn(B, 1, 64, T)
    encoder  = AudioEncoder(embed_dim=256)
    emb      = encoder(mel)

    print(f"Input shape:     {mel.shape}")       # (4, 1, 64, 1292)
    print(f"Embedding shape: {emb.shape}")       # (4, 256)

    # Test with classifier
    enc_cls    = AudioEncoderWithClassifier(embed_dim=256, num_genres=16)
    emb, logits = enc_cls(mel)
    print(f"Embedding:       {emb.shape}")       # (4, 256)
    print(f"Genre logits:    {logits.shape}")    # (4, 16)

    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nEncoder params:  {total_params:,}")
    print("✅ Encoder test passed!")
