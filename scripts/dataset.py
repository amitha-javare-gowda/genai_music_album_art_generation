"""
dataset.py — SpectroGen PyTorch Dataset
Loads (mel-spectrogram, album cover) pairs for training.
"""

import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class SpectroGenDataset(Dataset):
    def __init__(self, covers_dir, spectro_dir, split='train',
                 val_ratio=0.1, test_ratio=0.1, seed=42, augment=True):
        """
        Args:
            covers_dir:  path to covers_128/ folder
            spectro_dir: path to spectrograms/ folder
            split:       'train', 'val', or 'test'
            augment:     apply augmentation (train only)
        """
        self.covers_dir  = covers_dir
        self.spectro_dir = spectro_dir
        self.augment     = augment and (split == 'train')

        # Find valid pairs (must have both cover AND spectrogram)
        cover_ids   = set(f.replace('.jpg', '') for f in os.listdir(covers_dir)  if f.endswith('.jpg'))
        spectro_ids = set(f.replace('.npy', '') for f in os.listdir(spectro_dir) if f.endswith('.npy'))
        all_ids     = sorted(cover_ids & spectro_ids)

        if len(all_ids) == 0:
            raise ValueError(f"No valid pairs found in {covers_dir} and {spectro_dir}")

        print(f"Total valid pairs: {len(all_ids)}")

        # Reproducible split
        rng     = np.random.default_rng(seed)
        indices = rng.permutation(len(all_ids))
        n_test  = int(len(all_ids) * test_ratio)
        n_val   = int(len(all_ids) * val_ratio)

        if split == 'test':
            self.ids = [all_ids[i] for i in indices[:n_test]]
        elif split == 'val':
            self.ids = [all_ids[i] for i in indices[n_test:n_test + n_val]]
        else:
            self.ids = [all_ids[i] for i in indices[n_test + n_val:]]

        print(f"Split '{split}': {len(self.ids)} samples")

        # ── Image transforms ──────────────────────────────────────────
        # Diffusion models expect images in [-1, 1]
        if self.augment:
            self.img_transform = T.Compose([
                T.Resize((128, 128)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.RandomRotation(degrees=10),
                T.RandomCrop(128, padding=8, padding_mode='reflect'),
                T.ToTensor(),                        # [0, 1]
                T.Normalize([0.5, 0.5, 0.5],         # → [-1, 1]
                            [0.5, 0.5, 0.5])
            ])
        else:
            self.img_transform = T.Compose([
                T.Resize((128, 128)),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5],
                            [0.5, 0.5, 0.5])
            ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        track_id = self.ids[idx]

        # Load cover art → (3, 128, 128) in [-1, 1]
        img_path = os.path.join(self.covers_dir, f'{track_id}.jpg')
        img      = Image.open(img_path).convert('RGB')
        img      = self.img_transform(img)

        # Load mel-spectrogram → (1, 64, T) in [0, 1]
        mel_path = os.path.join(self.spectro_dir, f'{track_id}.npy')
        mel      = np.load(mel_path)                  # (64, T)
        mel      = torch.tensor(mel).unsqueeze(0)     # (1, 64, T)

        return img, mel


def get_dataloaders(covers_dir, spectro_dir, batch_size=32,
                    num_workers=4, val_ratio=0.1, test_ratio=0.1):
    """
    Returns train, val, test DataLoaders.
    """
    train_ds = SpectroGenDataset(covers_dir, spectro_dir, split='train',
                                  val_ratio=val_ratio, test_ratio=test_ratio, augment=True)
    val_ds   = SpectroGenDataset(covers_dir, spectro_dir, split='val',
                                  val_ratio=val_ratio, test_ratio=test_ratio, augment=False)
    test_ds  = SpectroGenDataset(covers_dir, spectro_dir, split='test',
                                  val_ratio=val_ratio, test_ratio=test_ratio, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Quick test — run: python dataset.py
    import sys

    covers_dir  = sys.argv[1] if len(sys.argv) > 1 else '../data/covers_128'
    spectro_dir = sys.argv[2] if len(sys.argv) > 2 else '../data/spectrograms'

    train_loader, val_loader, test_loader = get_dataloaders(
        covers_dir, spectro_dir, batch_size=8, num_workers=0
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    # Check one batch
    imgs, mels = next(iter(train_loader))
    print(f"\nImage batch shape:       {imgs.shape}")    # (8, 3, 128, 128)
    print(f"Spectrogram batch shape: {mels.shape}")    # (8, 1, 64, T)
    print(f"Image value range:       [{imgs.min():.2f}, {imgs.max():.2f}]")  # [-1, 1]
    print(f"Mel value range:         [{mels.min():.2f}, {mels.max():.2f}]")  # [0, 1]
    print("\n✅ Dataset test passed!")
