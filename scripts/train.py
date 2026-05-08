"""
train.py — SpectroGen Training Script
Full training loop with:
  - Mixed precision (fp16)
  - W&B logging
  - Checkpointing
  - Validation loss tracking
  - Gradient clipping
  - Sweep-aware unique run naming
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import wandb
from tqdm import tqdm

from dataset  import get_dataloaders
from encoder  import AudioEncoder
from unet     import UNet
from diffusion import DDPM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--covers_dir',    type=str,   default='../data/covers_128')
    parser.add_argument('--spectro_dir',   type=str,   default='../data/spectrograms')
    parser.add_argument('--num_workers',   type=int,   default=4)
    parser.add_argument('--base_channels', type=int,   default=64)
    parser.add_argument('--embed_dim',     type=int,   default=256)
    parser.add_argument('--time_dim',      type=int,   default=128)
    parser.add_argument('--dropout',       type=float, default=0.1)
    parser.add_argument('--T',             type=int,   default=500)
    parser.add_argument('--schedule',      type=str,   default='cosine',
                        choices=['cosine', 'linear'])
    parser.add_argument('--epochs',        type=int,   default=100)
    parser.add_argument('--batch_size',    type=int,   default=32)
    parser.add_argument('--lr',            type=float, default=2e-4)
    parser.add_argument('--grad_clip',     type=float, default=1.0)
    parser.add_argument('--warmup_steps',  type=int,   default=500)
    parser.add_argument('--run_name',      type=str,   default='spectrogen')
    parser.add_argument('--ckpt_dir',      type=str,   default='../checkpoints')
    parser.add_argument('--log_every',     type=int,   default=100)
    parser.add_argument('--sample_every',  type=int,   default=1000)
    parser.add_argument('--save_every',    type=int,   default=5)
    parser.add_argument('--wandb_project', type=str,   default='spectrogen')
    return parser.parse_args()


def save_checkpoint(model, optimizer, scaler, epoch, step, loss, path):
    """Save model, optimizer and scaler state to disk."""
    
    torch.save({
        'epoch':     epoch,
        'step':      step,
        'model':     model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler':    scaler.state_dict(),
        'loss':      loss,
    }, path)
    print(f"Saved checkpoint: {path}")


def load_checkpoint(path, model, optimizer, scaler):
    """Load model, optimizer and scaler state from checkpoint file."""
    
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scaler.load_state_dict(ckpt['scaler'])
    return ckpt['epoch'], ckpt['step'], ckpt['loss']


@torch.no_grad()
def log_samples(ddpm, val_loader, device, step, num_samples=4):
    ddpm.eval()
    imgs, mels = next(iter(val_loader))
    mels = mels[:num_samples].to(device)
    samples = ddpm.sample_ddim(mels, steps=50)
    samples = (samples.clamp(-1, 1) + 1) / 2
    real = (imgs[:num_samples].to(device).clamp(-1, 1) + 1) / 2
    wandb.log({
        'samples/generated': [wandb.Image(s.cpu()) for s in samples],
        'samples/real':      [wandb.Image(r.cpu()) for r in real],
    }, step=step)
    ddpm.train()


def get_lr(step, warmup_steps, base_lr):
    """Linear warmup: ramp lr from 0 to base_lr over warmup_steps steps."""
    
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    return base_lr


def train():
    args   = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Unique run name for sweep runs to avoid checkpoint conflicts
    if 'WANDB_SWEEP_ID' in os.environ:
        args.run_name = f"sweep_{args.schedule}_T{args.T}_d{args.embed_dim}"

    print(f"Device: {device}")
    print(f"Run:    {args.run_name}")
    os.makedirs(args.ckpt_dir, exist_ok=True)

    wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    train_loader, val_loader, _ = get_dataloaders(
        covers_dir  = args.covers_dir,
        spectro_dir = args.spectro_dir,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")

    encoder = AudioEncoder(embed_dim=args.embed_dim).to(device)
    unet    = UNet(
        in_channels   = 3,
        base_channels = args.base_channels,
        audio_dim     = args.embed_dim,
        time_dim      = args.time_dim,
        dropout       = args.dropout,
    ).to(device)
    ddpm = DDPM(unet=unet, encoder=encoder, T=args.T,
                schedule=args.schedule, device=device).to(device)

    total_params = sum(p.numel() for p in ddpm.parameters())
    print(f"Total parameters: {total_params:,}")
    wandb.config.update({'total_params': total_params})

    optimizer = torch.optim.AdamW(ddpm.parameters(), lr=args.lr,
                                  betas=(0.9, 0.999), weight_decay=1e-4)
    scaler = GradScaler()

    # Resume with architecture mismatch protection
    start_epoch, global_step = 0, 0
    latest_ckpt = os.path.join(args.ckpt_dir, f'{args.run_name}_latest.pt')
    if os.path.exists(latest_ckpt):
        try:
            print(f"Resuming from {latest_ckpt}")
            start_epoch, global_step, _ = load_checkpoint(
                latest_ckpt, ddpm, optimizer, scaler)
            start_epoch += 1
            print(f"Resumed from epoch {start_epoch}")
        except RuntimeError as e:
            print(f"Architecture mismatch — starting fresh. ({e})")
            start_epoch, global_step = 0, 0

    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        ddpm.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for imgs, mels in pbar:
            imgs = imgs.to(device)
            mels = mels.to(device)

            lr = get_lr(global_step, args.warmup_steps, args.lr)
            for g in optimizer.param_groups:
                g['lr'] = lr

            optimizer.zero_grad()
            with autocast():
                loss = ddpm.loss(imgs, mels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(ddpm.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss  += loss.item()
            global_step += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{lr:.2e}'})

            if global_step % args.log_every == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/lr':   lr,
                    'train/step': global_step,
                }, step=global_step)

            if global_step % args.sample_every == 0:
                log_samples(ddpm, val_loader, device, global_step)

        # Validation
        ddpm.eval()
        val_losses = []
        with torch.no_grad():
            for imgs, mels in val_loader:
                imgs = imgs.to(device)
                mels = mels.to(device)
                with autocast():
                    val_loss = ddpm.loss(imgs, mels)
                val_losses.append(val_loss.item())

        avg_val_loss   = sum(val_losses) / len(val_losses)
        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} | train_loss: {avg_train_loss:.4f} | val_loss: {avg_val_loss:.4f}")

        wandb.log({
            'epoch/train_loss': avg_train_loss,
            'epoch/val_loss':   avg_val_loss,
            'epoch':            epoch + 1,
        }, step=global_step)

        save_checkpoint(ddpm, optimizer, scaler, epoch, global_step,
                        avg_val_loss, latest_ckpt)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(args.ckpt_dir, f'{args.run_name}_best.pt')
            save_checkpoint(ddpm, optimizer, scaler, epoch, global_step,
                            avg_val_loss, best_path)
            print(f"New best val loss: {best_val_loss:.4f}")

        if (epoch + 1) % args.save_every == 0:
            periodic_path = os.path.join(args.ckpt_dir,
                                         f'{args.run_name}_epoch{epoch+1:03d}.pt')
            save_checkpoint(ddpm, optimizer, scaler, epoch, global_step,
                            avg_val_loss, periodic_path)

    print("Training complete!")
    wandb.finish()


if __name__ == '__main__':
    train()
