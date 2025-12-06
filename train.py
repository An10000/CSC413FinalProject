# train.py
import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from dataset import create_dataloaders
from models import UNetGenerator, PatchDiscriminator


def save_sample_images(L, real_ab, fake_ab, out_dir, step, max_batch=4):
    """
    Save a few sample LAB->RGB results for qualitative inspection.
    这里先留空 / TODO：等你需要时可以加上 Lab->RGB 可视化。
    为了简洁，先只做训练 loop，图像可视化可以后面补。
    """
    pass  # TODO: implement visualization if needed


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    loaders = create_dataloaders(
        color_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=0.1,
        test_split=0.1,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # Models
    netG = UNetGenerator(in_channels=1, out_channels=2, base_ch=args.base_ch).to(device)
    netD = PatchDiscriminator(in_channels=3, base_ch=args.base_ch).to(device)

    # Losses
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()
    lambda_L1 = args.lambda_L1

    # Optimizers
    optimizer_G = Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    step = 0
    for epoch in range(1, args.epochs + 1):
        netG.train()
        netD.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for i, (L, ab) in enumerate(pbar):
            L = L.to(device)   # (N,1,H,W)
            ab = ab.to(device) # (N,2,H,W)
            batch_size = L.size(0)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Real image pair (L, ab_real)
            real_pair = torch.cat([L, ab], dim=1)  # (N,3,H,W)
            pred_real = netD(real_pair)
            label_real = torch.ones_like(pred_real, device=device)
            loss_D_real = criterion_GAN(pred_real, label_real)

            # Fake image pair (L, ab_fake)
            fake_ab = netG(L).detach()
            fake_pair = torch.cat([L, fake_ab], dim=1)
            pred_fake = netD(fake_pair)
            label_fake = torch.zeros_like(pred_fake, device=device)
            loss_D_fake = criterion_GAN(pred_fake, label_fake)

            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            fake_ab = netG(L)
            fake_pair = torch.cat([L, fake_ab], dim=1)
            pred_fake_for_G = netD(fake_pair)

            # Adversarial loss: want D to classify fake as real
            loss_G_GAN = criterion_GAN(pred_fake_for_G, label_real)
            # L1 reconstruction loss
            loss_G_L1 = criterion_L1(fake_ab, ab) * lambda_L1

            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            optimizer_G.step()

            step += 1
            pbar.set_postfix({
                "loss_D": loss_D.item(),
                "loss_G": loss_G.item(),
                "loss_L1": loss_G_L1.item(),
            })

        # ---- simple validation placeholder ----
        netG.eval()
        with torch.no_grad():
            val_L1 = 0.0
            n_val = 0
            for L_val, ab_val in val_loader:
                L_val = L_val.to(device)
                ab_val = ab_val.to(device)
                fake_ab_val = netG(L_val)
                val_L1 += criterion_L1(fake_ab_val, ab_val).item() * L_val.size(0)
                n_val += L_val.size(0)
            val_L1 /= n_val
            print(f"[Epoch {epoch}] Val L1: {val_L1:.4f}")

        # Save checkpoint each epoch
        ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "netG_state": netG.state_dict(),
            "netD_state": netD.state_dict(),
            "optG_state": optimizer_G.state_dict(),
            "optD_state": optimizer_D.state_dict(),
        }, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory with color JPG/PNG images from Kaggle dataset")
    parser.add_argument("--checkpoint_dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--image_size", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--base_ch", type=int, default=64)
    parser.add_argument("--lambda_L1", type=float, default=100.0)
    args = parser.parse_args()

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    train(args)
