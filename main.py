import os
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

from tqdm import tqdm

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ColorizationDataset(Dataset):
    def __init__(self, gray_dir: str, color_dir: str, image_size: int = 256):
        self.gray_dir = Path(gray_dir)
        self.color_dir = Path(color_dir)
        self.image_size = image_size

        assert self.gray_dir.exists(), f"{self.gray_dir} does not exist"
        assert self.color_dir.exists(), f"{self.color_dir} does not exist"

        def valid_image(p: Path):
            return (
                p.is_file()
                and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                and not p.name.startswith("._")
            )

        self.gray_paths = sorted([p for p in self.gray_dir.iterdir() if valid_image(p)])
        self.color_paths = sorted([p for p in self.color_dir.iterdir() if valid_image(p)])

        assert len(self.gray_paths) > 0, f"No valid images in {self.gray_dir}"
        assert len(self.color_paths) > 0, f"No valid images in {self.color_dir}"
        assert len(self.gray_paths) == len(self.color_paths), (
            f"Gray ({len(self.gray_paths)}) and color ({len(self.color_paths)}) "
            f"counts do not match"
        )

        self.gray_transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

        self.color_transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.gray_paths)

    def __getitem__(self, idx):
        gray_path = self.gray_paths[idx]
        color_path = self.color_paths[idx]

        gray = Image.open(gray_path).convert("L")
        color = Image.open(color_path).convert("RGB")

        gray = self.gray_transform(gray)
        color = self.color_transform(color)

        return gray, color

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNetGenerator(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 3,
                 features=(64, 128, 256, 512)):
        super().__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        ch_in = in_channels
        for feature in features:
            self.downs.append(DoubleConv(ch_in, feature))
            ch_in = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape[2:] != skip_connection.shape[2:]:
                x = F.interpolate(
                    x,
                    size=skip_connection.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )

            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](x)

        x = self.final_conv(x)
        x = torch.tanh(x)
        return x

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 1 + 3, features=(64, 128, 256, 512)):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        in_ch = features[0]
        for feature in features[1:]:
            layers.append(
                nn.Conv2d(
                    in_ch,
                    feature,
                    kernel_size=4,
                    stride=2 if feature != features[-1] else 1,
                    padding=1,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(feature))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_ch = feature

        layers.append(
            nn.Conv2d(
                in_ch,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
            )
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x_gray, x_color):
        # x_gray: [B, 1, H, W], x_color: [B, 3, H, W]
        x = torch.cat([x_gray, x_color], dim=1)  # [B, 4, H, W]
        return self.model(x)  # [B, 1, H', W']

def train_one_epoch(
    G: nn.Module,
    D: nn.Module,
    loader: DataLoader,
    opt_G: optim.Optimizer,
    opt_D: optim.Optimizer,
    criterion_GAN: nn.Module,
    criterion_L1: nn.Module,
    device: torch.device,
    epoch: int,
    num_epochs: int,
    lambda_L1: float,
    lambda_adv: float,
):
    G.train()
    D.train()

    running_loss_G = 0.0
    running_loss_D = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{num_epochs} [Train]")
    for gray, color in pbar:
        gray = gray.to(device)
        color = color.to(device)
        batch_size = gray.size(0)
        total_samples += batch_size
        opt_D.zero_grad()

        pred_real = D(gray, color)
        label_real = torch.ones_like(pred_real, device=device)
        loss_D_real = criterion_GAN(pred_real, label_real)

        fake_color = G(gray).detach()
        pred_fake = D(gray, fake_color)
        label_fake = torch.zeros_like(pred_fake, device=device)
        loss_D_fake = criterion_GAN(pred_fake, label_fake)

        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        opt_D.step()

        opt_G.zero_grad()

        fake_color = G(gray)
        pred_fake_for_G = D(gray, fake_color)

        loss_G_GAN = criterion_GAN(pred_fake_for_G, label_real)

        loss_G_L1 = criterion_L1(fake_color, color) * lambda_L1

        loss_G = lambda_adv * loss_G_GAN + loss_G_L1
        loss_G.backward()
        opt_G.step()

        running_loss_G += loss_G.item() * batch_size
        running_loss_D += loss_D.item() * batch_size

        avg_G = running_loss_G / total_samples
        avg_D = running_loss_D / total_samples
        pbar.set_postfix(
            loss_G=f"{avg_G:.6f}",
            loss_D=f"{avg_D:.6f}",
        )

    avg_loss_G = running_loss_G / total_samples
    avg_loss_D = running_loss_D / total_samples
    return avg_loss_G, avg_loss_D


def evaluate(
    G: nn.Module,
    loader: DataLoader,
    criterion_L1: nn.Module,
    device: torch.device,
    epoch: int,
    num_epochs: int,
) -> float:
    G.eval()
    running_loss_L1 = 0.0
    total_samples = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{num_epochs} [Val]")
        for gray, color in pbar:
            gray = gray.to(device)
            color = color.to(device)

            fake_color = G(gray)
            loss_L1 = criterion_L1(fake_color, color)

            batch_size = gray.size(0)
            running_loss_L1 += loss_L1.item() * batch_size
            total_samples += batch_size

            avg_L1 = running_loss_L1 / total_samples
            pbar.set_postfix(L1=f"{avg_L1:.6f}")

    return running_loss_L1 / total_samples


def print_hparams(hparams: dict, epoch: int):
    msg_parts = [f"epoch={epoch}"]
    for k, v in hparams.items():
        msg_parts.append(f"{k}={v}")
    print("[HParams] " + ", ".join(msg_parts), flush=True)

def main():
    image_size = 256
    batch_size = 16
    num_epochs = 30
    lr_G_init = 1e-4
    lr_D_init = 1e-4
    beta1 = 0.5
    beta2 = 0.999
    lambda_L1 = 100.0
    lambda_adv = 1.0
    lr_step_size = 10
    lr_gamma = 0.5
    seed = 42

    set_seed(seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_gray_dir = "./trainset/gray"
    train_color_dir = "./trainset/color"
    val_gray_dir = "./valset/gray"
    val_color_dir = "./valset/color"

    assert os.path.isdir(train_gray_dir), f"Train gray directory not found: {train_gray_dir}"
    assert os.path.isdir(train_color_dir), f"Train color directory not found: {train_color_dir}"
    assert os.path.isdir(val_gray_dir), f"Val gray directory not found: {val_gray_dir}"
    assert os.path.isdir(val_color_dir), f"Val color directory not found: {val_color_dir}"

    train_dataset = ColorizationDataset(
        gray_dir=train_gray_dir,
        color_dir=train_color_dir,
        image_size=image_size,
    )
    val_dataset = ColorizationDataset(
        gray_dir=val_gray_dir,
        color_dir=val_color_dir,
        image_size=image_size,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    G = UNetGenerator(in_channels=1, out_channels=3).to(device)
    D = PatchDiscriminator(in_channels=1 + 3).to(device)

    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()

    opt_G = optim.Adam(G.parameters(), lr=lr_G_init, betas=(beta1, beta2))
    opt_D = optim.Adam(D.parameters(), lr=lr_D_init, betas=(beta1, beta2))

    scheduler_G = optim.lr_scheduler.StepLR(
        opt_G, step_size=lr_step_size, gamma=lr_gamma
    )
    scheduler_D = optim.lr_scheduler.StepLR(
        opt_D, step_size=lr_step_size, gamma=lr_gamma
    )

    best_val_L1 = float("inf")

    for epoch in range(1, num_epochs + 1):
        current_lr_G = opt_G.param_groups[0]["lr"]
        current_lr_D = opt_D.param_groups[0]["lr"]

        hparams_epoch = {
            "image_size": image_size,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "lr_G": current_lr_G,
            "lr_D": current_lr_D,
            "beta1": beta1,
            "beta2": beta2,
            "lambda_L1": lambda_L1,
            "lambda_adv": lambda_adv,
            "lr_step_size": lr_step_size,
            "lr_gamma": lr_gamma,
            "seed": seed,
        }
        print_hparams(hparams_epoch, epoch=epoch)

        train_loss_G, train_loss_D = train_one_epoch(
            G=G,
            D=D,
            loader=train_loader,
            opt_G=opt_G,
            opt_D=opt_D,
            criterion_GAN=criterion_GAN,
            criterion_L1=criterion_L1,
            device=device,
            epoch=epoch,
            num_epochs=num_epochs,
            lambda_L1=lambda_L1,
            lambda_adv=lambda_adv,
        )

        val_L1 = evaluate(
            G=G,
            loader=val_loader,
            criterion_L1=criterion_L1,
            device=device,
            epoch=epoch,
            num_epochs=num_epochs,
        )

        print(
            f"Epoch {epoch}/{num_epochs} - "
            f"Train G Loss: {train_loss_G:.6f}  "
            f"Train D Loss: {train_loss_D:.6f}  "
            f"Val L1: {val_L1:.6f}"
        )

        if val_L1 < best_val_L1:
            best_val_L1 = val_L1
            torch.save(G.state_dict(), "model_GAN.pth")
            print(f"  -> Saved new best generator with Val L1 = {best_val_L1:.6f}")

        scheduler_G.step()
        scheduler_D.step()


if __name__ == "__main__":
    main()
