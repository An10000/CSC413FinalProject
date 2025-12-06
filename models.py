# models.py
import torch
import torch.nn as nn


# ---------- U-Net building blocks ----------

class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm -> LeakyReLU (downsampling block)."""
    def __init__(self, in_ch, out_ch, normalize=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not normalize)
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """ConvTranspose2d -> BatchNorm -> ReLU (+ dropout optional)."""
    def __init__(self, in_ch, out_ch, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetGenerator(nn.Module):
    """
    U-Net generator for L->ab colorization.
    Input:  (N, 1, H, W)
    Output: (N, 2, H, W)
    """
    def __init__(self, in_channels=1, out_channels=2, base_ch=64):
        super().__init__()

        # Encoder
        self.down1 = ConvBlock(in_channels, base_ch, normalize=False)   # 64
        self.down2 = ConvBlock(base_ch, base_ch * 2)                    # 128
        self.down3 = ConvBlock(base_ch * 2, base_ch * 4)                # 256
        self.down4 = ConvBlock(base_ch * 4, base_ch * 8)                # 512
        self.down5 = ConvBlock(base_ch * 8, base_ch * 8)                # 512

        # Bottleneck (no stride)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_ch * 8, base_ch * 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.up5 = UpBlock(base_ch * 8, base_ch * 8, dropout=True)
        self.up4 = UpBlock(base_ch * 8 * 2, base_ch * 8, dropout=True)
        self.up3 = UpBlock(base_ch * 8 * 2, base_ch * 4)
        self.up2 = UpBlock(base_ch * 4 * 2, base_ch * 2)
        self.up1 = UpBlock(base_ch * 2 * 2, base_ch)

        # Final layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # output in [-1, 1]
        )

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        b = self.bottleneck(d5)

        # Decoder with skip connections (concat along channel dim)
        u5 = self.up5(b)
        u5 = torch.cat([u5, d5], dim=1)

        u4 = self.up4(u5)
        u4 = torch.cat([u4, d4], dim=1)

        u3 = self.up3(u4)
        u3 = torch.cat([u3, d3], dim=1)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, d2], dim=1)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)

        out = self.final(u1)
        return out


# ---------- PatchGAN Discriminator ----------

class PatchDiscriminator(nn.Module):
    """
    70x70 PatchGAN discriminator from Pix2Pix.
    Input: concat(L, ab_pred or ab_real) -> (N, 3, H, W)
    Output: (N, 1, H', W') patch-level real/fake logits.
    """
    def __init__(self, in_channels=3, base_ch=64):
        super().__init__()
        # No norm in first layer
        sequence = [
            nn.Conv2d(in_channels, base_ch, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        curr_ch = base_ch
        for n_mult in [2, 4, 8]:
            sequence += [
                nn.Conv2d(curr_ch, base_ch * n_mult, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(base_ch * n_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            curr_ch = base_ch * n_mult

        # Last conv: stride 1 for 70x70 receptive field (approx)
        sequence += [
            nn.Conv2d(curr_ch, 1, kernel_size=4, stride=1, padding=1)
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)
