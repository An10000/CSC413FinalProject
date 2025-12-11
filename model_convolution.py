import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import to_pil_image

MODEL_PATH = "./models_select/model_GAN.pth"
INPUT_PATH = "./testset/gray"
OUTPUT_DIR = "./test_color_output"
IMAGE_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def load_model(model_path: str, device: torch.device) -> nn.Module:
    model = UNetGenerator(in_channels=1, out_channels=3)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"[INFO] Loaded pix2pix generator from: {model_path}")
    return model

def colorize_image(model: nn.Module, img_path: Path, device: torch.device) -> Image.Image:
    img = Image.open(img_path).convert("L")

    w, h = img.size
    min_side = min(w, h)
    if min_side < IMAGE_SIZE:
        scale = IMAGE_SIZE / float(min_side)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img = img.resize((new_w, new_h), resample=Image.BILINEAR)
        w, h = img.size
        print(f"[INFO] Upscaled {img_path.name} to {w}x{h} for model input")

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])

    gray = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(gray)
        pred = (pred * 0.5) + 0.5
        pred = torch.clamp(pred, 0.0, 1.0)
        pred = pred.squeeze(0).cpu()

    color_img = to_pil_image(pred)
    return color_img

def gather_image_paths(input_path: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}

    if input_path.is_file():
        if input_path.suffix.lower() in exts:
            return [input_path]
        else:
            raise ValueError(f"Input file is not a supported image: {input_path}")

    if input_path.is_dir():
        img_paths = [
            p for p in input_path.iterdir()
            if p.is_file()
            and p.suffix.lower() in exts
            and not p.name.startswith("._")
        ]
        return img_paths

    raise FileNotFoundError(f"Input path not found: {input_path}")

def main():
    print(f"Using device: {DEVICE}")

    model_path = Path(MODEL_PATH)
    assert model_path.is_file(), f"Model file not found: {model_path}"

    input_path = Path(INPUT_PATH)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_paths = gather_image_paths(input_path)
    if not img_paths:
        print(f"[WARN] No valid images found from: {input_path}")
        return

    print(f"[INFO] Found {len(img_paths)} image(s) to colorize from: {input_path}")

    model = load_model(str(model_path), DEVICE)

    for img_path in img_paths:
        try:
            color_img = colorize_image(model, img_path, DEVICE)
            out_name = img_path.name if input_path.is_dir() else f"color_{img_path.name}"
            out_path = output_dir / out_name
            color_img.save(out_path)
            print(f"[OK] {img_path} -> {out_path} ({color_img.size[0]}x{color_img.size[1]})")
        except Exception as e:
            print(f"[ERR] Failed on {img_path}: {e}")


if __name__ == "__main__":
    main()
