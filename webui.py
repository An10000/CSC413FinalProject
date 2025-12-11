import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image
import gradio as gr


UI_TITLE           = "UNet Colorization WebUI"

TAB_SINGLE         = "Colorize"

TXT_MODEL_DD       = "Select model (from ./models_select)"
TXT_UPLOAD_SINGLE  = "Drag or click to upload a grayscale image"
TXT_OUTPUT_PREVIEW = "Colorized result"

BTN_REFRESH_MODEL  = "🔄 Refresh model list"
BTN_COLORIZE_ONE   = "Start colorization"

STYLE_PRIMARY      = "#1f6feb"
STYLE_BG           = "#0d1117"
STYLE_FONT         = "#c9d1d9"
STYLE_ACCENT       = "#238636"
STYLE_WARN         = "#ff9800"

MODEL_PATH  = "./models_select/best_pix2pix_generator_V2.0.pth"
MODELS_DIR  = "./models_select"
OUTPUT_DIR  = "./test_color_output"
IMAGE_SIZE  = 256
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, features=(64, 128, 256, 512)):
        super().__init__()
        self.downs, self.ups = nn.ModuleList(), nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, 2))
            self.ups.append(DoubleConv(f * 2, f))

        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            sc = skips[i // 2]
            if x.shape[2:] != sc.shape[2:]:
                x = F.interpolate(x, sc.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat((sc, x), 1)
            x = self.ups[i + 1](x)

        return torch.tanh(self.final_conv(x))

_global_model, _global_path = None, None


def load_model(model_path, device):
    global _global_model, _global_path
    # Reuse cached model if the path is the same
    if _global_model is not None and _global_path == model_path:
        return _global_model

    m = UNetGenerator(1, 3)
    m.load_state_dict(torch.load(model_path, map_location=device))
    m.to(device).eval()
    _global_model, _global_path = m, model_path
    print(f"[INFO] Loaded model: {model_path}")
    return m

def preprocess_gray(img: Image.Image):
    """Convert PIL image to normalized tensor, resized so the shorter side >= IMAGE_SIZE."""
    img = img.convert("L")
    w, h = img.size
    min_side = min(w, h)
    if min_side < IMAGE_SIZE:
        scale = IMAGE_SIZE / min_side
        img = img.resize((int(w * scale), int(h * scale)))
    t = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5], [0.5])
    ])
    return t(img).unsqueeze(0), img.size


def to_pil(pred: torch.Tensor):
    """Convert model output tensor in [-1, 1] back to a PIL image."""
    pred = (pred * 0.5 + 0.5).clamp(0, 1).squeeze().cpu()
    return to_pil_image(pred)


def get_model_choices():
    d = Path(MODELS_DIR)
    c = sorted([str(p) for p in d.glob("*.pth")]) if d.is_dir() else []
    return (c or [MODEL_PATH], (c[0] if c else MODEL_PATH))


def refresh_models():
    choices, default = get_model_choices()
    return gr.update(choices=choices, value=default)

def ui_color_one(img, model):
    if img is None:
        return None
    m = load_model(model, DEVICE)
    x, _ = preprocess_gray(img)
    x = x.to(DEVICE)
    with torch.no_grad():
        pred = m(x)
    return to_pil(pred)

def build_ui():
    choices, default = get_model_choices()
    with gr.Blocks(title=UI_TITLE) as demo:
        demo.css = f"body{{background:{STYLE_BG}; color:{STYLE_FONT}}}"
        gr.Markdown(f"<h1 style='color:{STYLE_PRIMARY}'>{UI_TITLE}</h1>")

        refresh = gr.Button(BTN_REFRESH_MODEL)
        with gr.Tab(TAB_SINGLE):
            with gr.Row():
                inp = gr.Image(label=TXT_UPLOAD_SINGLE, type="pil")
                out = gr.Image(
                    label=TXT_OUTPUT_PREVIEW,
                    type="pil",
                    format="jpeg"  # Force JPEG output
                )
            model1 = gr.Dropdown(label=TXT_MODEL_DD, choices=choices, value=default)
            gr.Button(BTN_COLORIZE_ONE).click(ui_color_one, [inp, model1], out)

        refresh.click(refresh_models, outputs=model1)
    return demo

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    build_ui().launch(server_name="127.0.0.1", server_port=7860, share=False)
