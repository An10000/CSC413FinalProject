GT_DIR = "./testset/color"

EVAL_DIRS = {
    "L1_UNet": "./test_color_output_L1",
    "GAN_UNet": "./test_color_output_GAN",
}

RESIZE_FOR_IQA = (256, 256)
RESIZE_FOR_HIST = (256, 256)

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

import numpy as np
from pathlib import Path
from typing import Dict, List

from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as T
import torchvision.models as models

import pyiqa
from skimage import color as skcolor


def get_image_paths(dir_path: str | Path) -> List[Path]:
    p = Path(dir_path)
    assert p.is_dir(), f"Directory not found: {p}"
    return sorted([
        x for x in p.iterdir()
        if x.suffix.lower() in VALID_EXTS and not x.name.startswith("._")
    ])


def load_rgb(path: Path, size=None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.BICUBIC)
    return img

class NIQEMetric:
    def __init__(self, device: torch.device):
        self.device = device
        self.metric = pyiqa.create_metric('niqe').to(device)
        self.metric.eval()
        self.to_tensor = T.ToTensor()

    def folder_score(self, dir_path: str, size=None) -> float:
        paths = get_image_paths(dir_path)
        scores = []
        with torch.no_grad():
            for p in tqdm(paths, desc=f"NIQE [{dir_path}]"):
                img = load_rgb(p, size=size)
                ten = self.to_tensor(img).unsqueeze(0).to(self.device)
                s = self.metric(ten).item()
                scores.append(s)
        return float(np.mean(scores)) if scores else float("nan")

class NIMAMetric:
    def __init__(self, device: torch.device):
        self.device = device
        self.metric = pyiqa.create_metric('nima').to(device)
        self.metric.eval()
        self.to_tensor = T.ToTensor()

    def folder_score(self, dir_path: str, size=None) -> float:
        paths = get_image_paths(dir_path)
        scores = []
        with torch.no_grad():
            for p in tqdm(paths, desc=f"NIMA [{dir_path}]"):
                img = load_rgb(p, size=size)
                ten = self.to_tensor(img).unsqueeze(0).to(self.device)
                s = self.metric(ten).item()
                scores.append(s)
        return float(np.mean(scores)) if scores else float("nan")

def image_colorfulness(arr: np.ndarray) -> float:
    arr = arr.astype(np.float32)
    R, G, B = arr[..., 0], arr[..., 1], arr[..., 2]
    rg = R - G
    yb = 0.5 * (R + G) - B

    mean_rg = np.mean(rg)
    mean_yb = np.mean(yb)
    std_rg = np.std(rg)
    std_yb = np.std(yb)

    std_rg_yb = np.sqrt(std_rg ** 2 + std_yb ** 2)
    mean_rg_yb = np.sqrt(mean_rg ** 2 + mean_yb ** 2)

    return float(std_rg_yb + 0.3 * mean_rg_yb)


def folder_colorfulness(dir_path: str, size=None) -> float:
    paths = get_image_paths(dir_path)
    scores = []
    for p in tqdm(paths, desc=f"Colorfulness [{dir_path}]"):
        img = load_rgb(p, size=size)
        arr = np.array(img)
        scores.append(image_colorfulness(arr))
    return float(np.mean(scores)) if scores else float("nan")



class ClassificationConsistency:
    def __init__(self, device: torch.device):
        self.device = device
        self.model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
        ).to(device)
        self.model.eval()
        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _top1(self, img: Image.Image) -> int:
        with torch.no_grad():
            x = self.transform(img).unsqueeze(0).to(self.device)
            logits = self.model(x)
            return int(torch.argmax(logits, dim=1).item())

    def consistency(self, gt_dir: str, fake_dir: str) -> float:
        gt_paths: Dict[str, Path] = {p.name: p for p in get_image_paths(gt_dir)}
        fk_paths: Dict[str, Path] = {p.name: p for p in get_image_paths(fake_dir)}

        names = sorted(set(gt_paths.keys()) & set(fk_paths.keys()))
        if not names:
            print(f"[WARN] No common filenames between {gt_dir} and {fake_dir}")
            return float("nan")

        same, total = 0, 0
        for name in tqdm(names, desc=f"ClsConsis [{fake_dir}]"):
            img_gt = load_rgb(gt_paths[name])
            img_fk = load_rgb(fk_paths[name])
            c_gt = self._top1(img_gt)
            c_fk = self._top1(img_fk)
            if c_gt == c_fk:
                same += 1
            total += 1

        return same / total if total > 0 else float("nan")

def folder_lab_histogram(dir_path: str, size=None, bins: int = 32) -> np.ndarray:
    paths = get_image_paths(dir_path)
    hist = np.zeros((bins, bins), dtype=np.float64)

    for p in tqdm(paths, desc=f"LabHist [{dir_path}]"):
        img = load_rgb(p, size=size)
        arr = np.array(img).astype(np.float32) / 255.0  # 0~1
        lab = skcolor.rgb2lab(arr)
        a = lab[..., 1].ravel()
        b = lab[..., 2].ravel()

        h, _, _ = np.histogram2d(
            a, b,
            bins=bins,
            range=[[-128, 127], [-128, 127]]
        )
        hist += h

    if hist.sum() > 0:
        hist = hist / hist.sum()
    return hist


def hist_intersection(h1: np.ndarray, h2: np.ndarray) -> float:
    return float(np.minimum(h1, h2).sum())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    niqe_metric = NIQEMetric(device)
    nima_metric = NIMAMetric(device)
    cls_consis = ClassificationConsistency(device)
    print("\n=== Ground Truth Lab Histogram ===")
    gt_hist = folder_lab_histogram(GT_DIR, size=RESIZE_FOR_HIST)
    print("\n=== Ground Truth baseline ===")
    gt_niqe = niqe_metric.folder_score(GT_DIR, size=RESIZE_FOR_IQA)
    gt_nima = nima_metric.folder_score(GT_DIR, size=RESIZE_FOR_IQA)
    gt_color = folder_colorfulness(GT_DIR, size=RESIZE_FOR_IQA)

    print("\n>>> Ground Truth")
    print(f"  NIQE         : {gt_niqe:.4f}  (low)")
    print(f"  NIMA         : {gt_nima:.4f}  (high)")
    print(f"  Colorfulness : {gt_color:.4f}  (high)")

    results = []
    print("\n==================")
    for name, dir_path in EVAL_DIRS.items():
        print(f"\n--- Evaluating [{name}] at {dir_path} ---")

        m_niqe = niqe_metric.folder_score(dir_path, size=RESIZE_FOR_IQA)
        m_nima = nima_metric.folder_score(dir_path, size=RESIZE_FOR_IQA)
        m_color = folder_colorfulness(dir_path, size=RESIZE_FOR_IQA)
        m_consis = cls_consis.consistency(GT_DIR, dir_path)

        fake_hist = folder_lab_histogram(dir_path, size=RESIZE_FOR_HIST)
        m_hist = hist_intersection(gt_hist, fake_hist)

        results.append({
            "name": name,
            "NIQE": m_niqe,
            "NIMA": m_nima,
            "Colorfulness": m_color,
            "ClsConsistency": m_consis,
            "HistIntersection": m_hist,
        })
    print("\nAdvanced Metrics:")
    header = [
        "Model",
        "NIQE(↓)",
        "NIMA(↑)",
        "Color(↑)",
        "ClsConsis(↑)",
        "HistInter(↑)",
    ]
    print("{:15s} {:10s} {:10s} {:10s} {:14s} {:12s}".format(*header))
    for r in results:
        print("{:15s} {:10.4f} {:10.4f} {:10.4f} {:14.4f} {:12.4f}".format(
            r["name"],
            r["NIQE"],
            r["NIMA"],
            r["Colorfulness"],
            r["ClsConsistency"],
            r["HistIntersection"],
        ))
    print("====================================================")

if __name__ == "__main__":
    main()
