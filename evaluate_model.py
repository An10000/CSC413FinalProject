REAL_DIR = "./testset/color"
FAKE_DIR = "test_color_output_L1"

RESIZE_TO = (256, 256)

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

LPIPS_BACKBONE = "alex"

import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torchvision.transforms as T
import lpips
from cleanfid import fid
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def load_img(path: Path, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.BICUBIC)
    return img


def calc_ssim_psnr(dir_real, dir_fake, size=None):
    real_paths = sorted([p for p in Path(dir_real).glob("*") if p.suffix.lower() in VALID_EXTS])
    fake_paths = sorted([p for p in Path(dir_fake).glob("*") if p.suffix.lower() in VALID_EXTS])

    assert len(real_paths) == len(fake_paths) and len(real_paths) > 0, "DIR is empty"

    total_ssim = 0.0
    total_psnr = 0.0

    for r,f in tqdm(list(zip(real_paths,fake_paths)), desc="SSIM/PSNR"):
        img_r = load_img(r, size)
        img_f = load_img(f, size)

        arr_r = np.array(img_r)
        arr_f = np.array(img_f)

        total_ssim += ssim(arr_r, arr_f, channel_axis=2, data_range=255)
        total_psnr += psnr(arr_r, arr_f, data_range=255)

    return total_ssim / len(real_paths), total_psnr / len(real_paths)


def calc_lpips(dir_real, dir_fake, size=None):
    loss_fn = lpips.LPIPS(net=LPIPS_BACKBONE).cuda()

    t = T.ToTensor()

    real_paths = sorted([p for p in Path(dir_real).glob("*") if p.suffix.lower() in VALID_EXTS])
    fake_paths = sorted([p for p in Path(dir_fake).glob("*") if p.suffix.lower() in VALID_EXTS])

    total = 0.0

    for r,f in tqdm(list(zip(real_paths,fake_paths)), desc="LPIPS"):
        img_r = load_img(r, size)
        img_f = load_img(f, size)

        ten_r = t(img_r).unsqueeze(0).cuda()
        ten_f = t(img_f).unsqueeze(0).cuda()

        total += loss_fn(ten_r, ten_f).item()

    return total / len(real_paths)


def calc_fid(dir_real, dir_fake, num_workers=0, batch_size=32):
    return fid.compute_fid(
        dir_real,
        dir_fake,
        num_workers=num_workers,
        batch_size=batch_size,
    )

def evaluate_all(real_dir, fake_dir, size=None):
    print("\n=== Evaluating ===")
    print(f"Real: {real_dir}")
    print(f"Fake: {fake_dir}")
    print("==================")

    fid_score = calc_fid(real_dir, fake_dir)
    lpips_score = calc_lpips(real_dir, fake_dir, size)
    ssim_score, psnr_score = calc_ssim_psnr(real_dir, fake_dir, size)

    print("\n===== Results =====")
    print(f"FID  : {fid_score:.4f}")
    print(f"LPIPS: {lpips_score:.4f} (low)")
    print(f"SSIM : {ssim_score:.4f} (high)")
    print(f"PSNR : {psnr_score:.4f} (high)")
    print("===================\n")

    return {
        "FID": fid_score,
        "LPIPS": lpips_score,
        "SSIM": ssim_score,
        "PSNR": psnr_score
    }

if __name__ == "__main__":
    evaluate_all(REAL_DIR, FAKE_DIR, size=RESIZE_TO)
