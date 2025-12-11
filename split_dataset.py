import os
import random
import shutil
from pathlib import Path

SRC_GRAY_DIR = Path("./dataset/gray")
SRC_COLOR_DIR = Path("./dataset/color")

TRAIN_ROOT = Path("./trainset")
VAL_ROOT = Path("./valset")
TEST_ROOT = Path("./testset")

RANDOM_SEED = 25
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

def list_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(
        [
            p for p in folder.iterdir()
            if p.is_file()
            and p.suffix.lower() in exts
            and not p.name.startswith("._")
        ]
    )

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def clear_dir(path: Path):
    if not path.exists():
        return
    for p in path.iterdir():
        if p.is_file():
            p.unlink()

def main():
    assert SRC_GRAY_DIR.is_dir(), f"gray DIR  not exist: {SRC_GRAY_DIR}"
    assert SRC_COLOR_DIR.is_dir(), f"color DIR  not exist: {SRC_COLOR_DIR}"

    gray_imgs = list_images(SRC_GRAY_DIR)
    if not gray_imgs:
        raise RuntimeError(f"no photo in {SRC_GRAY_DIR}")

    paired = []
    for g in gray_imgs:
        c = SRC_COLOR_DIR / g.name
        if c.is_file() and not c.name.startswith("._"):
            paired.append((g, c))
        else:
            print(f"[WARN]: {g.name}")

    if not paired:
        raise RuntimeError("no photos in {SRC_GRAY_DIR}")

    print(f"[INFO] {len(paired)} photos")

    random.seed(RANDOM_SEED)
    random.shuffle(paired)

    n = len(paired)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_test = n - n_train - n_val

    train_set = paired[:n_train]
    val_set = paired[n_train:n_train + n_val]
    test_set = paired[n_train + n_val:]

    print(f"[SPLIT] train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}")

    # 4. 创建 & 清空目标目录
    train_gray_dir = TRAIN_ROOT / "gray"
    train_color_dir = TRAIN_ROOT / "color"
    val_gray_dir = VAL_ROOT / "gray"
    val_color_dir = VAL_ROOT / "color"
    test_gray_dir = TEST_ROOT / "gray"

    for d in [train_gray_dir, train_color_dir, val_gray_dir, val_color_dir, test_gray_dir]:
        ensure_dir(d)
        clear_dir(d)
    for g, c in train_set:
        shutil.copy2(g, train_gray_dir / g.name)
        shutil.copy2(c, train_color_dir / c.name)
    for g, c in val_set:
        shutil.copy2(g, val_gray_dir / g.name)
        shutil.copy2(c, val_color_dir / c.name)
    for g, c in test_set:
        shutil.copy2(g, test_gray_dir / g.name)
    print("\nDONE")
    print(f"  Train gray : {len(list_images(train_gray_dir))}")
    print(f"  Train color: {len(list_images(train_color_dir))}")
    print(f"  Val gray   : {len(list_images(val_gray_dir))}")
    print(f"  Val color  : {len(list_images(val_color_dir))}")
    print(f"  Test gray  : {len(list_images(test_gray_dir))}")
    print(f"\n  Train: {train_gray_dir} & {train_color_dir}")
    print(f"  Val  : {val_gray_dir} & {val_color_dir}")
    print(f"  Test : {test_gray_dir} (only gray)")


if __name__ == "__main__":
    main()
