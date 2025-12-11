import shutil
from pathlib import Path

GRAY_TEST_DIR = Path("./testset/gray")
COLOR_DATASET_DIR = Path("./dataset3/color")
OUTPUT_COLOR_TEST = Path("./testset/color")

def match_color_by_name():
    assert GRAY_TEST_DIR.exists(), f"Not found: {GRAY_TEST_DIR}"
    assert COLOR_DATASET_DIR.exists(), f"Not found: {COLOR_DATASET_DIR}"

    OUTPUT_COLOR_TEST.mkdir(parents=True, exist_ok=True)

    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}

    count = 0
    missing = []

    for gray_path in GRAY_TEST_DIR.iterdir():
        if gray_path.suffix.lower() not in valid_ext or gray_path.name.startswith("._"):
            continue

        color_path = COLOR_DATASET_DIR / gray_path.name
        if color_path.exists():
            shutil.copy(color_path, OUTPUT_COLOR_TEST / gray_path.name)
            count += 1
        else:
            missing.append(gray_path.name)

    print(f"[DONE] Copied {count} matching color images to: {OUTPUT_COLOR_TEST}")

    if missing:
        print("\n[WARN] These files were NOT found in dataset/color:")
        for m in missing:
            print("  -", m)


if __name__ == "__main__":
    match_color_by_name()
