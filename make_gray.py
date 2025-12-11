import os
from pathlib import Path
from PIL import Image

COLOR_DIR = "./dataset3"
GRAY_DIR  = "./dataset3_gray"
KEEP_EXT  = True


def gather_image_paths(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    img_paths = []
    for p in root.rglob("*"):
        if (
            p.is_file()
            and p.suffix.lower() in exts
            and not p.name.startswith("._")
        ):
            img_paths.append(p)
    return img_paths

def color_to_gray_batch(color_dir: Path, gray_dir: Path):
    img_paths = gather_image_paths(color_dir)
    if not img_paths:
        print(f"[WARN] No images found under: {color_dir}")
        return

    print(f"[INFO] Found {len(img_paths)} image(s) in {color_dir}")

    for src_path in img_paths:
        rel_path = src_path.relative_to(color_dir)

        if KEEP_EXT:
            out_rel = rel_path
        else:
            out_rel = rel_path.with_suffix(".jpg")

        dst_path = gray_dir / out_rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            img = Image.open(src_path).convert("L")
            img.save(dst_path)
            print(f"[OK] {src_path} -> {dst_path}")
        except Exception as e:
            print(f"[ERR] Failed on {src_path}: {e}")


def main():
    color_dir = Path(COLOR_DIR)
    gray_dir = Path(GRAY_DIR)

    assert color_dir.is_dir(), f"Color directory not found: {color_dir}"
    gray_dir.mkdir(parents=True, exist_ok=True)

    color_to_gray_batch(color_dir, gray_dir)


if __name__ == "__main__":
    main()
