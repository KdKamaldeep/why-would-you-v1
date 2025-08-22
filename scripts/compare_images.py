#!/usr/bin/env python3
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import math

try:
    from skimage.metrics import structural_similarity as ssim
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False


def load_image(path: str, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.LANCZOS)
    return img


def mse(img_a: np.ndarray, img_b: np.ndarray) -> float:
    err = np.mean((img_a.astype(np.float32) - img_b.astype(np.float32)) ** 2)
    return float(err)


def psnr(img_a: np.ndarray, img_b: np.ndarray) -> float:
    m = mse(img_a, img_b)
    if m == 0:
        return float("inf")
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(m))


def compute_ssim(img_a: np.ndarray, img_b: np.ndarray) -> float | None:
    if not _HAS_SKIMAGE:
        return None
    # Compute SSIM on luminance
    a_gray = np.dot(img_a[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
    b_gray = np.dot(img_b[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
    s, _ = ssim(a_gray, b_gray, full=True, data_range=255)
    return float(s)


def ahash(img: Image.Image) -> int:
    gray = img.convert("L").resize((8, 8), Image.LANCZOS)
    arr = np.asarray(gray, dtype=np.float32)
    avg = arr.mean()
    bits = (arr > avg).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def hamming_distance(x: int, y: int) -> int:
    return int(bin(x ^ y).count("1"))


def compare_images(base_path: str, compare_path: str, out_path: str | None = None):
    base = load_image(base_path)
    cmp_ = load_image(compare_path, size=base.size)

    a = np.asarray(base)
    b = np.asarray(cmp_)

    metrics = {}
    metrics["mse"] = mse(a, b)
    metrics["psnr_db"] = psnr(a, b)
    s = compute_ssim(a, b)
    if s is not None:
        metrics["ssim"] = s
    metrics["ahash_hamming"] = hamming_distance(ahash(base), ahash(cmp_))

    if out_path:
        # Create a side-by-side composite with labels and metrics
        w, h = base.size
        pad = 20
        header_h = 60
        canvas = Image.new("RGB", (w * 2 + pad, h + header_h), (30, 30, 30))
        draw = ImageDraw.Draw(canvas)
        canvas.paste(base, (0, header_h))
        canvas.paste(cmp_, (w + pad, header_h))
        title = "Left: Original | Right: Generated"
        metrics_text = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
        draw.text((10, 10), title, fill=(255, 255, 255))
        draw.text((10, 30), metrics_text, fill=(200, 200, 200))
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--frame", required=True)
    parser.add_argument("--out", required=False, default=None)
    args = parser.parse_args()

    m = compare_images(args.base, args.frame, args.out)
    print(m)
