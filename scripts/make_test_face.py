#!/usr/bin/env python3
from pathlib import Path
from PIL import Image, ImageDraw

def make_face(path: str, size: int = 512):
    img = Image.new("RGB", (size, size), (240, 230, 220))
    d = ImageDraw.Draw(img)

    # Head
    margin = int(size * 0.08)
    d.ellipse([margin, margin, size - margin, size - margin], fill=(255, 224, 189), outline=(200, 170, 140), width=4)

    # Eyes
    eye_w, eye_h = int(size * 0.12), int(size * 0.08)
    eye_y = int(size * 0.4)
    eye_x_offset = int(size * 0.18)
    left_eye_box = [size//2 - eye_x_offset - eye_w, eye_y - eye_h//2, size//2 - eye_x_offset + eye_w, eye_y + eye_h//2]
    right_eye_box = [size//2 + eye_x_offset - eye_w, eye_y - eye_h//2, size//2 + eye_x_offset + eye_w, eye_y + eye_h//2]
    d.ellipse(left_eye_box, fill=(255,255,255), outline=(0,0,0), width=3)
    d.ellipse(right_eye_box, fill=(255,255,255), outline=(0,0,0), width=3)

    # Pupils
    pup_w = int(size * 0.04)
    left_pupil = [size//2 - eye_x_offset - pup_w//2, eye_y - pup_w//2, size//2 - eye_x_offset + pup_w//2, eye_y + pup_w//2]
    right_pupil = [size//2 + eye_x_offset - pup_w//2, eye_y - pup_w//2, size//2 + eye_x_offset + pup_w//2, eye_y + pup_w//2]
    d.ellipse(left_pupil, fill=(40,40,40))
    d.ellipse(right_pupil, fill=(40,40,40))

    # Brows
    brow_y = int(size * 0.33)
    brow_len = int(size * 0.18)
    d.line([(size//2 - eye_x_offset - brow_len//2, brow_y), (size//2 - eye_x_offset + brow_len//2, brow_y - 6)], fill=(60,40,30), width=6)
    d.line([(size//2 + eye_x_offset - brow_len//2, brow_y - 6), (size//2 + eye_x_offset + brow_len//2, brow_y)], fill=(60,40,30), width=6)

    # Nose
    nose_top = (size//2, int(size * 0.48))
    nose_left = (size//2 - int(size*0.03), int(size * 0.58))
    nose_right = (size//2 + int(size*0.03), int(size * 0.58))
    d.line([nose_top, nose_left, nose_right], fill=(160,120,90), width=4)

    # Mouth
    mouth_y = int(size * 0.7)
    mouth_w = int(size * 0.3)
    d.arc([size//2 - mouth_w, mouth_y - 20, size//2 + mouth_w, mouth_y + 40], start=200, end=340, fill=(180,60,80), width=6)

    # Minimal hair
    hair_top = int(size * 0.1)
    d.rectangle([margin, hair_top, size - margin, hair_top + int(size*0.1)], fill=(50, 30, 20))

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)

if __name__ == "__main__":
    make_face("faces/test_face.png", size=512)
    print("Saved faces/test_face.png")
