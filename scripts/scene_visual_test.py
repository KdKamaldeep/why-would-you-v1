#!/usr/bin/env python3
"""
Scene Visual Prompt Test

Reads a storyboard JSON, extracts each scene's visual_prompt (and optional negative_prompt),
and generates one image per scene into the scene_test/ folder using ImageGenerator.

Examples:
  python scripts/scene_visual_test.py --storyboard storyboards/example_with_voices_plain.json
  python scripts/scene_visual_test.py --storyboard output/storyboard.json --video-format normal --limit 3
  python scripts/scene_visual_test.py --style realistic --storyboard storyboards/example_with_voices_plain.json
"""

import os
import sys
import json
import logging
from pathlib import Path
import argparse

# Ensure we can import from src/
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from core.image_generator import ImageGenerator  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("scene_visual_test")


def resolve_storyboard_path(user_path: str | None) -> Path:
    """Resolve storyboard path with sensible fallbacks."""
    if user_path:
        return Path(user_path)
    # Prefer example storyboard if present
    example = PROJECT_ROOT / "storyboards" / "example_with_voices_plain.json"
    if example.exists():
        return example
    # Fallback to last generated storyboard
    output_storyboard = PROJECT_ROOT / "output" / "storyboard.json"
    if output_storyboard.exists():
        return output_storyboard
    # As a last resort, try example_with_voices_ssml.json
    ssml_example = PROJECT_ROOT / "storyboards" / "example_with_voices_ssml.json"
    if ssml_example.exists():
        return ssml_example
    raise FileNotFoundError("No storyboard file found. Provide --storyboard path or create output/storyboard.json")


def get_model_path_for_style(style: str) -> str | None:
    """Get appropriate model path based on selected style."""
    if style == "cartoon":
        # Cartoon models
        cartoon_models = [
            "models/toonyou_beta6.safetensors",
            "models/anything-v4.5.safetensors", 
            "models/counterfeit-v3.0.safetensors"
        ]
        for model in cartoon_models:
            if Path(model).exists():
                return model
    else:  # realistic
        # Realistic models
        realistic_models = [
            "models/realistic-vision-v5.1.safetensors",
            "models/realistic-vision-v4.safetensors",
            "models/dreamshaper-v8.safetensors",
            "models/deliberate-v3.safetensors"
        ]
        for model in realistic_models:
            if Path(model).exists():
                return model
    
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate one image per scene using only the scene's visual_prompt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/scene_visual_test.py --storyboard storyboards/example_with_voices_plain.json
  python scripts/scene_visual_test.py --storyboard output/storyboard.json --video-format normal --limit 3
  python scripts/scene_visual_test.py --style realistic --storyboard storyboards/example_with_voices_plain.json
        """,
    )

    parser.add_argument(
        "--storyboard",
        type=str,
        help="Path to storyboard JSON file containing scenes[] with visual_prompt",
        default=None,
    )
    parser.add_argument(
        "--style",
        choices=["cartoon", "realistic"],
        default="cartoon",
        help="Visual style: cartoon (default) or realistic",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="scene_test",
        help="Directory to save generated scene images (default: scene_test)",
    )
    parser.add_argument(
        "--video-format",
        "-f",
        choices=["shorts", "normal"],
        default="shorts",
        help="Image dimensions preset: shorts=768x1024 (9:16), normal=1920x1080 (16:9)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N scenes",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="Start from this 1-based scene index (default: 1)",
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Disable image validation and retries (faster)",
    )

    args = parser.parse_args()

    storyboard_path = resolve_storyboard_path(args.storyboard)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.video_format.lower() == "shorts":
        width, height = 768, 1024
    else:
        width, height = 1920, 1080

    logger.info("Using storyboard: %s", storyboard_path)
    logger.info("Output directory: %s", output_dir)
    logger.info("Style: %s", args.style)
    logger.info("Dimensions: %dx%d", width, height)

    with open(storyboard_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenes = data.get("scenes", [])
    if not scenes:
        logger.error("No scenes[] found in storyboard: %s", storyboard_path)
        sys.exit(1)

    # Apply start and limit
    start_idx_1_based = max(1, int(args.start_index))
    start_zero = start_idx_1_based - 1
    selected = scenes[start_zero: (start_zero + args.limit) if args.limit else None]

    # Get appropriate model for selected style
    model_path = get_model_path_for_style(args.style)
    if model_path:
        logger.info("Using %s model: %s", args.style, model_path)
    else:
        logger.info("No %s model found, will use default model", args.style)

    # Initialize the image generator once
    logger.info("Initializing ImageGenerator...")
    image_gen = ImageGenerator(model_path=model_path, width=width, height=height)
    if image_gen.is_sd_available():
        logger.info("Stable Diffusion available on device: %s", image_gen.device)
        logger.info("Model path: %s", image_gen.model_path)
    else:
        logger.warning("Stable Diffusion not available; placeholder images will be created")

    generated_paths: list[str] = []

    for idx, scene in enumerate(selected, start=start_idx_1_based):
        visual_prompt = scene.get("visual_prompt") or scene.get("prompt") or scene.get("description")
        if not visual_prompt:
            logger.warning("Scene %d missing visual_prompt/description; skipping", idx)
            continue
        negative_prompt = scene.get("negative_prompt")

        # Style-specific prompt enhancement
        if args.style == "realistic":
            # Enhance prompt for realistic style
            if not visual_prompt.lower().startswith(("photorealistic", "realistic", "high quality")):
                visual_prompt = f"photorealistic, high quality, detailed, {visual_prompt}"
            # Update negative prompt to avoid cartoon elements
            if negative_prompt:
                negative_prompt += ", cartoon, anime, illustration, drawing"
            else:
                negative_prompt = "cartoon, anime, illustration, drawing, painting"
        else:  # cartoon
            # Enhance prompt for cartoon style
            if not visual_prompt.lower().startswith(("cartoon", "anime")):
                visual_prompt = f"cartoon illustration, 2D animation style, flat colors, simple shapes, {visual_prompt}"
            # Update negative prompt to avoid realistic elements
            if negative_prompt:
                negative_prompt += ", photorealistic, realistic, 3d render, cgi"
            else:
                negative_prompt = "photorealistic, realistic, 3d render, cgi"

        out_path = output_dir / f"scene_{idx:02d}_{args.style}.png"
        logger.info("[%d/%d] Generating %s image: %s", idx - start_idx_1_based + 1, len(selected), args.style, out_path)
        logger.info("Prompt: %s", visual_prompt)
        if negative_prompt:
            logger.info("Negative prompt: %s", negative_prompt)

        if args.no_validation:
            final_path = image_gen.generate_cartoon_image(visual_prompt, str(out_path), negative_prompt=negative_prompt)
        else:
            final_path = image_gen.generate_cartoon_image_with_validation(visual_prompt, str(out_path), max_attempts=3, negative_prompt=negative_prompt)
        generated_paths.append(final_path)

    if not generated_paths:
        logger.error("No images were generated.")
        sys.exit(2)

    print("\n✅ Generated images:")
    for p in generated_paths:
        try:
            size_kb = Path(p).stat().st_size / 1024
            print(f"  - {p} ({size_kb:.1f} KB)")
        except Exception:
            print(f"  - {p}")
    print(f"\n📁 Saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
