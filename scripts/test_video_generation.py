#!/usr/bin/env python3
"""
Simple script to test WAN 2.1 text-to-video generation with a prompt.
Usage:
    python -m scripts.test_video_generation --prompt "A cat walks on the grass"
    python -m scripts.test_video_generation -p "A dog running in a park"
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.wan_t2v import WanT2VGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Test WAN 2.1 text-to-video generation with a prompt"
    )
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        required=True,
        help="Text prompt for video generation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/test_video.mp4",
        help="Output video path (default: outputs/test_video.mp4)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Video width (default: 832)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Video height (default: 480)"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=49,
        help="Number of frames to generate (default: 49)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=12,
        help="Frames per second (default: 12)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of inference steps (default: 30)"
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=6.0,
        help="Guidance scale (default: 6.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional)"
    )
    parser.add_argument(
        "--negative-prompt", "--negative",
        type=str,
        default=None,
        help="Negative prompt (default: excludes non-realistic styles like cartoon, anime, etc.)"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("WAN 2.1 Text-to-Video Generation Test")
    logger.info("=" * 60)
    logger.info(f"📝 Prompt: {args.prompt}")
    logger.info(f"💾 Output: {args.output}")
    logger.info(f"📐 Dimensions: {args.width}x{args.height}")
    logger.info(f"🎞️ Frames: {args.num_frames} @ {args.fps}fps (~{args.num_frames/args.fps:.1f}s)")
    logger.info(f"⚙️ Steps: {args.steps}, Guidance: {args.guidance}")
    if args.seed:
        logger.info(f"🎲 Seed: {args.seed}")
    logger.info("=" * 60)
    
    try:
        # Initialize WAN generator
        logger.info("🔄 Initializing WAN 2.1 T2V generator...")
        generator = WanT2VGenerator(
            width=args.width,
            height=args.height,
            num_frames=args.num_frames,
            fps=args.fps,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            negative_prompt=args.negative_prompt
        )
        
        if not generator.is_available():
            logger.error("❌ WAN pipeline is not available. Please check your setup.")
            return 1
        
        logger.info("✅ WAN generator initialized successfully")
        
        # Generate video
        logger.info("🎬 Starting video generation...")
        # Use custom negative prompt if provided, otherwise use generator's default
        custom_negative_prompt = args.negative_prompt if args.negative_prompt else None
        if custom_negative_prompt:
            logger.info(f"🚫 Using custom negative prompt: {custom_negative_prompt[:100]}{'...' if len(custom_negative_prompt) > 100 else ''}")
        
        output_file = generator.generate_video(
            prompt=args.prompt,
            output_path=str(output_path),
            seed=args.seed,
            negative_prompt=custom_negative_prompt
        )
        
        logger.info("=" * 60)
        logger.info("✅ Video generation completed successfully!")
        logger.info(f"📹 Output file: {output_file}")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error during video generation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
