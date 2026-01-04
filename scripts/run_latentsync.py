#!/usr/bin/env python3
"""
Standalone LatentSync CLI utility

Test LatentSync lip synchronization on individual video/audio pairs.
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()
if not os.getenv("GEMINI_API_KEY") and Path("config.env").exists():
    load_dotenv("config.env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run LatentSync lip synchronization on video and audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/run_latentsync.py --video_in scene.mp4 --audio_in narration.wav --out output_lipsync.mp4
  
  # With face detection mode
  python scripts/run_latentsync.py --video_in scene.mp4 --audio_in narration.wav --out output.mp4 --face_mode manual
  
  # With character reference
  python scripts/run_latentsync.py --video_in scene.mp4 --audio_in narration.wav --out output.mp4 --character_ref character_face.png
        """
    )
    
    parser.add_argument(
        "--video_in",
        type=str,
        required=True,
        help="Input video file path (MP4)"
    )
    
    parser.add_argument(
        "--audio_in",
        type=str,
        required=True,
        help="Input audio file path (WAV/AAC)"
    )
    
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output video file path (MP4)"
    )
    
    parser.add_argument(
        "--face_mode",
        type=str,
        choices=["auto", "manual", "none"],
        default="auto",
        help="Face detection mode (default: auto)"
    )
    
    parser.add_argument(
        "--target_fps",
        type=int,
        default=24,
        help="Target FPS for output (default: 24)"
    )
    
    parser.add_argument(
        "--keep_fps",
        action="store_true",
        help="Keep original video FPS instead of using target_fps"
    )
    
    parser.add_argument(
        "--character_ref",
        type=str,
        default=None,
        help="Optional character reference image path"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Override LATENTSYNC_MODEL_PATH from environment"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Override LATENTSYNC_DEVICE from environment"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    video_path = Path(args.video_in)
    audio_path = Path(args.audio_in)
    
    if not video_path.exists():
        logger.error(f"❌ Input video not found: {args.video_in}")
        sys.exit(1)
    
    if not audio_path.exists():
        logger.error(f"❌ Input audio not found: {args.audio_in}")
        sys.exit(1)
    
    # Import LatentSync
    try:
        from core.latentsync import LatentSyncRunner
    except ImportError as e:
        logger.error(f"❌ Failed to import LatentSync: {e}")
        logger.error("💡 Make sure you're running from the project root and dependencies are installed")
        sys.exit(1)
    
    # Create LatentSync runner
    model_path = args.model_path or os.getenv("LATENTSYNC_MODEL_PATH", "")
    device = args.device or os.getenv("LATENTSYNC_DEVICE", "cuda")
    
    if not model_path:
        logger.error("❌ LATENTSYNC_MODEL_PATH not set and --model_path not provided")
        logger.error("💡 Set LATENTSYNC_MODEL_PATH in .env or use --model_path")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("🎬 LatentSync Lip Sync Test")
    logger.info("=" * 60)
    logger.info(f"📹 Video: {args.video_in}")
    logger.info(f"🎵 Audio: {args.audio_in}")
    logger.info(f"📁 Output: {args.out}")
    logger.info(f"👤 Face mode: {args.face_mode}")
    logger.info(f"🎞️ FPS: {'original' if args.keep_fps else args.target_fps}")
    if args.character_ref:
        logger.info(f"🖼️ Character ref: {args.character_ref}")
    logger.info("=" * 60)
    
    runner = LatentSyncRunner(
        model_path=model_path,
        device=device,
        face_crop=args.face_mode
    )
    
    if not runner.available:
        logger.error("❌ LatentSync not available")
        logger.error("💡 Check LATENTSYNC_MODEL_PATH and model installation")
        sys.exit(1)
    
    # Run LatentSync
    result = runner.run(
        video_in=str(video_path),
        audio_in=str(audio_path),
        video_out=args.out,
        face_crop=args.face_mode,
        keep_fps=args.keep_fps,
        target_fps=args.target_fps,
        character_reference_image=args.character_ref
    )
    
    # Print results
    logger.info("=" * 60)
    if result['success']:
        logger.info("✅ LatentSync completed successfully!")
        logger.info(f"📹 Output: {result['video_path']}")
        logger.info(f"⏱️ Duration: {result['duration_out']:.2f}s")
        logger.info(f"🎞️ FPS: {result['fps']:.1f}")
        
        if result['warnings']:
            logger.warning("⚠️ Warnings:")
            for warning in result['warnings']:
                logger.warning(f"   • {warning}")
    else:
        logger.error("❌ LatentSync failed!")
        logger.error(f"Error: {result.get('error', 'Unknown error')}")
        sys.exit(1)
    
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

