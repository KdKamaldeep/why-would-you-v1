#!/usr/bin/env python3
"""
Test script for MuseTalk lip sync with WAN and Coqui TTS.

This script:
1. Generates a video using WAN (text-to-video)
2. Generates audio using Coqui TTS
3. Applies MuseTalk lip sync
4. Outputs the final lip-synced video

Usage:
    python scripts/test_musetalk_wan_coqui.py --prompt "A cat playing piano" --text "Hello, this is a test of MuseTalk lip sync."
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.wan_t2v import WanT2VGenerator
from src.core.coqui_voice_synthesizer import CoquiVoiceSynthesizer, CoquiVoiceConfig
from src.core.sync_musetalk import lipsync_musetalk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def test_musetalk_wan_coqui(
    prompt: str,
    text: str,
    output_dir: str = "output/test_musetalk",
    fps: int = 24,
    num_frames: int = 50,
    voice: Optional[str] = None,
    device: str = "cuda"
):
    """
    Test MuseTalk lip sync with WAN and Coqui.
    
    Args:
        prompt: Text prompt for WAN video generation
        text: Text to synthesize with Coqui TTS
        output_dir: Output directory for generated files
        fps: Frames per second for video
        num_frames: Number of frames to generate
        voice: Voice name for Coqui TTS (optional)
        device: Device to use for MuseTalk ("cuda" or "cpu")
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("MuseTalk + WAN + Coqui Test")
    logger.info("=" * 80)
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Text: {text}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"FPS: {fps}")
    logger.info(f"Num frames: {num_frames}")
    logger.info(f"Device: {device}")
    logger.info("=" * 80)
    
    # Step 1: Generate video with WAN (skip if already exists)
    video_path = output_path / "wan_video.mp4"
    
    if video_path.exists():
        logger.info(f"\n🎬 Step 1: WAN video already exists, skipping generation...")
        logger.info(f"   Using existing video: {video_path}")
        file_size = video_path.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"   File size: {file_size:.2f} MB")
    else:
        logger.info("\n🎬 Step 1: Generating video with WAN...")
        wan_generator = WanT2VGenerator()
        
        try:
            result = wan_generator.generate_video(
                prompt=prompt,
                output_path=str(video_path),
                seed=None,
                num_frames=num_frames,
                fps=fps
            )
            
            # Handle return type: dict (with metadata) or string (backward compatible)
            if isinstance(result, dict):
                actual_video_path = result.get('video_path', str(video_path))
            else:
                actual_video_path = result if result else str(video_path)
            
            if not Path(actual_video_path).exists():
                logger.error(f"❌ WAN video generation failed: {actual_video_path} does not exist")
                return False
            
            logger.info(f"✅ WAN video generated: {actual_video_path}")
            video_path = Path(actual_video_path)
            
        except Exception as e:
            logger.error(f"❌ WAN video generation failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    # Step 2: Generate audio with Coqui TTS (skip if already exists)
    audio_path = output_path / "coqui_audio.wav"
    
    if audio_path.exists():
        logger.info(f"\n🎵 Step 2: Coqui audio already exists, skipping generation...")
        logger.info(f"   Using existing audio: {audio_path}")
        file_size = audio_path.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"   File size: {file_size:.2f} MB")
    else:
        logger.info("\n🎵 Step 2: Generating audio with Coqui TTS...")
        try:
            coqui_config = CoquiVoiceConfig()
            coqui_synthesizer = CoquiVoiceSynthesizer(config=coqui_config)
            
            synthesized_audio = coqui_synthesizer.synthesize_voice(
                narration_lines=[text],
                output_path=str(audio_path),
                speaker=voice
            )
            
            if not Path(synthesized_audio).exists():
                logger.error(f"❌ Coqui audio generation failed: {synthesized_audio} does not exist")
                return False
            
            logger.info(f"✅ Coqui audio generated: {synthesized_audio}")
            audio_path = Path(synthesized_audio)
            
        except Exception as e:
            logger.error(f"❌ Coqui audio generation failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    # Step 3: Apply MuseTalk lip sync
    logger.info("\n🎙️ Step 3: Applying MuseTalk lip sync...")
    try:
        output_video_path = output_path / "final_musetalk_lipsync.mp4"
        
        success = lipsync_musetalk(
            in_video_mp4=str(video_path),
            in_audio_wav=str(audio_path),
            out_video_mp4=str(output_video_path),
            fps=fps,
            musetalk_dir=None,  # Use default from sync_musetalk.py
            python_cmd=None,  # Use MuseTalk venv Python
            bbox_shift=0,  # Default bbox shift
            device=device
        )
        
        if not success:
            logger.error(f"❌ MuseTalk lip sync failed")
            return False
        
        if not output_video_path.exists():
            logger.error(f"❌ Output video does not exist: {output_video_path}")
            return False
        
        file_size = output_video_path.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"✅ MuseTalk lip sync completed!")
        logger.info(f"   Output: {output_video_path} ({file_size:.2f} MB)")
        
    except Exception as e:
        logger.error(f"❌ MuseTalk lip sync failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("✅ Test completed successfully!")
    logger.info("=" * 80)
    logger.info(f"WAN video: {video_path}")
    logger.info(f"Coqui audio: {audio_path}")
    logger.info(f"Final output: {output_video_path}")
    logger.info("=" * 80)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test MuseTalk lip sync with WAN and Coqui TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic test
  python scripts/test_musetalk_wan_coqui.py --prompt "A cat playing piano" --text "Hello, this is a test."

  # Custom output directory and settings
  python scripts/test_musetalk_wan_coqui.py \\
    --prompt "A dog dancing" \\
    --text "This is a test of MuseTalk lip sync with WAN and Coqui." \\
    --output-dir output/my_test \\
    --fps 24 \\
    --num-frames 50 \\
    --device cuda
        """
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for WAN video generation"
    )
    
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to synthesize with Coqui TTS"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/test_musetalk",
        help="Output directory for generated files (default: output/test_musetalk)"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Frames per second for video (default: 24)"
    )
    
    parser.add_argument(
        "--num-frames",
        type=int,
        default=50,
        help="Number of frames to generate (default: 50)"
    )
    
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        help="Voice name for Coqui TTS (optional)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for MuseTalk (default: cuda)"
    )
    
    args = parser.parse_args()
    
    success = test_musetalk_wan_coqui(
        prompt=args.prompt,
        text=args.text,
        output_dir=args.output_dir,
        fps=args.fps,
        num_frames=args.num_frames,
        voice=args.voice,
        device=args.device
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

