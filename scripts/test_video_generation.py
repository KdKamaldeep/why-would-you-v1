#!/usr/bin/env python3
"""
Simple script to test WAN 2.2 text-to-video generation with a prompt.
Can optionally add audio narration using Coqui TTS.

Usage:
    # Video only
    python -m scripts.test_video_generation --prompt "A cat walks on the grass"
    
    # Video with audio
    python -m scripts.test_video_generation -p "A dog running in a park" \\
      --audio-text "This is a dog running through the park on a sunny day."
    
    # With voice cloning
    python -m scripts.test_video_generation -p "Sunset over mountains" \\
      --audio-text "Beautiful sunset scene" --voice-file path/to/reference.wav
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.wan_t2v import WanT2VGenerator
from src.core.coqui_voice_synthesizer import CoquiVoiceSynthesizer, CoquiVoiceConfig
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Test WAN 2.2 text-to-video generation with a prompt"
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
    parser.add_argument(
        "--audio-text",
        type=str,
        default=None,
        help="Text to convert to speech and add as audio track (optional)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language for audio synthesis (default: en)"
    )
    parser.add_argument(
        "--voice-file",
        type=str,
        default=None,
        help="Path to reference audio file for voice cloning (optional)"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("WAN 2.2 Text-to-Video Generation Test")
    logger.info("=" * 60)
    logger.info(f"📝 Prompt: {args.prompt}")
    logger.info(f"💾 Output: {args.output}")
    logger.info(f"📐 Dimensions: {args.width}x{args.height}")
    logger.info(f"🎞️ Frames: {args.num_frames} @ {args.fps}fps (~{args.num_frames/args.fps:.1f}s)")
    logger.info(f"⚙️ Steps: {args.steps}, Guidance: {args.guidance}")
    if args.seed:
        logger.info(f"🎲 Seed: {args.seed}")
    if args.audio_text:
        logger.info(f"🎵 Audio text: {args.audio_text[:100]}{'...' if len(args.audio_text) > 100 else ''}")
        logger.info(f"🌐 Audio language: {args.language}")
        if args.voice_file:
            logger.info(f"🎤 Voice file: {args.voice_file}")
    logger.info("=" * 60)
    
    try:
        # Initialize WAN generator
        logger.info("🔄 Initializing WAN 2.2 T2V generator...")
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
        
        # Force cleanup after video generation
        import gc
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
        logger.info("=" * 60)
        logger.info("✅ Video generation completed successfully!")
        logger.info(f"📹 Output file: {output_file}")
        
        # Add audio if audio text is provided
        if args.audio_text:
            logger.info("=" * 60)
            logger.info("🎵 Generating audio from text...")
            logger.info(f"📝 Audio text: {args.audio_text}")
            
            try:
                # Generate audio
                audio_output = Path(output_path).parent / f"{output_path.stem}_audio.wav"
                logger.info(f"🎤 Generating speech audio...")
                
                voice_config = CoquiVoiceConfig(language=args.language)
                voice_synthesizer = CoquiVoiceSynthesizer(voice_config)
                
                generated_audio = voice_synthesizer.synthesize_voice(
                    [args.audio_text],
                    str(audio_output),
                    speaker=None,
                    voice_clone_audio=args.voice_file
                )
                
                logger.info(f"✅ Audio generated: {generated_audio}")
                
                # Mix audio with video
                logger.info("🎬 Mixing audio with video...")
                final_output = Path(output_path).parent / f"{output_path.stem}_with_audio.mp4"
                
                # Use FFmpeg to add audio to video
                cmd = [
                    'ffmpeg', '-y',
                    '-i', str(output_file),  # Video input
                    '-i', generated_audio,   # Audio input
                    '-c:v', 'copy',          # Copy video codec (no re-encoding)
                    '-c:a', 'aac',           # Encode audio as AAC
                    '-b:a', '192k',          # Audio bitrate
                    '-shortest',             # Use shortest stream duration
                    str(final_output)
                ]
                
                try:
                    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                    logger.info(f"✅ Audio mixed with video: {final_output}")
                except subprocess.CalledProcessError as e:
                    logger.error(f"❌ FFmpeg error: {e}")
                    logger.error(f"FFmpeg stderr: {e.stderr}")
                    raise
                
                # Clean up temporary audio file
                try:
                    Path(generated_audio).unlink()
                    logger.info("🧹 Cleaned up temporary audio file")
                except Exception as e:
                    logger.warning(f"Could not clean up audio file: {e}")
                
                logger.info("=" * 60)
                logger.info("✅ Final video with audio: {}".format(final_output))
                logger.info("=" * 60)
                
            except Exception as e:
                logger.error(f"❌ Error generating/mixing audio: {e}")
                import traceback
                logger.error(traceback.format_exc())
                logger.info(f"📹 Video without audio is available at: {output_file}")
                return 1
        else:
            logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error during video generation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
