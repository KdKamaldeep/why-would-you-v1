#!/usr/bin/env python3
"""
Script to check sync_wav2_lip by generating audio and video with WAN and Coqui,
then running lipsync using Wav2Lip.

If audio and video already exist, generation is skipped.

Usage:
    # Generate video using I2V (Gemini generates image, then WAN generates video)
    python scripts/check_sync_bridge.py \
        --video-prompt "A cat walks on the grass" \
        --audio-text "This is a cat walking on the grass" \
        --output-dir outputs/test_sync

    # Skip generation if files exist
    python scripts/check_sync_bridge.py \
        --video-prompt "A cat walks on the grass" \
        --audio-text "This is a cat walking on the grass" \
        --output-dir outputs/test_sync \
        --video-path outputs/test_sync/video.mp4 \
        --audio-path outputs/test_sync/audio.wav

    # Use T2V mode (skip Gemini, use text-to-video directly)
    python scripts/check_sync_bridge.py \
        --video-prompt "A cat walks on the grass" \
        --audio-text "This is a cat walking on the grass" \
        --output-dir outputs/test_sync \
        --skip-gemini
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables (try .env first, then config.env as fallback)
load_dotenv()
if not os.getenv("GEMINI_API_KEY") and Path("config.env").exists():
    load_dotenv("config.env")

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.wan_t2v import WanT2VGenerator
from src.core.coqui_voice_synthesizer import CoquiVoiceSynthesizer, CoquiVoiceConfig
from src.core.gemini_image_generator import GeminiImageGenerator
from src.core.sync_wav2_lip import lipsync_wav2lip

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def check_file_exists(file_path: str, description: str) -> bool:
    """Check if a file exists and log the result."""
    exists = os.path.exists(file_path) and os.path.isfile(file_path)
    if exists:
        logger.info(f"✅ {description} already exists: {file_path}")
    else:
        logger.info(f"❌ {description} not found: {file_path}")
    return exists


def generate_video_with_wan(
    prompt: str,
    output_path: str,
    width: int = 1280,
    height: int = 720,
    num_frames: int = 25,
    fps: int = 24,
    num_inference_steps: int = 30,
    guidance_scale: float = 6.0,
    seed: int = None,
    negative_prompt: str = None,
    skip_gemini: bool = False
) -> str:
    """
    Generate video using WAN I2V (Image-to-Video) with Gemini-generated image.
    
    Args:
        prompt: Text prompt for video generation (used for both Gemini image and WAN video)
        output_path: Path to save the video
        width: Video width
        height: Video height
        num_frames: Number of frames to generate
        fps: Frames per second
        num_inference_steps: Number of inference steps
        guidance_scale: Guidance scale
        seed: Random seed (optional)
        negative_prompt: Negative prompt (optional)
        skip_gemini: If True, skip Gemini image generation and use T2V mode (optional)
        
    Returns:
        Path to generated video file
    """
    logger.info("=" * 60)
    logger.info("🎬 Generating video with WAN I2V (Image-to-Video)...")
    logger.info("=" * 60)
    
    # Create output directory if needed
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    initial_image_path = None
    
    # Step 1: Generate initial frame with Gemini (unless skipped)
    if not skip_gemini:
        logger.info("=" * 60)
        logger.info("🎨 Step 1: Generating initial frame with Gemini...")
        logger.info("=" * 60)
        
        gemini_generator = GeminiImageGenerator()
        
        if not gemini_generator.available:
            logger.warning("⚠️ Gemini not available, falling back to T2V mode")
            logger.warning("💡 Make sure GEMINI_API_KEY is set and google-genai is installed")
        else:
            initial_image_path = output_path_obj.parent / f"{output_path_obj.stem}_initial_frame.png"
            logger.info(f"📝 Generating image from prompt: {prompt[:100]}...")
            
            generated_image = gemini_generator.generate_image(
                prompt=prompt,
                output_path=str(initial_image_path),
                width=width,
                height=height
            )
            
            if generated_image:
                initial_image_path = generated_image
                logger.info(f"✅ Initial frame generated: {initial_image_path}")
            else:
                logger.warning("⚠️ Failed to generate initial frame, falling back to T2V mode")
                initial_image_path = None
    else:
        logger.info("⏭️ Skipping Gemini image generation (using T2V mode)")
    
    # Step 2: Generate video with WAN (I2V if image available, T2V otherwise)
    logger.info("=" * 60)
    logger.info(f"🎬 Step 2: Generating video with WAN ({'I2V' if initial_image_path else 'T2V'} mode)...")
    logger.info("=" * 60)
    
    # Initialize WAN generator
    wan_generator = WanT2VGenerator(
        width=width,
        height=height,
        num_frames=num_frames,
        fps=fps,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt or "text, subtitles, watermark, blurry, low quality, cartoon, anime, manga, illustration, painting, drawing, sketch, bad anatomy, distorted, deformed, ugly"
    )
    
    # Generate video (I2V if image available, T2V otherwise)
    result = wan_generator.generate_video(
        prompt=prompt,
        output_path=output_path,
        seed=seed,
        negative_prompt=negative_prompt,
        image=str(initial_image_path) if initial_image_path else None  # Pass image for I2V mode
    )
    
    # Handle return value (can be string or dict)
    if isinstance(result, dict):
        video_path = result.get('video_path', output_path)
    else:
        video_path = result
    
    logger.info(f"✅ Video generated: {video_path}")
    return video_path


def combine_video_audio(video_path: str, audio_path: str, output_path: str) -> str:
    """
    Combine video and audio into final output using ffmpeg.
    
    Args:
        video_path: Path to input video file
        audio_path: Path to input audio file
        output_path: Path to output video file with audio
        
    Returns:
        Path to output video file
        
    Raises:
        RuntimeError: If combination fails
    """
    logger.info("=" * 60)
    logger.info("🎬 Combining synced video with audio...")
    logger.info("=" * 60)
    logger.info(f"📹 Video: {video_path}")
    logger.info(f"🎵 Audio: {audio_path}")
    logger.info(f"🎬 Output: {output_path}")
    
    # Create output directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,  # Video input
        '-i', audio_path,  # Audio input
        '-c:v', 'copy',    # Copy video codec (no re-encoding)
        '-c:a', 'aac',     # Encode audio as AAC
        '-b:a', '192k',    # Audio bitrate
        '-shortest',       # Use shortest stream duration
        '-movflags', '+faststart',
        output_path
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"✅ Final video with audio created: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to combine video and audio: {e}")
        if e.stderr:
            logger.error(f"STDERR: {e.stderr[-500:]}")
        raise RuntimeError(f"Failed to combine video and audio: {e}")


def convert_video_fps(input_video: str, output_video: str, target_fps: int = 25) -> str:
    """
    Convert video to target FPS using ffmpeg.
    
    Args:
        input_video: Path to input video file
        output_video: Path to output video file
        target_fps: Target frame rate (default: 25)
        
    Returns:
        Path to converted video file
    """
    logger.info(f"🔄 Converting video FPS: {input_video} -> {output_video}")
    logger.info(f"Target FPS: {target_fps}")
    
    # Create output directory if needed
    Path(output_video).parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        'ffmpeg', '-y',
        '-i', input_video,
        '-vf', f'fps={target_fps}',
        '-c:v', 'libx264',
        '-preset', 'veryfast',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        output_video
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"✅ Video FPS converted: {output_video}")
        return output_video
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ FPS conversion failed: {e}")
        logger.error(f"STDERR: {e.stderr[-500:] if e.stderr else 'No stderr'}")
        raise


def generate_audio_with_coqui(
    text: str,
    output_path: str,
    language: str = "en",
    voice_clone_audio: str = None
) -> str:
    """
    Generate audio using Coqui TTS.
    
    Args:
        text: Text to synthesize
        output_path: Path to save the audio
        language: Language code (default: "en")
        voice_clone_audio: Path to reference audio for voice cloning (optional)
        
    Returns:
        Path to generated audio file
    """
    logger.info("=" * 60)
    logger.info("🎵 Generating audio with Coqui TTS...")
    logger.info("=" * 60)
    
    # Create output directory if needed
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize Coqui TTS
    voice_config = CoquiVoiceConfig(language=language)
    voice_synthesizer = CoquiVoiceSynthesizer(voice_config)
    
    # Generate audio
    audio_path = voice_synthesizer.synthesize_voice(
        narration_lines=[text],
        output_path=output_path,
        speaker=None,
        voice_clone_audio=voice_clone_audio
    )
    
    logger.info(f"✅ Audio generated: {audio_path}")
    return audio_path


def find_wav2lip_checkpoint(checkpoint_path: Optional[str] = None) -> str:
    """
    Find Wav2Lip checkpoint file.
    
    Args:
        checkpoint_path: Explicit checkpoint path (optional)
        
    Returns:
        Absolute path to checkpoint file
        
    Raises:
        FileNotFoundError: If no checkpoint found
    """
    if checkpoint_path:
        checkpoint_abs = os.path.abspath(checkpoint_path)
        if os.path.exists(checkpoint_abs):
            return checkpoint_abs
        else:
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_abs}")
    
    # Try default location
    default_checkpoint = Path("/workspace/Wav2Lip/checkpoints/Wav2Lip-SD-GAN.pt")
    if default_checkpoint.exists():
        logger.info(f"✅ Auto-detected Wav2Lip checkpoint: {default_checkpoint}")
        return str(default_checkpoint.absolute())
    
    raise FileNotFoundError(
        f"No Wav2Lip checkpoint found. Please provide --wav2lip-checkpoint or place Wav2Lip-SD-GAN.pt in /workspace/Wav2Lip/checkpoints/"
    )


def run_wav2lip_lipsync(
    video_path: str,
    audio_path: str,
    output_path: str,
    checkpoint_path: str,
    fps: int = 24,
    wav2lip_dir: Optional[str] = None,
    verbose: bool = False
) -> bool:
    """
    Run sync_wav2_lip to perform lipsync using Wav2Lip.
    
    Args:
        video_path: Path to input video
        audio_path: Path to input audio
        output_path: Path to output synchronized video
        checkpoint_path: Path to Wav2Lip checkpoint file
        fps: Target frame rate (default: 24)
        wav2lip_dir: Path to Wav2Lip directory (optional, uses default if None)
        verbose: Enable verbose logging
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 60)
    logger.info("🔄 Running Wav2Lip lipsync...")
    logger.info("=" * 60)
    
    # Convert paths to absolute
    video_path_abs = os.path.abspath(video_path)
    audio_path_abs = os.path.abspath(audio_path)
    output_path_abs = os.path.abspath(output_path)
    
    # Create output directory if needed
    Path(output_path_abs).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        success = lipsync_wav2lip(
            in_video_mp4=video_path_abs,
            in_audio_wav=audio_path_abs,
            out_video_mp4=output_path_abs,
            fps=fps,
            wav2lip_dir=wav2lip_dir,  # None uses default from sync_wav2_lip.py
            checkpoint_path=checkpoint_path,
            python_cmd=None  # Uses Wav2Lip venv Python by default
        )
        
        if success:
            logger.info(f"✅ Wav2Lip lipsync completed: {output_path_abs}")
            return True
        else:
            logger.error(f"❌ Wav2Lip lipsync failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error running Wav2Lip lipsync: {e}")
        import traceback
        if verbose:
            logger.debug(traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate audio and video with WAN and Coqui, then run lipsync",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate everything from scratch with Wav2Lip
  python scripts/check_sync_bridge.py \\
    --video-prompt "A cat walks on the grass" \\
    --audio-text "This is a cat walking on the grass" \\
    --output-dir outputs/test_sync

  # Use existing video and audio files
  python scripts/check_sync_bridge.py \\
    --video-path outputs/test_sync/video.mp4 \\
    --audio-path outputs/test_sync/audio.wav \\
    --output-dir outputs/test_sync

  # Specify custom checkpoint path
  python scripts/check_sync_bridge.py \\
    --video-prompt "A dog running" \\
    --audio-text "A dog is running" \\
    --output-dir outputs/test_sync \\
    --wav2lip-checkpoint /absolute/path/to/Wav2Lip-SD-GAN.pt

  # Custom WAN settings
  python scripts/check_sync_bridge.py \\
    --video-prompt "A dog running" \\
    --audio-text "A dog is running" \\
    --output-dir outputs/test_sync \\
    --width 1280 --height 720 --num-frames 50 --fps 24
        """
    )
    
    # Video generation arguments
    parser.add_argument(
        '--video-prompt',
        type=str,
        default=None,
        help='Text prompt for video generation (required if --video-path not provided)'
    )
    parser.add_argument(
        '--video-path',
        type=str,
        default=None,
        help='Path to existing video file (skips generation if provided)'
    )
    
    # Audio generation arguments
    parser.add_argument(
        '--audio-text',
        type=str,
        default=None,
        help='Text to synthesize as audio (required if --audio-path not provided)'
    )
    parser.add_argument(
        '--audio-path',
        type=str,
        default=None,
        help='Path to existing audio file (skips generation if provided)'
    )
    
    # Output arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for generated files'
    )
    parser.add_argument(
        '--output-name',
        type=str,
        default='synced',
        help='Base name for output files (default: synced)'
    )
    
    # WAN video generation settings
    parser.add_argument(
        '--width',
        type=int,
        default=1280,
        help='Video width (default: 1280)'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=720,
        help='Video height (default: 720)'
    )
    parser.add_argument(
        '--num-frames',
        type=int,
        default=25,
        help='Number of frames to generate (default: 25)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=24,
        help='Frames per second (default: 24)'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=30,
        help='Number of inference steps (default: 30)'
    )
    parser.add_argument(
        '--guidance',
        type=float,
        default=6.0,
        help='Guidance scale for video generation (default: 6.0)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for video generation (optional)'
    )
    parser.add_argument(
        '--negative-prompt',
        type=str,
        default=None,
        help='Negative prompt for video generation (optional)'
    )
    parser.add_argument(
        '--skip-gemini',
        action='store_true',
        help='Skip Gemini image generation and use T2V mode instead of I2V (optional)'
    )
    
    # Coqui TTS settings
    parser.add_argument(
        '--language',
        type=str,
        default='en',
        help='Language for audio synthesis (default: en)'
    )
    parser.add_argument(
        '--voice-file',
        type=str,
        default=None,
        help='Path to reference audio file for voice cloning (optional)'
    )
    
    # Wav2Lip settings
    parser.add_argument(
        '--sync-fps',
        type=int,
        default=24,
        help='Target frame rate for lipsync (default: 24 for Wav2Lip)'
    )
    parser.add_argument(
        '--wav2lip-checkpoint',
        type=str,
        default=None,
        help='Path to Wav2Lip checkpoint file (optional, auto-detects from /workspace/Wav2Lip/checkpoints/wav2lip_gan.pth if not provided)'
    )
    parser.add_argument(
        '--wav2lip-dir',
        type=str,
        default=None,
        help='Path to Wav2Lip directory (optional, uses /workspace/Wav2Lip by default)'
    )
    
    # Other options
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--skip-lipsync',
        action='store_true',
        help='Skip lipsync step (only generate video and audio)'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine video path
    if args.video_path:
        video_path = args.video_path
        if not check_file_exists(video_path, "Video file"):
            logger.error("❌ Video file not found. Please provide a valid --video-path or use --video-prompt to generate.")
            sys.exit(1)
        logger.info("⏭️  Skipping video generation (using existing file)")
    elif args.video_prompt:
        video_path = str(output_dir / "video.mp4")
        if check_file_exists(video_path, "Video file"):
            logger.info("⏭️  Skipping video generation (file already exists)")
        else:
            video_path = generate_video_with_wan(
                prompt=args.video_prompt,
                output_path=video_path,
                width=args.width,
                height=args.height,
                num_frames=args.num_frames,
                fps=args.fps,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                seed=args.seed,
                negative_prompt=args.negative_prompt,
                skip_gemini=args.skip_gemini
            )
    else:
        logger.error("❌ Either --video-prompt or --video-path must be provided")
        sys.exit(1)
    
    # Determine audio path
    if args.audio_path:
        audio_path = args.audio_path
        if not check_file_exists(audio_path, "Audio file"):
            logger.error("❌ Audio file not found. Please provide a valid --audio-path or use --audio-text to generate.")
            sys.exit(1)
        logger.info("⏭️  Skipping audio generation (using existing file)")
    elif args.audio_text:
        audio_path = str(output_dir / "audio.wav")
        if check_file_exists(audio_path, "Audio file"):
            logger.info("⏭️  Skipping audio generation (file already exists)")
        else:
            audio_path = generate_audio_with_coqui(
                text=args.audio_text,
                output_path=audio_path,
                language=args.language,
                voice_clone_audio=args.voice_file
            )
    else:
        logger.error("❌ Either --audio-text or --audio-path must be provided")
        sys.exit(1)
    
    # Run lipsync if not skipped (only Wav2Lip supported)
    if not args.skip_lipsync:
        output_path = str(output_dir / f"{args.output_name}.mp4")
        
        # Wav2Lip path
        logger.info("=" * 60)
        logger.info("🎬 Using Wav2Lip for lipsync")
        logger.info("=" * 60)
        
        # Convert video FPS if needed (Wav2Lip typically works with 24fps, but can use any)
        wav2lip_fps = args.sync_fps if args.sync_fps else args.fps
        if wav2lip_fps != args.fps:
            logger.info("=" * 60)
            logger.info(f"🔄 Converting video from {args.fps} fps to {wav2lip_fps} fps for Wav2Lip...")
            logger.info("=" * 60)
            converted_video_path = str(output_dir / f"{Path(video_path).stem}_{wav2lip_fps}fps.mp4")
            try:
                video_path = convert_video_fps(
                    input_video=video_path,
                    output_video=converted_video_path,
                    target_fps=wav2lip_fps
                )
            except Exception as e:
                logger.error(f"❌ Failed to convert video FPS: {e}")
                logger.warning("⚠️ Continuing with original video (may cause sync issues)")
        
        # Find Wav2Lip checkpoint (auto-detect if not provided)
        try:
            wav2lip_checkpoint = find_wav2lip_checkpoint(args.wav2lip_checkpoint)
        except FileNotFoundError as e:
            logger.error(f"❌ {e}")
            sys.exit(1)
        
        # Run Wav2Lip (outputs synced video without audio)
        # Normalization happens inside lipsync_wav2lip
        synced_video_path = str(output_dir / f"{Path(output_path).stem}_synced_only.mp4")
        success = run_wav2lip_lipsync(
            video_path=video_path,
            audio_path=audio_path,
            output_path=synced_video_path,
            checkpoint_path=wav2lip_checkpoint,
            fps=wav2lip_fps,
            wav2lip_dir=args.wav2lip_dir,
            verbose=args.verbose
        )
        
        if success:
            # Combine synced video with original audio
            try:
                final_output = combine_video_audio(
                    video_path=synced_video_path,
                    audio_path=audio_path,
                    output_path=output_path
                )
                logger.info("=" * 60)
                logger.info("✅ All steps completed successfully!")
                logger.info(f"📹 Original video: {video_path}")
                logger.info(f"🎵 Audio: {audio_path}")
                logger.info(f"🎬 Synced video (no audio): {synced_video_path}")
                logger.info(f"🎬 Final output (with audio): {final_output}")
                logger.info("=" * 60)
            except Exception as e:
                logger.error("=" * 60)
                logger.error(f"❌ Failed to combine video and audio: {e}")
                logger.error("=" * 60)
                sys.exit(1)
        else:
            logger.error("=" * 60)
            logger.error("❌ Wav2Lip lipsync failed. Check logs above for details.")
            logger.error("=" * 60)
            sys.exit(1)
    else:
        logger.info("=" * 60)
        logger.info("✅ Video and audio generation completed (lipsync skipped)")
        logger.info(f"📹 Video: {video_path}")
        logger.info(f"🎵 Audio: {audio_path}")
        logger.info("=" * 60)


if __name__ == '__main__':
    main()
