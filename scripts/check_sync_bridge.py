#!/usr/bin/env python3
"""
Script to check sync_bridge by generating audio and video with WAN and Coqui,
then running lipsync on them.

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


def find_latentsync_checkpoint(checkpoint_path: Optional[str] = None) -> str:
    """
    Find LatentSync checkpoint file.
    
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
    default_checkpoint_dir = Path("/workspace/LatentSync/checkpoints")
    if default_checkpoint_dir.exists() and default_checkpoint_dir.is_dir():
        # Look for common checkpoint file extensions
        checkpoint_extensions = ['.ckpt', '.pth', '.pt', '.safetensors']
        checkpoints = []
        
        for ext in checkpoint_extensions:
            checkpoints.extend(list(default_checkpoint_dir.glob(f"*{ext}")))
        
        if checkpoints:
            # Sort by modification time (most recent first)
            checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            checkpoint_path = checkpoints[0]
            logger.info(f"✅ Auto-detected checkpoint: {checkpoint_path}")
            return str(checkpoint_path.absolute())
    
    raise FileNotFoundError(
        f"No checkpoint found. Please provide --inference-ckpt-path or place a checkpoint in {default_checkpoint_dir}"
    )


def run_sync_bridge(
    video_path: str,
    audio_path: str,
    output_path: str,
    inference_ckpt_path: str,
    fps: int = 25,
    sample_rate: int = 16000,
    guidance_scale: float = 1.5,
    verbose: bool = False
) -> bool:
    """
    Run sync_bridge.py to perform lipsync.
    
    Args:
        video_path: Path to input video (must be absolute)
        audio_path: Path to input audio (must be absolute)
        output_path: Path to output synchronized video (must be absolute)
        inference_ckpt_path: Path to LatentSync inference checkpoint file (must be absolute)
        fps: Target frame rate (default: 25)
        sample_rate: Target sample rate (default: 16000)
        guidance_scale: Guidance scale for LatentSync (default: 1.5)
        verbose: Enable verbose logging
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 60)
    logger.info("🔄 Running sync_bridge (lipsync)...")
    logger.info("=" * 60)
    
    # Convert paths to absolute
    video_path_abs = os.path.abspath(video_path)
    audio_path_abs = os.path.abspath(audio_path)
    output_path_abs = os.path.abspath(output_path)
    inference_ckpt_path_abs = os.path.abspath(inference_ckpt_path)
    
    # Create output directory if needed
    Path(output_path_abs).parent.mkdir(parents=True, exist_ok=True)
    
    # Get sync_bridge script path
    sync_bridge_script = Path(__file__).parent.parent / "src" / "core" / "sync_bridge.py"
    
    if not sync_bridge_script.exists():
        logger.error(f"❌ sync_bridge.py not found at: {sync_bridge_script}")
        return False
    
    # Build command
    cmd = [
        sys.executable,
        str(sync_bridge_script),
        '--video_path', video_path_abs,
        '--audio_path', audio_path_abs,
        '--out_path', output_path_abs,
        '--inference_ckpt_path', inference_ckpt_path_abs,
        '--fps', str(fps),
        '--sr', str(sample_rate),
        '--guidance_scale', str(guidance_scale)
    ]
    
    if verbose:
        cmd.append('--verbose')
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info(f"✅ Lipsync completed: {output_path_abs}")
            return True
        else:
            logger.error(f"❌ Lipsync failed with exit code {result.returncode}")
            if result.stderr:
                logger.error(f"STDERR: {result.stderr[-1000:]}")
            if result.stdout:
                logger.error(f"STDOUT: {result.stdout[-1000:]}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error running sync_bridge: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate audio and video with WAN and Coqui, then run lipsync",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate everything from scratch (auto-detects checkpoint from /workspace/LatentSync/checkpoints)
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
    --inference-ckpt-path /absolute/path/to/checkpoint.ckpt

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
    
    # Sync bridge settings
    parser.add_argument(
        '--inference-ckpt-path',
        type=str,
        default=None,
        help='Absolute path to LatentSync inference checkpoint file (optional, auto-detects from /workspace/LatentSync/checkpoints if not provided)'
    )
    parser.add_argument(
        '--sync-fps',
        type=int,
        default=25,
        help='Target frame rate for sync_bridge (default: 25)'
    )
    parser.add_argument(
        '--sync-sr',
        type=int,
        default=16000,
        help='Target sample rate for sync_bridge (default: 16000)'
    )
    parser.add_argument(
        '--sync-guidance',
        type=float,
        default=1.5,
        help='Guidance scale for LatentSync (default: 1.5)'
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
    
    # Run lipsync if not skipped
    if not args.skip_lipsync:
        output_path = str(output_dir / f"{args.output_name}.mp4")
        
        # Convert video to 25 fps (LatentSync requirement) before sync
        # WAN generates at 24 fps, but LatentSync needs 25 fps
        if args.sync_fps != args.fps:
            logger.info("=" * 60)
            logger.info(f"🔄 Converting video from {args.fps} fps to {args.sync_fps} fps for LatentSync...")
            logger.info("=" * 60)
            converted_video_path = str(output_dir / f"{Path(video_path).stem}_25fps.mp4")
            try:
                video_path = convert_video_fps(
                    input_video=video_path,
                    output_video=converted_video_path,
                    target_fps=args.sync_fps
                )
            except Exception as e:
                logger.error(f"❌ Failed to convert video FPS: {e}")
                logger.warning("⚠️ Continuing with original video (may cause sync issues)")
        
        # Find checkpoint (auto-detect if not provided)
        try:
            inference_ckpt_path = find_latentsync_checkpoint(args.inference_ckpt_path)
        except FileNotFoundError as e:
            logger.error(f"❌ {e}")
            sys.exit(1)
        
        success = run_sync_bridge(
            video_path=video_path,
            audio_path=audio_path,
            output_path=output_path,
            inference_ckpt_path=inference_ckpt_path,
            fps=args.sync_fps,
            sample_rate=args.sync_sr,
            guidance_scale=args.sync_guidance,
            verbose=args.verbose
        )
        
        if success:
            logger.info("=" * 60)
            logger.info("✅ All steps completed successfully!")
            logger.info(f"📹 Video: {video_path}")
            logger.info(f"🎵 Audio: {audio_path}")
            logger.info(f"🎬 Synced output: {output_path}")
            logger.info("=" * 60)
        else:
            logger.error("=" * 60)
            logger.error("❌ Lipsync failed. Check logs above for details.")
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
