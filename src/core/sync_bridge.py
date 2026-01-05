#!/usr/bin/env python3
"""
sync_bridge.py - Bridge script to sync WAN-generated video with audio using LatentSync.

This script preprocesses video and audio files, then runs LatentSync inference
to create a synchronized output video.
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Hardcoded LatentSync paths
LATENTSYNC_ROOT = "/workspace/LatentSync"
LATENTSYNC_PYTHON = "/workspace/LatentSync/venv/bin/python"
LATENTSYNC_SCRIPT = "/workspace/LatentSync/scripts/inference.py"


def run_cmd(cmd: list[str], cwd: Optional[str] = None, capture_output: bool = True, env: Optional[dict] = None) -> Tuple[int, str, str]:
    """
    Run a shell command and return exit code, stdout, stderr.
    
    Args:
        cmd: Command as list of strings
        cwd: Working directory (optional)
        capture_output: Whether to capture stdout/stderr
        env: Environment variables dict (optional)
        
    Returns:
        Tuple of (exit_code, stdout, stderr)
    """
    logger.debug(f"Running command: {' '.join(cmd)}")
    if cwd:
        logger.debug(f"Working directory: {cwd}")
    if env:
        logger.debug(f"Using custom environment with PYTHONPATH: {env.get('PYTHONPATH', 'not set')}")
    
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=capture_output,
        text=True,
        check=False
    )
    
    return result.returncode, result.stdout, result.stderr


def ensure_ok(condition: bool, message: str) -> None:
    """
    Fail fast if condition is False.
    
    Args:
        condition: Condition to check
        message: Error message if condition is False
    """
    if not condition:
        logger.error(message)
        sys.exit(1)


def validate_paths(video_path: str, audio_path: str, out_path: str, inference_ckpt_path: str) -> None:
    """
    Validate input paths and requirements.
    
    Args:
        video_path: Path to input video file
        audio_path: Path to input audio file
        out_path: Path to output file
        inference_ckpt_path: Path to LatentSync checkpoint file
    """
    logger.info("Validating paths and requirements...")
    
    # Check for absolute paths
    ensure_ok(os.path.isabs(video_path), f"❌ --video_path must be absolute: {video_path}")
    ensure_ok(os.path.isabs(audio_path), f"❌ --audio_path must be absolute: {audio_path}")
    ensure_ok(os.path.isabs(out_path), f"❌ --out_path must be absolute: {out_path}")
    ensure_ok(os.path.isabs(inference_ckpt_path), f"❌ --inference_ckpt_path must be absolute: {inference_ckpt_path}")
    
    # Check input files exist
    ensure_ok(os.path.exists(video_path), f"❌ Video file does not exist: {video_path}")
    ensure_ok(os.path.exists(audio_path), f"❌ Audio file does not exist: {audio_path}")
    ensure_ok(os.path.exists(inference_ckpt_path), f"❌ Inference checkpoint file does not exist: {inference_ckpt_path}")
    
    # Create parent directory for output if needed
    out_parent = Path(out_path).parent
    if not out_parent.exists():
        logger.info(f"Creating output directory: {out_parent}")
        out_parent.mkdir(parents=True, exist_ok=True)
    
    # Check ffmpeg exists
    ffmpeg_path = shutil.which("ffmpeg")
    ensure_ok(ffmpeg_path is not None, "❌ ffmpeg not found in PATH")
    logger.info(f"✅ Found ffmpeg: {ffmpeg_path}")
    
    # Check LatentSync python exists
    ensure_ok(os.path.exists(LATENTSYNC_PYTHON), f"❌ LatentSync python not found: {LATENTSYNC_PYTHON}")
    ensure_ok(os.path.isfile(LATENTSYNC_PYTHON), f"❌ LatentSync python is not a file: {LATENTSYNC_PYTHON}")
    logger.info(f"✅ Found LatentSync python: {LATENTSYNC_PYTHON}")
    
    # Check LatentSync script exists
    ensure_ok(os.path.exists(LATENTSYNC_SCRIPT), f"❌ LatentSync script not found: {LATENTSYNC_SCRIPT}")
    ensure_ok(os.path.isfile(LATENTSYNC_SCRIPT), f"❌ LatentSync script is not a file: {LATENTSYNC_SCRIPT}")
    logger.info(f"✅ Found LatentSync script: {LATENTSYNC_SCRIPT}")
    
    # Check LatentSync root exists
    ensure_ok(os.path.exists(LATENTSYNC_ROOT), f"❌ LatentSync root not found: {LATENTSYNC_ROOT}")
    logger.info(f"✅ Found LatentSync root: {LATENTSYNC_ROOT}")
    
    logger.info("✅ All validations passed")


def preprocess_video(input_video: str, output_video: str, fps: int) -> None:
    """
    Preprocess video: convert to CFR, remove audio, encode H.264.
    
    Args:
        input_video: Path to input video file
        output_video: Path to output video file
        fps: Target frame rate (must be exact CFR)
    """
    logger.info(f"Preprocessing video: {input_video} -> {output_video}")
    logger.info(f"Target FPS: {fps} (CFR)")
    
    cmd = [
        'ffmpeg', '-y',
        '-i', input_video,
        '-an',  # Remove audio
        '-vf', f'fps={fps},format=yuv420p',  # Force CFR and pixel format
        '-r', str(fps),  # Additional CFR enforcement
        '-c:v', 'libx264',
        '-preset', 'veryfast',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        output_video
    ]
    
    exit_code, stdout, stderr = run_cmd(cmd)
    ensure_ok(exit_code == 0, f"❌ Video preprocessing failed:\n{stderr}")
    
    logger.info(f"✅ Video preprocessed: {output_video}")


def preprocess_audio(input_audio: str, output_audio: str, sample_rate: int) -> None:
    """
    Preprocess audio: convert to mono, resample, PCM 16-bit.
    
    Args:
        input_audio: Path to input audio file
        output_audio: Path to output audio file
        sample_rate: Target sample rate
    """
    logger.info(f"Preprocessing audio: {input_audio} -> {output_audio}")
    logger.info(f"Target sample rate: {sample_rate} Hz, mono, PCM 16-bit")
    
    cmd = [
        'ffmpeg', '-y',
        '-i', input_audio,
        '-ac', '1',  # Mono
        '-ar', str(sample_rate),  # Sample rate
        '-acodec', 'pcm_s16le',  # PCM 16-bit
        output_audio
    ]
    
    exit_code, stdout, stderr = run_cmd(cmd)
    ensure_ok(exit_code == 0, f"❌ Audio preprocessing failed:\n{stderr}")
    
    logger.info(f"✅ Audio preprocessed: {output_audio}")


def run_latentsync(temp_video: str, temp_audio: str, out_path: str, inference_ckpt_path: str, guidance_scale: float) -> None:
    """
    Run LatentSync inference with real-time log streaming.
    
    Args:
        temp_video: Path to preprocessed video file
        temp_audio: Path to preprocessed audio file
        out_path: Path to output file
        inference_ckpt_path: Path to LatentSync checkpoint file
        guidance_scale: Guidance scale for LatentSync
    """
    logger.info("Running LatentSync inference...")
    logger.info(f"Video: {temp_video}")
    logger.info(f"Audio: {temp_audio}")
    logger.info(f"Output: {out_path}")
    logger.info(f"Checkpoint: {inference_ckpt_path}")
    logger.info(f"Guidance scale: {guidance_scale}")
    
    # Make paths absolute for LatentSync
    temp_video_abs = os.path.abspath(temp_video)
    temp_audio_abs = os.path.abspath(temp_audio)
    out_path_abs = os.path.abspath(out_path)
    inference_ckpt_path_abs = os.path.abspath(inference_ckpt_path)
    
    cmd = [
        LATENTSYNC_PYTHON,
        LATENTSYNC_SCRIPT,
        '--video_path', temp_video_abs,
        '--audio_path', temp_audio_abs,
        '--video_out_path', out_path_abs,
        '--inference_ckpt_path', inference_ckpt_path_abs,
        '--guidance_scale', str(guidance_scale)
    ]
    
    logger.debug(f"LatentSync command: {' '.join(cmd)}")
    logger.debug(f"Working directory: {LATENTSYNC_ROOT}")
    
    # Set PYTHONPATH to LatentSync root so Python can find the latentsync module
    env = os.environ.copy()
    env["PYTHONPATH"] = LATENTSYNC_ROOT
    logger.debug(f"Setting PYTHONPATH to: {LATENTSYNC_ROOT}")
    
    # Stream output in real-time
    logger.info("=" * 60)
    logger.info("LatentSync Output (streaming):")
    logger.info("=" * 60)
    
    process = subprocess.Popen(
        cmd,
        cwd=LATENTSYNC_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
        text=True,
        bufsize=1,  # Line buffered
        universal_newlines=True
    )
    
    # Stream output line by line
    stdout_lines = []
    for line in process.stdout:
        line = line.rstrip()
        if line:  # Only print non-empty lines
            print(line, flush=True)  # Print to console in real-time
            stdout_lines.append(line)
    
    # Wait for process to complete
    exit_code = process.wait()
    stdout = '\n'.join(stdout_lines)
    stderr = ''  # Already merged into stdout
    
    if exit_code != 0:
        # Extract last ~2000 chars of stderr and stdout
        stderr_tail = stderr[-2000:] if len(stderr) > 2000 else stderr
        stdout_tail = stdout[-2000:] if len(stdout) > 2000 else stdout
        
        error_msg = f"❌ LatentSync inference failed (exit code {exit_code})\n"
        if stderr_tail:
            error_msg += f"\n=== STDERR (last 2000 chars) ===\n{stderr_tail}\n"
        if stdout_tail:
            error_msg += f"\n=== STDOUT (last 2000 chars) ===\n{stdout_tail}\n"
        
        raise RuntimeError(error_msg)
    
    logger.info("=" * 60)
    logger.info(f"✅ LatentSync inference completed: {out_path}")
    logger.info("=" * 60)


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Bridge WAN-generated video to LatentSync for audio-video synchronization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sync_bridge.py \\
    --video_path /absolute/path/to/video.mp4 \\
    --audio_path /absolute/path/to/audio.wav \\
    --out_path /absolute/path/to/output.mp4 \\
    --inference_ckpt_path /absolute/path/to/checkpoint.ckpt
  
  python sync_bridge.py \\
    --video_path /path/to/video.mp4 \\
    --audio_path /path/to/audio.mp3 \\
    --out_path /path/to/output.mp4 \\
    --inference_ckpt_path /path/to/checkpoint.ckpt \\
    --fps 30 \\
    --sr 22050 \\
    --guidance_scale 2.0 \\
    --keep_temp \\
    --verbose
        """
    )
    
    # Required arguments (absolute paths only)
    parser.add_argument(
        '--video_path',
        required=True,
        help='Absolute path to input WAN video (MP4)'
    )
    parser.add_argument(
        '--audio_path',
        required=True,
        help='Absolute path to input audio file (WAV/MP3)'
    )
    parser.add_argument(
        '--out_path',
        required=True,
        help='Absolute path to output synchronized video (MP4)'
    )
    parser.add_argument(
        '--inference_ckpt_path',
        required=True,
        help='Absolute path to LatentSync inference checkpoint file'
    )
    
    # Optional arguments
    parser.add_argument(
        '--fps',
        type=int,
        default=25,
        help='Target frame rate for video preprocessing (default: 25)'
    )
    parser.add_argument(
        '--sr',
        type=int,
        default=16000,
        help='Target sample rate for audio preprocessing (default: 16000)'
    )
    parser.add_argument(
        '--guidance_scale',
        type=float,
        default=1.5,
        help='Guidance scale for LatentSync (default: 1.5)'
    )
    parser.add_argument(
        '--keep_temp',
        action='store_true',
        help='Keep temporary files after completion'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled")
    
    # Validate paths and requirements
    validate_paths(args.video_path, args.audio_path, args.out_path, args.inference_ckpt_path)
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix='sync_bridge_')
    logger.info(f"Using temporary directory: {temp_dir}")
    
    temp_video = os.path.join(temp_dir, 'temp_video.mp4')
    temp_audio = os.path.join(temp_dir, 'temp_audio.wav')
    
    try:
        # Preprocess video and audio (MANDATORY, always run)
        logger.info("=" * 60)
        logger.info("Step 1: Preprocessing video")
        logger.info("=" * 60)
        preprocess_video(args.video_path, temp_video, args.fps)
        
        logger.info("=" * 60)
        logger.info("Step 2: Preprocessing audio")
        logger.info("=" * 60)
        preprocess_audio(args.audio_path, temp_audio, args.sr)
        
        # Run LatentSync
        logger.info("=" * 60)
        logger.info("Step 3: Running LatentSync inference")
        logger.info("=" * 60)
        run_latentsync(temp_video, temp_audio, args.out_path, args.inference_ckpt_path, args.guidance_scale)
        
        logger.info("=" * 60)
        logger.info("✅ Synchronization completed successfully!")
        logger.info(f"Output: {args.out_path}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Error during synchronization: {e}")
        sys.exit(1)
        
    finally:
        # Cleanup: delete temp files/dir unless --keep_temp is set
        if args.keep_temp:
            logger.info(f"⚠️  Keeping temporary files: {temp_dir}")
        else:
            logger.info("Cleaning up temporary files...")
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    logger.info(f"✅ Cleaned up: {temp_dir}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to clean up temp directory {temp_dir}: {e}")


if __name__ == '__main__':
    main()

