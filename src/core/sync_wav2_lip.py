#!/usr/bin/env python3
"""
sync_wav2_lip.py - Bridge script to sync video with audio using Wav2Lip.

This script normalizes video and audio inputs, then runs Wav2Lip inference
to create a lip-synced output video.
"""

import logging
import os
import shutil
import subprocess
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

# Hardcoded Wav2Lip paths
WAV2LIP_ROOT = "/workspace/Wav2Lip"
WAV2LIP_PYTHON = "/workspace/Wav2Lip/venv/bin/python"
WAV2LIP_SCRIPT = "/workspace/Wav2Lip/inference.py"
WAV2LIP_CHECKPOINT = "/workspace/Wav2Lip/checkpoints/Wav2Lip-SD-GAN.pt"


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
        logger.debug(f"Using custom environment")
    
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=capture_output,
        text=True,
        check=False
    )
    
    return result.returncode, result.stdout, result.stderr


def normalize_video(input_video: str, output_video: str, fps: int) -> bool:
    """
    Normalize video: force constant fps, ensure even dimensions, yuv420p format.
    Uses high-quality encoding (CRF 18) to minimize quality loss during re-encoding.
    
    Args:
        input_video: Path to input video file
        output_video: Path to output video file
        fps: Target frame rate
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.debug(f"Normalizing video: {input_video} -> {output_video}")
        logger.debug(f"Target FPS: {fps}, even dimensions, yuv420p")
        
        cmd = [
            'ffmpeg', '-y',
            '-i', input_video,
            '-r', str(fps),  # Force constant frame rate
            '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',  # Ensure even dimensions
            '-pix_fmt', 'yuv420p',
            '-c:v', 'libx264',  # Explicit codec
            '-crf', '18',  # High quality (lower = better, 18 is visually lossless)
            '-preset', 'medium',  # Balanced encoding speed/quality
            output_video
        ]
        
        exit_code, stdout, stderr = run_cmd(cmd)
        if exit_code != 0:
            logger.error(f"❌ Video normalization failed:\n{stderr}")
            return False
        
        logger.debug(f"✅ Video normalized: {output_video}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error normalizing video: {e}")
        return False


def normalize_audio(input_audio: str, output_audio: str) -> bool:
    """
    Normalize audio: mono, 16kHz.
    
    Args:
        input_audio: Path to input audio file
        output_audio: Path to output audio file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.debug(f"Normalizing audio: {input_audio} -> {output_audio}")
        logger.debug(f"Target: mono, 16kHz")
        
        cmd = [
            'ffmpeg', '-y',
            '-i', input_audio,
            '-ac', '1',  # Mono
            '-ar', '16000',  # 16kHz sample rate
            output_audio
        ]
        
        exit_code, stdout, stderr = run_cmd(cmd)
        if exit_code != 0:
            logger.error(f"❌ Audio normalization failed:\n{stderr}")
            return False
        
        logger.debug(f"✅ Audio normalized: {output_audio}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error normalizing audio: {e}")
        return False


def run_wav2lip(
    norm_video: str,
    norm_audio: str,
    output_video: str,
    wav2lip_dir: str,
    checkpoint_path: str,
    python_cmd: Optional[str] = None,
    pads: str = "0 20 0 0",
    nosmooth: bool = True
) -> bool:
    """
    Run Wav2Lip inference using its virtual environment Python.
    
    Args:
        norm_video: Path to normalized video file
        norm_audio: Path to normalized audio file
        output_video: Path to output video file
        wav2lip_dir: Path to Wav2Lip repository root
        checkpoint_path: Path to Wav2Lip checkpoint file
        python_cmd: Python command to use (optional, uses WAV2LIP_PYTHON if not provided)
        pads: Padding values for face detection (default: "0 20 0 0")
        nosmooth: Disable smoothing (default: True)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Use virtual environment Python if not specified
        if python_cmd is None:
            python_cmd = os.getenv("WAV2LIP_PYTHON", WAV2LIP_PYTHON)
        
        # Make paths absolute
        norm_video_abs = os.path.abspath(norm_video)
        norm_audio_abs = os.path.abspath(norm_audio)
        output_video_abs = os.path.abspath(output_video)
        checkpoint_path_abs = os.path.abspath(checkpoint_path)
        inference_script = os.path.join(wav2lip_dir, "inference.py")
        
        logger.debug(f"Running Wav2Lip inference...")
        logger.debug(f"Python: {python_cmd}")
        logger.debug(f"Video: {norm_video_abs}")
        logger.debug(f"Audio: {norm_audio_abs}")
        logger.debug(f"Output: {output_video_abs}")
        logger.debug(f"Checkpoint: {checkpoint_path_abs}")
        logger.debug(f"Script: {inference_script}")
        
        # Check if Python executable exists
        if not os.path.exists(python_cmd):
            logger.error(f"❌ Wav2Lip Python not found: {python_cmd}")
            logger.error(f"   Expected at: {WAV2LIP_PYTHON}")
            logger.error(f"   Or set WAV2LIP_PYTHON environment variable")
            return False
        
        # Check if inference script exists
        if not os.path.exists(inference_script):
            logger.error(f"❌ Wav2Lip inference script not found: {inference_script}")
            return False
        
        cmd = [
            python_cmd,
            '-u',  # Unbuffered output
            inference_script,
            '--checkpoint_path', checkpoint_path_abs,
            '--face', norm_video_abs,
            '--audio', norm_audio_abs,
            '--outfile', output_video_abs,
            '--pads', pads
        ]
        
        if nosmooth:
            cmd.append('--nosmooth')
        
        exit_code, stdout, stderr = run_cmd(cmd, cwd=wav2lip_dir)
        
        if exit_code != 0:
            logger.error(f"❌ Wav2Lip inference failed (exit code {exit_code})")
            logger.debug(f"STDOUT:\n{stdout[-2000:] if len(stdout) > 2000 else stdout}")
            logger.debug(f"STDERR:\n{stderr[-2000:] if len(stderr) > 2000 else stderr}")
            return False
        
        logger.debug(f"✅ Wav2Lip inference completed: {output_video_abs}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error running Wav2Lip: {e}")
        return False


def lipsync_wav2lip(
    in_video_mp4: str,
    in_audio_wav: str,
    out_video_mp4: str,
    fps: int,
    wav2lip_dir: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    python_cmd: Optional[str] = None,
    pads: str = "0 20 0 0",
    nosmooth: bool = True
) -> bool:
    """
    Lip-sync video with audio using Wav2Lip.
    
    Normalizes inputs, runs Wav2Lip inference, and verifies output.
    Uses Wav2Lip's virtual environment Python (similar to LatentSync).
    
    Args:
        in_video_mp4: Path to input video file
        in_audio_wav: Path to input audio file
        out_video_mp4: Path to output lip-synced video file
        fps: Target frame rate for video normalization
        wav2lip_dir: Path to Wav2Lip repository root (default: from env or /workspace/Wav2Lip)
        checkpoint_path: Path to Wav2Lip checkpoint file (default: from env or /workspace/Wav2Lip/checkpoints/wav2lip_gan.pth)
        python_cmd: Python command to use (default: from env or /workspace/Wav2Lip/venv/bin/python)
        
    Returns:
        True if lip-sync succeeded and out_video_mp4 exists with non-trivial size, else False.
    """
    # Use environment variables or defaults
    if wav2lip_dir is None:
        wav2lip_dir = os.getenv("WAV2LIP_DIR", WAV2LIP_ROOT)
    if checkpoint_path is None:
        checkpoint_path = os.getenv("WAV2LIP_CHECKPOINT", WAV2LIP_CHECKPOINT)
    if python_cmd is None:
        python_cmd = os.getenv("WAV2LIP_PYTHON", WAV2LIP_PYTHON)
    
    logger.info(f"🎬 Wav2Lip: Starting lip-sync process...")
    logger.info(f"   Input video: {in_video_mp4}")
    logger.info(f"   Input audio: {in_audio_wav}")
    logger.info(f"   Output video: {out_video_mp4}")
    logger.info(f"   FPS: {fps}")
    logger.info(f"   Wav2Lip dir: {wav2lip_dir}")
    logger.info(f"   Python: {python_cmd}")
    
    # Validate inputs exist
    if not os.path.exists(in_video_mp4):
        logger.error(f"❌ Input video does not exist: {in_video_mp4}")
        return False
    
    if not os.path.exists(in_audio_wav):
        logger.error(f"❌ Input audio does not exist: {in_audio_wav}")
        return False
    
    if not os.path.exists(wav2lip_dir):
        logger.error(f"❌ Wav2Lip directory does not exist: {wav2lip_dir}")
        return False
    
    if not os.path.exists(checkpoint_path):
        logger.error(f"❌ Wav2Lip checkpoint does not exist: {checkpoint_path}")
        return False
    
    # Create temporary directory for normalized files
    temp_dir = tempfile.mkdtemp(prefix='wav2lip_')
    try:
        norm_video = os.path.join(temp_dir, 'norm_video.mp4')
        norm_audio = os.path.join(temp_dir, 'norm_audio.wav')
        
        # Step 1: Normalize video
        logger.info("📹 Step 1: Normalizing video...")
        if not normalize_video(in_video_mp4, norm_video, fps):
            logger.error("❌ Video normalization failed")
            return False
        
        # Step 2: Normalize audio
        logger.info("🎵 Step 2: Normalizing audio...")
        if not normalize_audio(in_audio_wav, norm_audio):
            logger.error("❌ Audio normalization failed")
            return False
        
        # Step 3: Run Wav2Lip (using virtual environment Python)
        logger.info("🎬 Step 3: Running Wav2Lip inference...")
        if not run_wav2lip(norm_video, norm_audio, out_video_mp4, wav2lip_dir, checkpoint_path, python_cmd, pads, nosmooth):
            logger.error("❌ Wav2Lip inference failed")
            return False
        
        # Step 4: Verify output exists and has non-trivial size
        if not os.path.exists(out_video_mp4):
            logger.error(f"❌ Output video was not created: {out_video_mp4}")
            return False
        
        file_size = os.path.getsize(out_video_mp4)
        min_size = 1024  # 1KB minimum (non-trivial)
        if file_size < min_size:
            logger.error(f"❌ Output video is too small ({file_size} bytes), likely corrupted")
            return False
        
        logger.info(f"✅ Wav2Lip lip-sync completed successfully!")
        logger.info(f"   Output: {out_video_mp4} ({file_size / 1024 / 1024:.2f} MB)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Unexpected error during Wav2Lip lip-sync: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False
        
    finally:
        # Cleanup temporary files
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.debug(f"🧹 Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to clean up temp directory {temp_dir}: {e}")

