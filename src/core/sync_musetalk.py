#!/usr/bin/env python3
"""
sync_musetalk.py - Bridge script to sync video with audio using MuseTalk.

This script normalizes video and audio inputs, then runs MuseTalk inference
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

# Hardcoded MuseTalk paths
MUSETALK_ROOT = "/workspace/MuseTalk"
MUSETALK_PYTHON = "/workspace/MuseTalk/venv/bin/python"
MUSETALK_SCRIPT = "/workspace/MuseTalk/scripts/inference.py"
MUSETALK_MODEL_DIR = "/workspace/MuseTalk/models"


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
    
    # Return stdout and stderr (stderr may be empty if merged)
    stderr_output = result.stderr if result.stderr else ""
    return result.returncode, result.stdout, stderr_output


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
    Normalize audio: mono, 16kHz (MuseTalk typically works with 16kHz audio).
    
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


def run_musetalk(
    norm_video: str,
    norm_audio: str,
    output_video: str,
    musetalk_dir: str,
    python_cmd: Optional[str] = None,
    bbox_shift: int = 0,
    device: str = "cuda"
) -> bool:
    """
    Run MuseTalk inference using its virtual environment Python.
    
    Args:
        norm_video: Path to normalized video file
        norm_audio: Path to normalized audio file
        output_video: Path to output video file
        musetalk_dir: Path to MuseTalk repository root
        python_cmd: Python command to use (optional, uses MUSETALK_PYTHON if not provided)
        bbox_shift: Bounding box shift parameter (controls mask region, affects mouth openness, default: 0)
        device: Device to use ('cuda' or 'cpu', default: 'cuda')
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Use virtual environment Python if not specified
        if python_cmd is None:
            python_cmd = os.getenv("MUSETALK_PYTHON", MUSETALK_PYTHON)
        
        # Make paths absolute
        norm_video_abs = os.path.abspath(norm_video)
        norm_audio_abs = os.path.abspath(norm_audio)
        output_video_abs = os.path.abspath(output_video)
        
        musetalk_dir_abs = os.path.abspath(musetalk_dir)
        inference_script = "scripts/inference.py"  # Relative to musetalk_dir
        
        logger.debug(f"Running MuseTalk inference...")
        logger.debug(f"Python: {python_cmd}")
        logger.debug(f"Working directory: {musetalk_dir_abs}")
        logger.debug(f"Video: {norm_video_abs}")
        logger.debug(f"Audio: {norm_audio_abs}")
        logger.debug(f"Output: {output_video_abs}")
        logger.debug(f"Script: {inference_script}")
        logger.debug(f"Bbox shift: {bbox_shift}")
        logger.debug(f"Device: {device}")
        
        # Check if Python executable exists
        if not os.path.exists(python_cmd):
            logger.error(f"❌ MuseTalk Python not found: {python_cmd}")
            logger.error(f"   Expected at: {MUSETALK_PYTHON}")
            logger.error(f"   Or set MUSETALK_PYTHON environment variable")
            return False
        
        # Check if inference script exists
        inference_script_abs = os.path.join(musetalk_dir_abs, inference_script)
        if not os.path.exists(inference_script_abs):
            logger.error(f"❌ MuseTalk inference script not found: {inference_script_abs}")
            return False
        
        # Build command
        cmd = [
            python_cmd,
            '-u',  # Unbuffered output
            inference_script,  # Relative path (will be resolved from cwd)
            '--video_path', norm_video_abs,
            '--audio_path', norm_audio_abs,
            '--output_path', output_video_abs,
            '--bbox_shift', str(bbox_shift),
            '--device', device
        ]
        
        logger.info(f"Running MuseTalk command from: {musetalk_dir_abs}")
        logger.info(f"Command: {' '.join(cmd)}")
        
        exit_code, stdout, stderr = run_cmd(cmd, cwd=musetalk_dir_abs)
        
        if exit_code != 0:
            logger.error(f"❌ MuseTalk inference failed (exit code {exit_code})")
            logger.error(f"Command: {' '.join(cmd)}")
            logger.error(f"Working directory: {musetalk_dir_abs}")
            if stdout:
                logger.error(f"STDOUT (last 5000 chars):\n{stdout[-5000:] if len(stdout) > 5000 else stdout}")
            if stderr:
                logger.error(f"STDERR (last 5000 chars):\n{stderr[-5000:] if len(stderr) > 5000 else stderr}")
            if not stdout and not stderr:
                logger.error("No output captured from MuseTalk - this may indicate a Python environment issue")
            return False
        
        logger.debug(f"✅ MuseTalk inference completed: {output_video_abs}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error running MuseTalk: {e}")
        return False


def lipsync_musetalk(
    in_video_mp4: str,
    in_audio_wav: str,
    out_video_mp4: str,
    fps: int,
    musetalk_dir: Optional[str] = None,
    python_cmd: Optional[str] = None,
    bbox_shift: int = 0,
    device: str = "cuda"
) -> bool:
    """
    Lip-sync video with audio using MuseTalk.
    
    Normalizes inputs, runs MuseTalk inference, and verifies output.
    Uses MuseTalk's virtual environment Python.
    
    Args:
        in_video_mp4: Path to input video file
        in_audio_wav: Path to input audio file
        out_video_mp4: Path to output lip-synced video file
        fps: Target frame rate for video normalization
        musetalk_dir: Path to MuseTalk repository root (default: from env or /workspace/MuseTalk)
        python_cmd: Python command to use (default: from env or /workspace/MuseTalk/venv/bin/python)
        bbox_shift: Bounding box shift parameter (default: 0)
        device: Device to use ('cuda' or 'cpu', default: 'cuda')
        
    Returns:
        True if lip-sync succeeded and out_video_mp4 exists with non-trivial size, else False.
    """
    # Use environment variables or defaults
    if musetalk_dir is None:
        musetalk_dir = os.getenv("MUSETALK_DIR", MUSETALK_ROOT)
    if python_cmd is None:
        python_cmd = os.getenv("MUSETALK_PYTHON", MUSETALK_PYTHON)
    
    logger.info(f"🎬 MuseTalk: Starting lip-sync process...")
    logger.info(f"   Input video: {in_video_mp4}")
    logger.info(f"   Input audio: {in_audio_wav}")
    logger.info(f"   Output video: {out_video_mp4}")
    logger.info(f"   FPS: {fps}")
    logger.info(f"   MuseTalk dir: {musetalk_dir}")
    logger.info(f"   Python: {python_cmd}")
    logger.info(f"   Bbox shift: {bbox_shift}")
    logger.info(f"   Device: {device}")
    
    # Validate inputs exist
    if not os.path.exists(in_video_mp4):
        logger.error(f"❌ Input video does not exist: {in_video_mp4}")
        return False
    
    if not os.path.exists(in_audio_wav):
        logger.error(f"❌ Input audio does not exist: {in_audio_wav}")
        return False
    
    if not os.path.exists(musetalk_dir):
        logger.error(f"❌ MuseTalk directory does not exist: {musetalk_dir}")
        return False
    
    # Create temporary directory for normalized files
    temp_dir = tempfile.mkdtemp(prefix='musetalk_')
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
        
        # Step 3: Run MuseTalk (using virtual environment Python)
        logger.info("🎬 Step 3: Running MuseTalk inference...")
        if not run_musetalk(norm_video, norm_audio, out_video_mp4, musetalk_dir, python_cmd, bbox_shift, device):
            logger.error("❌ MuseTalk inference failed")
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
        
        logger.info(f"✅ MuseTalk lip-sync completed successfully!")
        logger.info(f"   Output: {out_video_mp4} ({file_size / 1024 / 1024:.2f} MB)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Unexpected error during MuseTalk lip-sync: {e}")
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

