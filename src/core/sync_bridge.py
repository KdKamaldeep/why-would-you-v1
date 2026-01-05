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
import threading
from pathlib import Path
from typing import Optional, Tuple, Dict
import cv2
import numpy as np

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

# Face detection model cache (singleton pattern)
_face_detector = None
_face_detector_type = None


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


def get_face_detector():
    """
    Get face detector (InsightFace preferred, MediaPipe fallback).
    Uses singleton pattern to avoid reloading models.
    
    Returns:
        Tuple of (detector_object, detector_type_string)
        Returns (None, None) if no detector available
    """
    global _face_detector, _face_detector_type
    
    if _face_detector is not None:
        return _face_detector, _face_detector_type
    
    # Try InsightFace first
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        _face_detector = app
        _face_detector_type = "insightface"
        logger.info("✅ Using InsightFace for face detection")
        return _face_detector, _face_detector_type
    except ImportError:
        logger.debug("InsightFace not available, trying MediaPipe...")
    except Exception as e:
        logger.warning(f"Failed to initialize InsightFace: {e}, trying MediaPipe...")
    
    # Fallback to MediaPipe
    try:
        import mediapipe as mp
        mp_face_detection = mp.solutions.face_detection
        detector = mp_face_detection.FaceDetection(
            model_selection=1,  # 0 = short-range, 1 = full-range
            min_detection_confidence=0.5
        )
        _face_detector = detector
        _face_detector_type = "mediapipe"
        logger.info("✅ Using MediaPipe for face detection")
        return _face_detector, _face_detector_type
    except ImportError:
        logger.warning("MediaPipe not available for face detection")
    except Exception as e:
        logger.warning(f"Failed to initialize MediaPipe: {e}")
    
    return None, None


def detect_face_bbox(frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect face bounding box in a frame.
    
    Args:
        frame: BGR frame (numpy array from cv2)
        
    Returns:
        Tuple of (x, y, w, h) bounding box, or None if no face detected
    """
    detector, detector_type = get_face_detector()
    
    if detector is None:
        logger.error("❌ No face detector available. Install insightface or mediapipe.")
        return None
    
    if detector_type == "insightface":
        # InsightFace expects BGR
        faces = detector.get(frame)
        if not faces:
            return None
        # Get first face (largest by confidence/area)
        face = faces[0]
        bbox = face.bbox.astype(int)  # (x1, y1, x2, y2)
        x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
        return (x, y, w, h)
    
    elif detector_type == "mediapipe":
        # MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb_frame)
        
        if not results.detections:
            return None
        
        # Get first detection (largest by default)
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        
        h, w_frame = frame.shape[:2]
        x = int(bbox.xmin * w_frame)
        y = int(bbox.ymin * h)
        w = int(bbox.width * w_frame)
        h_bbox = int(bbox.height * h)
        
        return (x, y, w, h_bbox)
    
    return None


def calculate_face_crop_params(video_path: str) -> Optional[Dict]:
    """
    Detect face in first valid frame and calculate crop parameters.
    
    Args:
        video_path: Path to input video
        
    Returns:
        Dictionary with crop parameters:
        - 'x', 'y', 'w', 'h': Original bounding box
        - 'crop_x', 'crop_y', 'crop_size': Square crop coordinates and size
        - 'orig_width', 'orig_height': Original video dimensions
        Returns None if no face detected
    """
    logger.info("🔍 Detecting face in video...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"❌ Failed to open video: {video_path}")
        return None
    
    # Try to read first few frames (some videos start with black frames)
    max_attempts = 30
    frame = None
    frame_num = 0
    
    for _ in range(max_attempts):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Try to detect face
        bbox = detect_face_bbox(frame)
        if bbox is not None:
            frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
            logger.info(f"✅ Face detected in frame {frame_num}")
            break
    
    cap.release()
    
    if frame is None or bbox is None:
        logger.warning("⚠️ No face detected in video, cannot perform face-centered cropping")
        return None
    
    x, y, w, h = bbox
    frame_height, frame_width = frame.shape[:2]
    
    logger.info(f"📐 Original bbox: x={x}, y={y}, w={w}, h={h}")
    logger.info(f"📐 Frame size: {frame_width}x{frame_height}")
    
    # Expand bounding box by 1.5×
    expansion_factor = 1.5
    new_w = int(w * expansion_factor)
    new_h = int(h * expansion_factor)
    
    # Center the expanded box on the original box
    center_x = x + w // 2
    center_y = y + h // 2
    new_x = center_x - new_w // 2
    new_y = center_y - new_h // 2
    
    # Clamp to frame bounds
    new_x = max(0, new_x)
    new_y = max(0, new_y)
    new_x = min(frame_width - new_w, new_x) if new_w < frame_width else 0
    new_y = min(frame_height - new_h, new_y) if new_h < frame_height else 0
    
    # Ensure box stays within bounds (adjust size if necessary)
    if new_x + new_w > frame_width:
        new_w = frame_width - new_x
    if new_y + new_h > frame_height:
        new_h = frame_height - new_y
    
    logger.info(f"📐 Expanded bbox: x={new_x}, y={new_y}, w={new_w}, h={new_h}")
    
    # Create square crop
    crop_size = max(new_w, new_h)
    
    # Center square on the expanded box center
    crop_x = center_x - crop_size // 2
    crop_y = center_y - crop_size // 2
    
    # Clamp to frame bounds
    crop_x = max(0, crop_x)
    crop_y = max(0, crop_y)
    if crop_x + crop_size > frame_width:
        crop_x = frame_width - crop_size
    if crop_y + crop_size > frame_height:
        crop_y = frame_height - crop_size
    crop_x = max(0, crop_x)
    crop_y = max(0, crop_y)
    
    # Adjust crop_size if it would exceed bounds
    if crop_x + crop_size > frame_width:
        crop_size = frame_width - crop_x
    if crop_y + crop_size > frame_height:
        crop_size = frame_height - crop_y
    
    logger.info(f"📐 Square crop: x={crop_x}, y={crop_y}, size={crop_size}")
    
    return {
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'crop_x': crop_x,
        'crop_y': crop_y,
        'crop_size': crop_size,
        'orig_width': frame_width,
        'orig_height': frame_height
    }


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
    
    # Check input files exist and are files (not directories)
    ensure_ok(os.path.exists(video_path), f"❌ Video file does not exist: {video_path}")
    ensure_ok(os.path.isfile(video_path), f"❌ Video path is not a file: {video_path}")
    ensure_ok(os.path.exists(audio_path), f"❌ Audio file does not exist: {audio_path}")
    ensure_ok(os.path.isfile(audio_path), f"❌ Audio path is not a file: {audio_path}")
    ensure_ok(os.path.exists(inference_ckpt_path), f"❌ Inference checkpoint file does not exist: {inference_ckpt_path}")
    ensure_ok(os.path.isfile(inference_ckpt_path), f"❌ Inference checkpoint path is not a file (may be a directory): {inference_ckpt_path}")
    
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


def preprocess_video(input_video: str, output_video: str, fps: int, crop_params: Optional[Dict] = None) -> None:
    """
    Preprocess video: detect face, crop to square, resize to 512x512, convert to CFR, remove audio, encode H.264.
    
    Args:
        input_video: Path to input video file
        output_video: Path to output video file
        fps: Target frame rate (must be exact CFR)
        crop_params: Optional crop parameters dict (if None, will detect face automatically)
    """
    logger.info(f"Preprocessing video: {input_video} -> {output_video}")
    logger.info(f"Target FPS: {fps} (CFR)")
    
    # Detect face and calculate crop if not provided
    if crop_params is None:
        crop_params = calculate_face_crop_params(input_video)
    
    if crop_params is None:
        logger.warning("⚠️ No face detected, using center crop fallback")
        # Fallback: use center crop
        cap = cv2.VideoCapture(input_video)
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            crop_size = min(w, h)
            crop_x = (w - crop_size) // 2
            crop_y = (h - crop_size) // 2
            crop_params = {
                'crop_x': crop_x,
                'crop_y': crop_y,
                'crop_size': crop_size,
                'orig_width': w,
                'orig_height': h
            }
        cap.release()
    
    if crop_params is None:
        logger.error("❌ Failed to determine crop parameters")
        ensure_ok(False, "Cannot proceed without crop parameters")
    
    # Extract frames, crop, resize to 512x512
    logger.info("🎬 Extracting and processing frames...")
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        ensure_ok(False, f"❌ Failed to open video: {input_video}")
    
    # Get video properties
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp(prefix='sync_bridge_frames_')
    frames_dir = os.path.join(temp_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    
    crop_x = crop_params['crop_x']
    crop_y = crop_params['crop_y']
    crop_size = crop_params['crop_size']
    target_size = 512
    
    frame_count = 0
    processed_frames = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Crop to square
            cropped = frame[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
            
            # Resize to 512x512
            resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
            
            processed_frames.append(resized)
            frame_count += 1
        
        cap.release()
        
        logger.info(f"✅ Processed {frame_count} frames, crop: {crop_size}x{crop_size} -> {target_size}x{target_size}")
        
        # Write frames as images
        for i, frame in enumerate(processed_frames):
            frame_path = os.path.join(frames_dir, f"{i:06d}.jpg")
            cv2.imwrite(frame_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        
        # Encode video from frames with CFR
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', os.path.join(frames_dir, '%06d.jpg'),
            '-an',  # Remove audio
            '-c:v', 'libx264',
            '-preset', 'veryfast',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            '-r', str(fps),  # CFR enforcement
            output_video
        ]
        
        exit_code, stdout, stderr = run_cmd(cmd)
        if exit_code != 0:
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)
            ensure_ok(False, f"❌ Video encoding failed:\n{stderr}")
        
        logger.info(f"✅ Video preprocessed: {output_video}")
        
    finally:
        # Cleanup temporary frames
        shutil.rmtree(temp_dir, ignore_errors=True)


def composite_synced_video(
    original_video: str,
    synced_cropped_video: str,
    output_video: str,
    crop_params: Dict,
    fps: int
) -> None:
    """
    Composite the synced cropped video back into the original video.
    
    Args:
        original_video: Path to original wide video
        synced_cropped_video: Path to synced 512x512 cropped video from LatentSync
        output_video: Path to output composited video
        crop_params: Crop parameters dict from calculate_face_crop_params
        fps: Frame rate
    """
    logger.info("🖼️ Compositing synced video back into original frames...")
    
    crop_x = crop_params['crop_x']
    crop_y = crop_params['crop_y']
    crop_size = crop_params['crop_size']
    orig_width = crop_params['orig_width']
    orig_height = crop_params['orig_height']
    
    # Open videos
    cap_orig = cv2.VideoCapture(original_video)
    cap_synced = cv2.VideoCapture(synced_cropped_video)
    
    if not cap_orig.isOpened():
        ensure_ok(False, f"❌ Failed to open original video: {original_video}")
    if not cap_synced.isOpened():
        ensure_ok(False, f"❌ Failed to open synced video: {synced_cropped_video}")
    
    # Get video properties
    synced_fps = cap_synced.get(cv2.CAP_PROP_FPS)
    synced_frame_count = int(cap_synced.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"📐 Original video: {orig_width}x{orig_height}")
    logger.info(f"📐 Crop region: {crop_x},{crop_y} size {crop_size}x{crop_size}")
    logger.info(f"📐 Synced video: {synced_frame_count} frames @ {synced_fps}fps")
    
    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp(prefix='sync_bridge_composite_')
    frames_dir = os.path.join(temp_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    
    frame_count = 0
    
    try:
        # Read frames from both videos and composite
        while True:
            ret_orig, frame_orig = cap_orig.read()
            ret_synced, frame_synced = cap_synced.read()
            
            if not ret_orig or not ret_synced:
                break
            
            # Resize synced frame back to original crop size
            frame_synced_resized = cv2.resize(
                frame_synced,
                (crop_size, crop_size),
                interpolation=cv2.INTER_LANCZOS4
            )
            
            # Composite: replace crop region in original frame
            frame_composited = frame_orig.copy()
            
            # Ensure crop region is within bounds
            if (crop_x + crop_size <= orig_width and 
                crop_y + crop_size <= orig_height and
                crop_x >= 0 and crop_y >= 0):
                frame_composited[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size] = frame_synced_resized
            
            # Save frame
            frame_path = os.path.join(frames_dir, f"{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, frame_composited, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            
            frame_count += 1
        
        cap_orig.release()
        cap_synced.release()
        
        logger.info(f"✅ Composited {frame_count} frames")
        
        # Encode final video (audio pipeline unchanged - LatentSync handles audio separately)
        cmd_video = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', os.path.join(frames_dir, '%06d.jpg'),
            '-an',  # No audio (audio pipeline unchanged)
            '-c:v', 'libx264',
            '-preset', 'veryfast',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            '-r', str(fps),
            output_video
        ]
        
        exit_code, stdout, stderr = run_cmd(cmd_video)
        if exit_code != 0:
            shutil.rmtree(temp_dir, ignore_errors=True)
            ensure_ok(False, f"❌ Video encoding failed:\n{stderr}")
        
        logger.info(f"✅ Composited video saved: {output_video}")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


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
    
    # Add -u flag to Python executable for unbuffered output
    cmd = [
        LATENTSYNC_PYTHON,
        '-u',  # Unbuffered Python output
        LATENTSYNC_SCRIPT,
        '--video_path', temp_video_abs,
        '--audio_path', temp_audio_abs,
        '--video_out_path', out_path_abs,
        '--unet_config_path', f"{LATENTSYNC_ROOT}/configs/unet/stage2.yaml",
        '--inference_ckpt_path', inference_ckpt_path_abs,
        '--guidance_scale', str(guidance_scale)
    ]
    
    logger.debug(f"LatentSync command: {' '.join(cmd)}")
    logger.debug(f"Working directory: {LATENTSYNC_ROOT}")
    
    # Set PYTHONPATH to LatentSync root so Python can find the latentsync module
    # Also disable Python output buffering for real-time log display
    env = os.environ.copy()
    env["PYTHONPATH"] = LATENTSYNC_ROOT
    env["PYTHONUNBUFFERED"] = "1"  # Disable Python output buffering
    logger.debug(f"Setting PYTHONPATH to: {LATENTSYNC_ROOT}")
    logger.debug("Setting PYTHONUNBUFFERED=1 for real-time output")
    
    # Create log file next to output video
    log_file_path = Path(out_path_abs).parent / "latentsync.log"
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Stream output in real-time with progress visibility
    logger.info("=" * 60)
    logger.info("LatentSync Output (streaming with progress):")
    logger.info(f"Log file: {log_file_path}")
    logger.info("=" * 60)
    
    # Use line-buffered output for real-time progress display
    # Capture BOTH stdout and stderr separately
    process = subprocess.Popen(
        cmd,
        cwd=LATENTSYNC_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # Line buffered for real-time output
        universal_newlines=True
    )
    
    # Stream output in real-time, handling both stdout and stderr
    stdout_lines = []
    stderr_lines = []
    log_lock = threading.Lock()
    
    # Open log file for writing
    log_file = open(log_file_path, 'w', encoding='utf-8')
    
    def read_stdout():
        """Read from stdout and log in real-time."""
        try:
            for line in iter(process.stdout.readline, ''):
                if not line:
                    if process.poll() is not None:
                        break
                    continue
                
                with log_lock:
                    log_file.write(line)
                    log_file.flush()
                
                line_clean = line.rstrip('\n\r')
                if line_clean:
                    stdout_lines.append(line_clean)
                    logger.info(f"LatentSync | {line_clean}")
        except Exception as e:
            logger.warning(f"Error reading stdout: {e}")
    
    def read_stderr():
        """Read from stderr and log in real-time."""
        try:
            for line in iter(process.stderr.readline, ''):
                if not line:
                    if process.poll() is not None:
                        break
                    continue
                
                with log_lock:
                    log_file.write(line)
                    log_file.flush()
                
                line_clean = line.rstrip('\n\r')
                if line_clean:
                    stderr_lines.append(line_clean)
                    logger.warning(f"LatentSync | {line_clean}")
        except Exception as e:
            logger.warning(f"Error reading stderr: {e}")
    
    # Start threads to read from both streams concurrently
    stdout_thread = threading.Thread(target=read_stdout, daemon=True)
    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    
    stdout_thread.start()
    stderr_thread.start()
    
    # Wait for both threads to finish
    stdout_thread.join()
    stderr_thread.join()
    
    # Close log file
    log_file.close()
    
    logger.info(f"✅ LatentSync logs saved to: {log_file_path}")
    
    # Wait for process to complete
    exit_code = process.wait()
    stdout = '\n'.join(stdout_lines)
    stderr = '\n'.join(stderr_lines)
    
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
        # Step 1: Detect face and calculate crop parameters
        logger.info("=" * 60)
        logger.info("Step 1: Detecting face and calculating crop parameters")
        logger.info("=" * 60)
        crop_params = calculate_face_crop_params(args.video_path)
        
        if crop_params is None:
            logger.warning("⚠️ No face detected, falling back to center crop")
            # Fallback: use center crop
            cap = cv2.VideoCapture(args.video_path)
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                crop_size = min(w, h)
                crop_x = (w - crop_size) // 2
                crop_y = (h - crop_size) // 2
                crop_params = {
                    'crop_x': crop_x,
                    'crop_y': crop_y,
                    'crop_size': crop_size,
                    'orig_width': w,
                    'orig_height': h
                }
            cap.release()
        
        if crop_params is None:
            ensure_ok(False, "❌ Failed to determine crop parameters")
        
        # Step 2: Preprocess video with face-centered cropping (crop + resize to 512x512)
        logger.info("=" * 60)
        logger.info("Step 2: Preprocessing video (face-centered crop + resize to 512x512)")
        logger.info("=" * 60)
        preprocess_video(args.video_path, temp_video, args.fps, crop_params)
        
        # Step 3: Preprocess audio
        logger.info("=" * 60)
        logger.info("Step 3: Preprocessing audio")
        logger.info("=" * 60)
        preprocess_audio(args.audio_path, temp_audio, args.sr)
        
        # Step 4: Run LatentSync on cropped video
        logger.info("=" * 60)
        logger.info("Step 4: Running LatentSync inference on cropped video")
        logger.info("=" * 60)
        temp_synced_video = os.path.join(temp_dir, 'temp_synced.mp4')
        run_latentsync(temp_video, temp_audio, temp_synced_video, args.inference_ckpt_path, args.guidance_scale)
        
        # Step 5: Composite synced video back into original frames
        logger.info("=" * 60)
        logger.info("Step 5: Compositing synced video back into original frames")
        logger.info("=" * 60)
        composite_synced_video(
            args.video_path,
            temp_synced_video,
            args.out_path,
            crop_params,
            args.fps
        )
        
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

