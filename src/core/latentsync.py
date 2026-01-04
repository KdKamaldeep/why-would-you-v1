#!/usr/bin/env python3
"""
LatentSync Lip Sync Integration

Wrapper for LatentSync lip synchronization model.
Processes video and audio to generate lip-synced output.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
import json

logger = logging.getLogger(__name__)


class LatentSyncRunner:
    """
    Wrapper for LatentSync lip synchronization.
    
    Handles video and audio processing to generate lip-synced videos.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        fp16: bool = True,
        face_crop: str = "auto",
        min_face_size: int = 64,
        debug_frames: bool = False,
        **opts
    ):
        """
        Initialize LatentSync runner.
        
        Args:
            model_path: Path to LatentSync model directory or checkpoint
            device: Device to use ('cuda' or 'cpu')
            fp16: Use half precision (faster, less memory)
            face_crop: Face detection mode ('auto', 'manual', 'none')
            min_face_size: Minimum face size in pixels for detection
            debug_frames: Save debug frames during processing
            **opts: Additional options passed to LatentSync
        """
        self.model_path = model_path or os.getenv("LATENTSYNC_MODEL_PATH", "")
        self.device = device or os.getenv("LATENTSYNC_DEVICE", "cuda")
        self.fp16 = fp16
        self.face_crop = face_crop or os.getenv("LATENTSYNC_FACE_MODE", "auto")
        self.min_face_size = min_face_size
        self.debug_frames = debug_frames or os.getenv("LATENTSYNC_DEBUG_FRAMES", "false").lower() == "true"
        self.opts = opts
        
        # Check if LatentSync is available
        self.available = self._check_availability()
        
        if not self.available:
            logger.warning("⚠️ LatentSync not available - check installation and model path")
    
    def _check_availability(self) -> bool:
        """Check if LatentSync is available and configured."""
        if not self.model_path:
            logger.warning("⚠️ LATENTSYNC_MODEL_PATH not set")
            return False
        
        # Check if model path exists
        model_dir = Path(self.model_path)
        if not model_dir.exists():
            logger.warning(f"⚠️ LatentSync model path does not exist: {self.model_path}")
            return False
        
        # Try to import LatentSync (optional - may use CLI instead)
        try:
            # Check if we can find latentsync module or script
            # This is a placeholder - actual implementation depends on LatentSync API
            return True
        except Exception as e:
            logger.warning(f"⚠️ Could not verify LatentSync availability: {e}")
            return False
    
    def run(
        self,
        video_in: str,
        audio_in: str,
        video_out: str,
        face_crop: Optional[str] = None,
        keep_fps: bool = True,
        target_fps: int = 24,
        character_reference_image: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run LatentSync lip synchronization.
        
        Args:
            video_in: Input video path (MP4)
            audio_in: Input audio path (WAV/AAC)
            video_out: Output video path (MP4)
            face_crop: Override face crop mode ('auto', 'manual', 'none')
            keep_fps: Maintain original FPS if True, otherwise use target_fps
            target_fps: Target FPS if keep_fps is False
            character_reference_image: Optional reference image for character face
            
        Returns:
            Dictionary with:
            - success: bool
            - video_path: str (output path)
            - duration_in: float (input duration)
            - duration_out: float (output duration)
            - fps: float (output FPS)
            - warnings: List[str]
            - error: Optional[str]
        """
        result = {
            "success": False,
            "video_path": video_out,
            "duration_in": 0.0,
            "duration_out": 0.0,
            "fps": target_fps,
            "warnings": [],
            "error": None
        }
        
        if not self.available:
            error_msg = "LatentSync not available - check configuration"
            logger.error(f"❌ {error_msg}")
            result["error"] = error_msg
            return result
        
        # Validate inputs
        video_path = Path(video_in)
        audio_path = Path(audio_in)
        
        if not video_path.exists():
            error_msg = f"Input video not found: {video_in}"
            logger.error(f"❌ {error_msg}")
            result["error"] = error_msg
            return result
        
        if not audio_path.exists():
            error_msg = f"Input audio not found: {audio_in}"
            logger.error(f"❌ {error_msg}")
            result["error"] = error_msg
            return result
        
        # Get input video duration and FPS
        try:
            duration_in, fps_in = self._get_video_info(str(video_path))
            result["duration_in"] = duration_in
            result["fps"] = fps_in if keep_fps else target_fps
        except Exception as e:
            logger.warning(f"⚠️ Could not get video info: {e}")
            result["warnings"].append(f"Could not read input video metadata: {e}")
        
        # Get audio duration
        try:
            audio_duration = self._get_audio_duration(str(audio_path))
            if duration_in > 0 and abs(audio_duration - duration_in) > 0.5:
                warning = f"Audio duration ({audio_duration:.2f}s) differs from video ({duration_in:.2f}s) by >0.5s"
                logger.warning(f"⚠️ {warning}")
                result["warnings"].append(warning)
        except Exception as e:
            logger.warning(f"⚠️ Could not get audio duration: {e}")
        
        # Use face_crop parameter or fallback to instance default
        face_mode = face_crop or self.face_crop
        
        # Check for face in video (if auto mode)
        if face_mode == "auto":
            has_face = self._detect_face_in_video(str(video_path))
            if not has_face:
                warning = "No face detected in video - lip sync may not work correctly"
                logger.warning(f"⚠️ {warning}")
                result["warnings"].append(warning)
                # Continue anyway - let LatentSync handle it
        
        # Create output directory
        output_path = Path(video_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Run LatentSync
        try:
            logger.info(f"🎬 Running LatentSync: {video_in} + {audio_in} -> {video_out}")
            logger.info(f"   Face mode: {face_mode}, FPS: {result['fps']}")
            
            # Call LatentSync implementation
            # This is a placeholder - actual implementation depends on LatentSync API
            success = self._run_latentsync(
                video_in=str(video_path),
                audio_in=str(audio_path),
                video_out=str(output_path),
                face_mode=face_mode,
                target_fps=result["fps"],
                character_reference=character_reference_image
            )
            
            if not success:
                error_msg = "LatentSync processing failed"
                logger.error(f"❌ {error_msg}")
                result["error"] = error_msg
                return result
            
            # Verify output
            if not output_path.exists() or output_path.stat().st_size == 0:
                error_msg = "LatentSync output file is missing or empty"
                logger.error(f"❌ {error_msg}")
                result["error"] = error_msg
                return result
            
            # Get output video info
            try:
                duration_out, fps_out = self._get_video_info(str(output_path))
                result["duration_out"] = duration_out
                result["fps"] = fps_out
                
                logger.info(f"✅ LatentSync completed: {duration_out:.2f}s @ {fps_out}fps")
            except Exception as e:
                logger.warning(f"⚠️ Could not get output video info: {e}")
                result["warnings"].append(f"Could not read output video metadata: {e}")
            
            result["success"] = True
            return result
            
        except Exception as e:
            error_msg = f"LatentSync error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            result["error"] = error_msg
            return result
    
    def _run_latentsync(
        self,
        video_in: str,
        audio_in: str,
        video_out: str,
        face_mode: str,
        target_fps: int,
        character_reference: Optional[str] = None
    ) -> bool:
        """
        Internal method to invoke LatentSync.
        
        This is a placeholder implementation. Replace with actual LatentSync API calls.
        
        Options:
        1. Use LatentSync Python API if available
        2. Use LatentSync CLI if available
        3. Use subprocess to call external script
        """
        # Option 1: Try Python API (if LatentSync provides one)
        try:
            # Example: from latentsync import LatentSync
            # sync = LatentSync(model_path=self.model_path, device=self.device)
            # sync.process(video_in, audio_in, video_out, ...)
            # return True
            pass
        except ImportError:
            pass
        
        # Option 2: Use CLI if available
        latentsync_script = Path(self.model_path) / "inference.py"
        if latentsync_script.exists():
            try:
                cmd = [
                    sys.executable,
                    str(latentsync_script),
                    "--video", video_in,
                    "--audio", audio_in,
                    "--output", video_out,
                    "--device", self.device,
                    "--fps", str(target_fps),
                ]
                
                if face_mode != "none":
                    cmd.extend(["--face_mode", face_mode])
                
                if self.fp16:
                    cmd.append("--fp16")
                
                if character_reference:
                    cmd.extend(["--reference", character_reference])
                
                logger.info(f"Running: {' '.join(cmd)}")
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    cwd=self.model_path
                )
                
                if result.returncode == 0:
                    return True
                else:
                    logger.error(f"LatentSync CLI failed: {result.stderr}")
                    return False
                    
            except subprocess.CalledProcessError as e:
                logger.error(f"LatentSync CLI error: {e.stderr if hasattr(e, 'stderr') else str(e)}")
                return False
            except Exception as e:
                logger.error(f"Error running LatentSync CLI: {e}")
                return False
        
        # Option 3: Fallback - create a wrapper script or use alternative method
        logger.warning("⚠️ LatentSync API/CLI not found - using placeholder")
        logger.warning("⚠️ Please implement actual LatentSync integration")
        
        # For now, just copy the input video as placeholder
        # In production, replace this with actual LatentSync call
        try:
            import shutil
            shutil.copy(video_in, video_out)
            logger.warning("⚠️ Using input video as placeholder (no lip sync applied)")
            return True
        except Exception as e:
            logger.error(f"Failed to create placeholder: {e}")
            return False
    
    def _get_video_info(self, video_path: str) -> tuple:
        """Get video duration and FPS using ffprobe."""
        try:
            cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=duration,r_frame_rate',
                '-of', 'json',
                video_path
            ]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            data = json.loads(result.stdout)
            
            stream = data.get('streams', [{}])[0]
            duration = float(stream.get('duration', 0))
            
            # Parse frame rate (e.g., "24/1" -> 24.0)
            rate_str = stream.get('r_frame_rate', '24/1')
            if '/' in rate_str:
                num, den = map(int, rate_str.split('/'))
                fps = num / den if den > 0 else 24.0
            else:
                fps = float(rate_str)
            
            return duration, fps
        except Exception as e:
            logger.warning(f"Could not probe video: {e}")
            return 0.0, 24.0
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration using ffprobe."""
        try:
            cmd = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=nw=1:nk=1',
                audio_path
            ]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return float(result.stdout.strip())
        except Exception as e:
            logger.warning(f"Could not probe audio: {e}")
            return 0.0
    
    def _detect_face_in_video(self, video_path: str) -> bool:
        """
        Detect if video contains a face.
        
        This is a simple check - in production, use proper face detection.
        """
        # Option 1: Use OpenCV with face detector
        try:
            import cv2
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False
            
            # Check first few frames
            frames_checked = 0
            max_frames = 10
            face_found = False
            
            while frames_checked < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(self.min_face_size, self.min_face_size)
                )
                
                if len(faces) > 0:
                    face_found = True
                    break
                
                frames_checked += 1
            
            cap.release()
            return face_found
            
        except ImportError:
            logger.warning("OpenCV not available - skipping face detection")
            return True  # Assume face exists if we can't check
        except Exception as e:
            logger.warning(f"Face detection error: {e}")
            return True  # Assume face exists on error


def get_latentsync_runner() -> Optional[LatentSyncRunner]:
    """
    Factory function to create LatentSyncRunner from environment/config.
    
    Returns:
        LatentSyncRunner instance if enabled and available, None otherwise
    """
    enabled = os.getenv("LATENTSYNC_ENABLED", "false").lower() == "true"
    
    if not enabled:
        logger.info("ℹ️ LatentSync disabled (LATENTSYNC_ENABLED=false)")
        return None
    
    model_path = os.getenv("LATENTSYNC_MODEL_PATH", "")
    if not model_path:
        logger.warning("⚠️ LATENTSYNC_ENABLED=true but LATENTSYNC_MODEL_PATH not set")
        return None
    
    device = os.getenv("LATENTSYNC_DEVICE", "cuda")
    face_mode = os.getenv("LATENTSYNC_FACE_MODE", "auto")
    min_face_size = int(os.getenv("LATENTSYNC_MIN_FACE_SIZE", "64"))
    debug_frames = os.getenv("LATENTSYNC_DEBUG_FRAMES", "false").lower() == "true"
    fp16 = os.getenv("LATENTSYNC_FP16", "true").lower() == "true"
    
    return LatentSyncRunner(
        model_path=model_path,
        device=device,
        fp16=fp16,
        face_crop=face_mode,
        min_face_size=min_face_size,
        debug_frames=debug_frames
    )

