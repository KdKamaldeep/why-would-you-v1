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
from typing import Optional, Dict, Any, Union, List, Tuple
import json
import random

logger = logging.getLogger(__name__)

# Import face alignment module
try:
    from .face_align import get_face_aligner, FaceAligner
    FACE_ALIGN_AVAILABLE = True
except ImportError:
    FACE_ALIGN_AVAILABLE = False
    logger.warning("⚠️ Face alignment module not available")


def get_cache_dir() -> str:
    """
    Get the cache directory for Hugging Face models.
    Always uses /workspace/.cache/huggingface (creates if needed).
    
    Returns:
        Cache directory path as string
    """
    workspace_cache = Path("/workspace/.cache/huggingface")
    workspace_cache.mkdir(parents=True, exist_ok=True)
    # Set HF_HOME environment variable as well
    os.environ["HF_HOME"] = str(workspace_cache)
    return str(workspace_cache)


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
        # Quality parameters (from LatentSync docs: inference_steps [20-50], guidance_scale [1.0-3.0])
        # Higher inference_steps = better quality but slower
        # Higher guidance_scale = better lip sync but may cause distortion
        self.inference_steps = int(os.getenv("LATENTSYNC_INFERENCE_STEPS", "40"))  # Default 40 for quality
        self.guidance_scale = float(os.getenv("LATENTSYNC_GUIDANCE_SCALE", "2.0"))  # Default 2.0 for balance
        # LatentSync repository root directory (default: /workspace/LatentSync)
        self.latentsync_dir = Path(os.getenv("LATENTSYNC_DIR", "/workspace/LatentSync"))
        # UNet config file relative path (default: configs/unet/stage2.yaml)
        self.unet_config_rel = os.getenv("LATENTSYNC_UNET_CONFIG_REL", "configs/unet/stage2.yaml")
        
        # Face alignment settings
        self.use_face_alignment = os.getenv("LATENTSYNC_USE_FACE_ALIGNMENT", "true").lower() == "true"
        self.face_detection_method = os.getenv("LATENTSYNC_FACE_DETECTION", "mediapipe")
        self.face_smoothing_alpha = float(os.getenv("LATENTSYNC_FACE_SMOOTHING", "0.7"))
        self.face_min_confidence = float(os.getenv("LATENTSYNC_FACE_MIN_CONFIDENCE", "0.5"))
        self.composite_back = os.getenv("LATENTSYNC_COMPOSITE_BACK", "true").lower() == "true"
        
        # Initialize face aligner if available
        self.face_aligner = None
        if self.use_face_alignment and FACE_ALIGN_AVAILABLE:
            try:
                self.face_aligner = get_face_aligner(
                    detection_method=self.face_detection_method,
                    crop_size=512,
                    smoothing_alpha=self.face_smoothing_alpha,
                    min_face_size=self.min_face_size,
                    min_confidence=self.face_min_confidence
                )
                logger.info(f"✅ Face alignment enabled (method: {self.face_detection_method})")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize face aligner: {e}")
                self.face_aligner = None
        
        # Python executable for LatentSync (can be from different virtualenv)
        latentsync_python = os.getenv("LATENTSYNC_PYTHON", None)
        if latentsync_python:
            python_path = Path(latentsync_python)
            if python_path.exists() and python_path.is_file():
                self.python_executable = str(python_path)
                logger.info(f"🐍 Using custom Python executable for LatentSync: {self.python_executable}")
            else:
                logger.warning(f"⚠️ LATENTSYNC_PYTHON path does not exist: {latentsync_python}, using default")
                self.python_executable = sys.executable
        else:
            self.python_executable = sys.executable
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
        
        # Check if model_path is a Hugging Face model ID (contains '/')
        # or if it's a local path that doesn't exist yet
        is_hf_model_id = '/' in self.model_path and not Path(self.model_path).exists()
        model_dir = Path(self.model_path)
        
        if is_hf_model_id or not model_dir.exists():
            # Try to download from Hugging Face
            if is_hf_model_id:
                logger.info(f"📥 Detected Hugging Face model ID: {self.model_path}")
                logger.info("🔄 Will download model automatically on first use")
            else:
                logger.info(f"📥 Model path does not exist: {self.model_path}")
                logger.info("🔄 Attempting to download from Hugging Face...")
            
            # Download model from Hugging Face
            downloaded_path = self._download_model_from_hf(self.model_path)
            if downloaded_path:
                self.model_path = downloaded_path
                model_dir = Path(self.model_path)
            else:
                logger.warning(f"⚠️ Could not download LatentSync model: {self.model_path}")
                return False
        
        # Verify model directory exists and contains expected files
        if not model_dir.exists() or not model_dir.is_dir():
            logger.warning(f"⚠️ LatentSync model path is not a valid directory: {self.model_path}")
            return False
        
        # Check for common LatentSync model files
        expected_files = [
            "latentsync_unet.pt",
            "inference.py",
            "config.json"
        ]
        has_any_file = any((model_dir / f).exists() for f in expected_files)
        
        if not has_any_file:
            # Check for subdirectories that might contain the model
            has_subdirs = any(d.is_dir() for d in model_dir.iterdir())
            if not has_subdirs:
                logger.warning(f"⚠️ LatentSync model directory exists but doesn't contain expected files: {self.model_path}")
                logger.warning("⚠️ Model may need to be downloaded manually")
                # Still return True - let the actual loading handle errors
                return True
        
        # Try to import LatentSync (optional - may use CLI instead)
        try:
            # Check if we can find latentsync module or script
            # This is a placeholder - actual implementation depends on LatentSync API
            return True
        except Exception as e:
            logger.warning(f"⚠️ Could not verify LatentSync availability: {e}")
            return False
    
    def _download_model_from_hf(self, model_id_or_path: str) -> Optional[str]:
        """
        Download LatentSync model from Hugging Face if needed.
        
        Args:
            model_id_or_path: Hugging Face model ID (e.g., "ByteDance/LatentSync-1.6") or local path
            
        Returns:
            Path to downloaded model directory, or None if download failed
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            logger.warning("⚠️ huggingface_hub not available - cannot download models automatically")
            logger.info("💡 Install with: pip install huggingface_hub")
            return None
        
        # Determine if it's a Hugging Face model ID
        is_hf_id = '/' in model_id_or_path and not Path(model_id_or_path).exists()
        
        if not is_hf_id:
            # It's a local path that doesn't exist - try common HF model IDs
            logger.info("💡 Trying default LatentSync model from Hugging Face: ByteDance/LatentSync-1.6")
            model_id = "ByteDance/LatentSync-1.6"
        else:
            model_id = model_id_or_path
        
        # Get cache directory (always uses /workspace/.cache/huggingface)
        cache_dir = get_cache_dir()
        logger.info(f"📁 Using Hugging Face cache directory: {cache_dir}")
        
        # Determine download location
        if is_hf_id:
            # For Hugging Face model IDs, use cache directory
            download_dir = Path(cache_dir) / "hub" / model_id.replace("/", "--")
        else:
            # For local paths that don't exist, download to cache directory
            download_dir = Path(cache_dir) / "hub" / model_id.replace("/", "--")
            logger.info(f"💡 Local path doesn't exist, downloading to cache: {download_dir}")
        
        download_dir.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"📥 Downloading LatentSync model: {model_id}")
            logger.info(f"📁 Download location: {download_dir}")
            logger.info("⏳ This may take a while on first run...")
            
            downloaded_path = snapshot_download(
                repo_id=model_id,
                local_dir=str(download_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            
            logger.info(f"✅ LatentSync model downloaded to: {downloaded_path}")
            return downloaded_path
            
        except Exception as e:
            logger.error(f"❌ Failed to download LatentSync model: {e}")
            logger.info("💡 You can download manually using:")
            logger.info(f"   huggingface-cli download {model_id} --local-dir {download_dir}")
            return None
    
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
        
        # Convert all paths to absolute paths to avoid issues with cwd
        # Store original paths for logging
        video_in_original = video_in
        audio_in_original = audio_in
        video_out_original = video_out
        
        video_path = Path(video_in).resolve()
        audio_path = Path(audio_in).resolve()
        video_out_path = Path(video_out).resolve()
        
        # Log path resolution
        if str(video_path) != video_in_original:
            logger.debug(f"📁 Resolved video path: {video_in_original} -> {video_path}")
        if str(audio_path) != audio_in_original:
            logger.debug(f"📁 Resolved audio path: {audio_in_original} -> {audio_path}")
        if str(video_out_path) != video_out_original:
            logger.debug(f"📁 Resolved output path: {video_out_original} -> {video_out_path}")
        
        # Validate inputs exist
        if not video_path.exists():
            error_msg = f"Input video not found: {video_in_original}"
            logger.error(f"❌ {error_msg}")
            logger.error(f"   Resolved path: {video_path}")
            result["error"] = error_msg
            return result
        
        if not audio_path.exists():
            error_msg = f"Input audio not found: {audio_in_original}"
            logger.error(f"❌ {error_msg}")
            logger.error(f"   Resolved path: {audio_path}")
            result["error"] = error_msg
            return result
        
        # Ensure output directory exists
        video_out_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"📁 Ensured output directory exists: {video_out_path.parent}")
        
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
        
        # Face alignment preprocessing (if enabled)
        preprocessed_video_path = str(video_path)  # Default to original
        bbox_track = None
        alignment_result = None
        
        if self.use_face_alignment and self.face_aligner:
            logger.info("🔍 Running face alignment preprocessing...")
            
            # Generate random frame indices for debug (5 frames)
            try:
                import cv2
                cap = cv2.VideoCapture(str(video_path))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                debug_frames = sorted(random.sample(range(total_frames), min(5, total_frames))) if total_frames > 5 else list(range(total_frames))
            except:
                debug_frames = []
            
            # Create preprocessed video path
            preprocessed_path = video_out_path.parent / f"{video_out_path.stem}_pre_latentsync.mp4"
            
            # Run face alignment
            alignment_result = self.face_aligner.process_video(
                video_path=str(video_path),
                output_path=str(preprocessed_path),
                save_debug=True,
                debug_frames=debug_frames
            )
            
            if alignment_result["success"]:
                # Check if we should proceed (gating logic)
                avg_confidence = alignment_result["avg_confidence"]
                frames_with_face = alignment_result["frames_with_face"]
                total_frames = alignment_result["frames_processed"]
                face_ratio = frames_with_face / total_frames if total_frames > 0 else 0
                
                # Gating: skip if confidence too low or too few faces detected
                if avg_confidence < self.face_min_confidence:
                    error_msg = f"Face detection confidence too low ({avg_confidence:.2f} < {self.face_min_confidence})"
                    logger.warning(f"⚠️ {error_msg} - skipping LatentSync")
                    result["error"] = error_msg
                    result["warnings"].append("Skipped LatentSync due to low face detection confidence")
                    return result
                
                if face_ratio < 0.5:  # Less than 50% of frames have faces
                    error_msg = f"Too few frames with faces ({frames_with_face}/{total_frames} = {face_ratio:.1%})"
                    logger.warning(f"⚠️ {error_msg} - skipping LatentSync")
                    result["error"] = error_msg
                    result["warnings"].append("Skipped LatentSync due to insufficient face detection")
                    return result
                
                # Use preprocessed video
                preprocessed_video_path = str(preprocessed_path)
                bbox_track = alignment_result["bbox_track"]
                logger.info(f"✅ Face alignment complete: {frames_with_face}/{total_frames} frames with faces (avg conf: {avg_confidence:.2f})")
            else:
                error_msg = f"Face alignment failed: {alignment_result.get('error', 'Unknown error')}"
                logger.warning(f"⚠️ {error_msg} - using original video")
                result["warnings"].append(error_msg)
                # Continue with original video
        else:
            # No face alignment - check for face in video (if auto mode)
            if face_mode == "auto":
                has_face = self._detect_face_in_video(str(video_path))
                if not has_face:
                    warning = "No face detected in video - lip sync may not work correctly"
                    logger.warning(f"⚠️ {warning}")
                    result["warnings"].append(warning)
                    # Continue anyway - let LatentSync handle it
        
        # Output directory already created above, continue with LatentSync processing
        
        # Run LatentSync
        try:
            logger.info(f"🎬 Running LatentSync: {video_in_original} + {audio_in_original} -> {video_out_original}")
            logger.info(f"   Face mode: {face_mode}, FPS: {result['fps']}")
            
            # Call LatentSync implementation with absolute paths
            # Use preprocessed video if face alignment was used
            latentsync_input_video = preprocessed_video_path if preprocessed_video_path != str(video_path) else str(video_path)
            
            # Create temporary output for LatentSync (will be composited back if needed)
            if self.composite_back and bbox_track:
                latentsync_output = video_out_path.parent / f"{video_out_path.stem}_latentsync_raw.mp4"
            else:
                latentsync_output = video_out_path
            
            success = self._run_latentsync(
                video_in=latentsync_input_video,
                audio_in=str(audio_path),
                video_out=str(latentsync_output),
                face_mode=face_mode,
                target_fps=result["fps"],
                character_reference=character_reference_image
            )
            
            # Composite back to original video if face alignment was used
            if success and self.composite_back and bbox_track and latentsync_output != video_out_path:
                logger.info("🖼️ Compositing lip-synced face back to original video...")
                composite_success = self._composite_face_back(
                    original_video=str(video_path),
                    synced_face_video=str(latentsync_output),
                    output_video=str(video_out_path),
                    bbox_track=bbox_track
                )
                if composite_success:
                    logger.info("✅ Compositing complete")
                else:
                    logger.warning("⚠️ Compositing failed, using LatentSync output directly")
                    # Copy LatentSync output to final output
                    import shutil
                    shutil.copy2(latentsync_output, video_out_path)
            
            if not success:
                error_msg = "LatentSync processing failed"
                logger.error(f"❌ {error_msg}")
                result["error"] = error_msg
                return result
            
            # Verify output (video_out is already absolute path)
            video_out_abs_path = Path(video_out)
            if not video_out_abs_path.exists() or video_out_abs_path.stat().st_size == 0:
                error_msg = "LatentSync output file is missing or empty"
                logger.error(f"❌ {error_msg}")
                logger.error(f"   Expected path: {video_out_abs_path}")
                result["error"] = error_msg
                return result
            
            # Get output video info
            try:
                duration_out, fps_out = self._get_video_info(str(video_out_abs_path))
                result["duration_out"] = duration_out
                result["fps"] = fps_out
                
                logger.info(f"✅ LatentSync completed: {duration_out:.2f}s @ {fps_out}fps")
            except Exception as e:
                logger.warning(f"⚠️ Could not get output video info: {e}")
                result["warnings"].append(f"Could not read output video metadata: {e}")
            
            result["success"] = True
            result["video_path"] = str(video_out_abs_path)  # Return absolute path
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
        
        Tries multiple methods to run LatentSync:
        1. Python package import (latentsync)
        2. Repository inference.py script
        3. Direct Python module execution
        """
        model_dir = Path(self.model_path)
        
        # Option 1: Try Python package import (if LatentSync is installed as package)
        try:
            import latentsync
            # If we can import it, try to use it
            # This depends on the actual LatentSync API structure
            logger.info("📦 Found LatentSync Python package")
            # TODO: Implement actual API calls once LatentSync package structure is known
            # For now, continue to other options
        except ImportError:
            pass
        
        # Option 2: Try repository inference script (check multiple common names and locations)
        inference_scripts = [
            model_dir / "inference.py",
            model_dir / "infer.py",
            model_dir / "run_inference.py",
            model_dir / "scripts" / "inference.py",
            model_dir / "scripts" / "infer.py",
            model_dir / "scripts" / "interface.py",  # User mentioned this location
            model_dir / "scripts" / "run_inference.py",
        ]
        
        # Use LATENTSYNC_DIR as the repository root
        latentsync_dir = self.latentsync_dir
        if not latentsync_dir.exists():
            logger.warning(f"⚠️ LATENTSYNC_DIR does not exist: {latentsync_dir}")
            logger.warning(f"   Set LATENTSYNC_DIR environment variable to the LatentSync repository root")
            return False
        
        # Check if scripts/inference.py exists (for module-style invocation)
        inference_script_path = latentsync_dir / "scripts" / "inference.py"
        script_module_name = "inference"
        if not inference_script_path.exists():
            # Fallback: check other common script names
            for script_name in ["infer.py", "interface.py", "run_inference.py"]:
                alt_script = latentsync_dir / "scripts" / script_name
                if alt_script.exists():
                    inference_script_path = alt_script
                    script_module_name = script_name.replace(".py", "")
                    break
        
        if not inference_script_path.exists():
            logger.error(f"❌ LatentSync inference script not found in: {latentsync_dir / 'scripts'}")
            logger.error(f"   Expected: {latentsync_dir / 'scripts' / 'inference.py'}")
            return False
        
        if inference_script_path.exists():
            try:
                logger.info(f"📝 Found LatentSync script at: {inference_script_path}")
                
                # Use custom Python executable if specified, otherwise use sys.executable
                python_exe = self.python_executable
                if python_exe != sys.executable:
                    logger.info(f"🐍 Using custom Python executable: {python_exe}")
                
                # Build command using module-style invocation: python -m scripts.inference
                # This ensures proper module resolution
                # Use absolute paths to avoid issues with cwd=/workspace/LatentSync
                # video_in, audio_in, video_out are already absolute paths from run() method
                script_module = f"scripts.{script_module_name}"
                video_in_abs = video_in  # Already absolute
                audio_in_abs = audio_in  # Already absolute
                video_out_abs = video_out  # Already absolute
                
                cmd = [
                    python_exe,
                    "-m",
                    script_module,
                    "--video_path", video_in_abs,
                    "--audio_path", audio_in_abs,
                    "--video_out_path", video_out_abs,
                ]
                
                logger.info(f"📹 Input video (absolute): {video_in_abs}")
                logger.info(f"🔊 Input audio (absolute): {audio_in_abs}")
                logger.info(f"📹 Output video (absolute): {video_out_abs}")
                
                # Add inference checkpoint path (model path)
                # Check for model checkpoint in HF cache or repository
                inference_ckpt_path = None
                if "huggingface" in str(model_dir) or "cache" in str(model_dir):
                    # Model is in HF cache, look for checkpoint file
                    cache_model_dir = Path(model_dir)
                    # Common checkpoint file names
                    ckpt_names = [
                        "latentsync_unet.pt",
                        "unet.pt",
                        "checkpoint.pt",
                        "model.pt",
                    ]
                    for ckpt_name in ckpt_names:
                        ckpt_path = cache_model_dir / ckpt_name
                        if ckpt_path.exists():
                            inference_ckpt_path = str(ckpt_path)
                            break
                    
                    # If not found, check subdirectories
                    if not inference_ckpt_path:
                        for subdir in cache_model_dir.iterdir():
                            if subdir.is_dir():
                                for ckpt_name in ckpt_names:
                                    ckpt_path = subdir / ckpt_name
                                    if ckpt_path.exists():
                                        inference_ckpt_path = str(ckpt_path)
                                        break
                                if inference_ckpt_path:
                                    break
                    
                    # If still not found, use the model directory (script might handle it)
                    if not inference_ckpt_path:
                        inference_ckpt_path = str(cache_model_dir)
                else:
                    # Check repository for checkpoint
                    repo_ckpt_paths = [
                        latentsync_dir / "checkpoints" / "latentsync_unet.pt",
                        latentsync_dir / "latentsync_unet.pt",
                        latentsync_dir / "checkpoints" / "unet.pt",
                    ]
                    for ckpt_path in repo_ckpt_paths:
                        if ckpt_path.exists():
                            inference_ckpt_path = str(ckpt_path)
                            break
                
                if inference_ckpt_path:
                    cmd.extend(["--inference_ckpt_path", inference_ckpt_path])
                    logger.info(f"📦 Using checkpoint: {inference_ckpt_path}")
                else:
                    logger.warning("⚠️ Could not find inference checkpoint, script may fail")
                
                # Add quality parameters (inference_steps and guidance_scale)
                # These are critical for output quality according to LatentSync docs
                cmd.extend(["--inference_steps", str(self.inference_steps)])
                cmd.extend(["--guidance_scale", str(self.guidance_scale)])
                logger.info(f"⚙️ Quality settings: inference_steps={self.inference_steps}, guidance_scale={self.guidance_scale}")
                
                # Add optional parameters if script supports them
                if self.fp16:
                    # Check if script supports --fp16
                    cmd.append("--fp16")
                
                # Device parameter
                if self.device:
                    cmd.extend(["--device", self.device])
                
                # Character reference image (if provided)
                if character_reference and Path(character_reference).exists():
                    # LatentSync might support --reference or --reference_image
                    cmd.extend(["--reference", str(Path(character_reference).resolve())])
                    logger.info(f"🖼️ Using character reference image: {character_reference}")
                
                # Use LATENTSYNC_DIR as working directory
                # This ensures relative paths in the script (like configs/unet.yaml) work correctly
                working_dir = latentsync_dir
                
                logger.info(f"🔧 Running LatentSync: {' '.join(cmd)}")
                logger.info(f"📁 Working directory: {working_dir}")
                logger.info(f"📁 LatentSync repository: {latentsync_dir}")
                
                # Create log file for LatentSync output (use absolute path)
                video_out_abs_path = Path(video_out)
                log_file = video_out_abs_path.parent / f"{video_out_abs_path.stem}_latentsync.log"
                
                logger.info(f"📝 Saving LatentSync logs to: {log_file}")
                
                # Prepare environment variables for subprocess
                # Add LatentSync repository to PYTHONPATH so imports work
                env = os.environ.copy()
                
                # Use latentsync_dir for PYTHONPATH
                pythonpath = str(latentsync_dir)
                # Add to existing PYTHONPATH if it exists
                if "PYTHONPATH" in env:
                    env["PYTHONPATH"] = f"{pythonpath}:{env['PYTHONPATH']}"
                else:
                    env["PYTHONPATH"] = pythonpath
                logger.info(f"🐍 Setting PYTHONPATH to include: {latentsync_dir}")
                
                # Build UNet config path: <LATENTSYNC_DIR>/<LATENTSYNC_UNET_CONFIG_REL>
                # Remove any "scripts/" prefix from the relative path
                config_rel = self.unet_config_rel
                if config_rel.startswith("scripts/"):
                    config_rel = config_rel.replace("scripts/", "", 1)
                    logger.warning(f"⚠️ Removed 'scripts/' prefix from config path: {self.unet_config_rel} -> {config_rel}")
                
                unet_config_path = latentsync_dir / config_rel
                
                # Validate config file exists
                if not unet_config_path.exists():
                    logger.error(f"❌ UNet config file not found: {unet_config_path}")
                    logger.error(f"   Expected path: {latentsync_dir}/{config_rel}")
                    
                    # List available config files for helpful error message
                    configs_dir = latentsync_dir / "configs"
                    if configs_dir.exists():
                        logger.info(f"📋 Searching for available config files in: {configs_dir}")
                        try:
                            # Walk configs directory and find all .yaml files
                            yaml_files = []
                            for root, dirs, files in os.walk(configs_dir):
                                for file in files:
                                    if file.endswith(('.yaml', '.yml')):
                                        rel_path = Path(root).relative_to(latentsync_dir)
                                        yaml_files.append(str(rel_path / file))
                            
                            if yaml_files:
                                logger.info(f"   Found {len(yaml_files)} YAML config file(s):")
                                # Show top 10 candidates
                                for yaml_file in sorted(yaml_files)[:10]:
                                    logger.info(f"     - {yaml_file}")
                                if len(yaml_files) > 10:
                                    logger.info(f"     ... and {len(yaml_files) - 10} more")
                                
                                # Suggest a likely candidate
                                unet_configs = [f for f in yaml_files if "unet" in f.lower()]
                                if unet_configs:
                                    suggested = unet_configs[0]
                                    logger.info(f"💡 Suggested config: {suggested}")
                                    logger.info(f"   Set LATENTSYNC_UNET_CONFIG_REL={suggested}")
                            else:
                                logger.warning(f"   No YAML files found in {configs_dir}")
                        except Exception as e:
                            logger.warning(f"   Could not list config files: {e}")
                    
                    logger.error(f"❌ Cannot proceed without config file")
                    return False
                
                # Add --unet_config_path parameter
                cmd.extend(["--unet_config_path", str(unet_config_path)])
                logger.info(f"📋 Using UNet config file: {unet_config_path}")
                
                # Run subprocess and capture output
                try:
                    with open(log_file, 'w', encoding='utf-8') as log_f:
                        # Write header
                        log_f.write(f"LatentSync Execution Log\n")
                        log_f.write(f"{'='*60}\n")
                        log_f.write(f"Command: {' '.join(cmd)}\n")
                        log_f.write(f"Working Directory: {working_dir}\n")
                        log_f.write(f"Python Executable: {python_exe}\n")
                        log_f.write(f"LATENTSYNC_DIR: {latentsync_dir}\n")
                        log_f.write(f"PYTHONPATH: {env.get('PYTHONPATH', 'Not set')}\n")
                        log_f.write(f"Input Video (absolute): {video_in_abs}\n")
                        log_f.write(f"Input Audio (absolute): {audio_in_abs}\n")
                        log_f.write(f"Output Video (absolute): {video_out_abs}\n")
                        log_f.write(f"{'='*60}\n\n")
                        log_f.flush()
                        
                        # Run subprocess with real-time output capture and progress display
                        import subprocess as sp
                        process = sp.Popen(
                            cmd,
                            stdout=sp.PIPE,
                            stderr=sp.STDOUT,  # Combine stderr into stdout
                            text=True,
                            cwd=str(working_dir),  # Use LATENTSYNC_DIR as cwd
                            env=env,  # Pass environment with PYTHONPATH
                            bufsize=1,  # Line buffered
                            universal_newlines=True
                        )
                        
                        # Capture output in real-time
                        stdout_lines = []
                        last_progress_line = ""
                        last_progress_time = 0
                        import time
                        import re
                        
                        logger.info("🔄 Starting LatentSync processing...")
                        
                        # Read output line by line
                        while True:
                            output = process.stdout.readline()
                            if output == '' and process.poll() is not None:
                                break
                            if output:
                                line = output.strip()
                                if not line:
                                    continue
                                
                                stdout_lines.append(line)
                                
                                # Write to log file immediately
                                log_f.write(line + '\n')
                                log_f.flush()
                                
                                # Parse and display progress
                                # Look for progress indicators (percentage, progress bars, etc.)
                                progress_indicators = [
                                    '%', '|', 'progress', 'step', '/', 'epoch', 'batch',
                                    'processing', 'generating', 'rendering', 'frame'
                                ]
                                
                                # Check if this line contains progress information
                                is_progress = any(indicator in line.lower() for indicator in progress_indicators)
                                
                                # Filter out verbose/repeated progress lines (throttle to once per second)
                                current_time = time.time()
                                if is_progress and (line != last_progress_line or current_time - last_progress_time > 1.0):
                                    # Extract percentage if available
                                    percent_match = re.search(r'(\d+(?:\.\d+)?)%', line)
                                    if percent_match:
                                        percent = percent_match.group(1)
                                        logger.info(f"⏳ LatentSync progress: {percent}%")
                                    else:
                                        # Extract step information (e.g., "Step 10/100")
                                        step_match = re.search(r'step\s+(\d+)/(\d+)', line, re.IGNORECASE)
                                        if step_match:
                                            current, total = step_match.groups()
                                            percent = int((int(current) / int(total)) * 100)
                                            logger.info(f"⏳ LatentSync progress: Step {current}/{total} ({percent}%)")
                                        else:
                                            # Show progress line but limit frequency and truncate
                                            logger.info(f"⏳ LatentSync: {line[:80]}...")  # Truncate long lines
                                    last_progress_line = line
                                    last_progress_time = current_time
                                
                                # Always log errors and important messages
                                elif any(keyword in line.lower() for keyword in ['error', 'failed', 'exception', 'traceback']):
                                    logger.error(f"❌ LatentSync: {line}")
                                elif any(keyword in line.lower() for keyword in ['warning']):
                                    # Filter out common warnings that are not critical
                                    if 'hf_xet' not in line.lower() and 'xet storage' not in line.lower():
                                        logger.warning(f"⚠️ LatentSync: {line}")
                                elif any(keyword in line.lower() for keyword in ['success', 'complete', 'done', 'finished']):
                                    logger.info(f"✅ LatentSync: {line}")
                                # Log first few lines for context
                                elif len(stdout_lines) <= 5:
                                    logger.debug(f"LatentSync: {line}")
                        
                        # Wait for process to complete and get return code
                        returncode = process.poll()
                        
                        # Get any remaining output
                        remaining_output, _ = process.communicate()
                        if remaining_output:
                            remaining_lines = remaining_output.strip().split('\n')
                            for line in remaining_lines:
                                if line.strip():
                                    stdout_lines.append(line.strip())
                                    log_f.write(line.strip() + '\n')
                        
                        # Write final status to log
                        log_f.write(f"\n{'='*60}\n")
                        log_f.write(f"Exit Code: {returncode}\n")
                        
                        # Combine all output
                        result_stdout = '\n'.join(stdout_lines)
                        
                        # Create a result-like object for compatibility
                        class ProcessResult:
                            def __init__(self, returncode, stdout):
                                self.returncode = returncode
                                self.stdout = stdout
                        
                        result = ProcessResult(returncode, result_stdout)
                    
                    if result.returncode == 0:
                        logger.info("✅ LatentSync inference completed successfully")
                        logger.info(f"📝 Full logs saved to: {log_file}")
                        return True
                    else:
                        logger.error(f"❌ LatentSync inference failed (exit code {result.returncode})")
                        logger.error(f"📝 Check logs for details: {log_file}")
                        # Show last few lines of error
                        if result.stdout:
                            error_lines = [l for l in result.stdout.split('\n') if l.strip()][-10:]
                            if error_lines:
                                logger.error("Last error lines:")
                                for line in error_lines:
                                    logger.error(f"  {line}")
                        return False
                        
                except Exception as e:
                    logger.error(f"❌ Error running LatentSync: {e}")
                    # Try to write error to log file
                    try:
                        with open(log_file, 'a', encoding='utf-8') as log_f:
                            log_f.write(f"\n\nException occurred: {str(e)}\n")
                            import traceback
                            log_f.write(traceback.format_exc())
                    except:
                        pass
                    import traceback
                    logger.error(traceback.format_exc())
                    return False
                    
            except Exception as e:
                logger.error(f"❌ Error running LatentSync inference script: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return False
        
        # Option 3: Try to find and run LatentSync as a Python module
        # Check if model_dir contains a Python package structure
        init_file = model_dir / "__init__.py"
        if init_file.exists() or any((model_dir / f).suffix == '.py' for f in model_dir.iterdir() if f.is_file()):
            try:
                # Try importing the module directly
                import importlib.util
                spec = importlib.util.spec_from_file_location("latentsync_module", model_dir / "inference.py")
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    # This would require knowing the actual API
                    logger.info("📦 Found Python module structure")
                    # TODO: Implement module execution once API is known
            except Exception as e:
                logger.debug(f"Module import attempt failed: {e}")
        
        # Option 4: Fallback - provide helpful error message
        logger.error("❌ LatentSync inference script not found")
        logger.error("📋 To use LatentSync, you need to:")
        logger.error("   1. Clone the LatentSync repository:")
        logger.error("      git clone https://github.com/bytedance/LatentSync.git /workspace/LatentSync")
        logger.error("   2. Set LATENTSYNC_MODEL_PATH:")
        logger.error("      - For repository: LATENTSYNC_MODEL_PATH=/workspace/LatentSync")
        logger.error("      - For HF model: LATENTSYNC_MODEL_PATH=ByteDance/LatentSync-1.6")
        logger.error("   3. Install LatentSync dependencies:")
        logger.error("      cd /workspace/LatentSync && pip install -r requirements.txt")
        logger.error("")
        logger.error(f"💡 Current model path: {self.model_path}")
        logger.error(f"💡 Checked for scripts in: {model_dir}")
        workspace_repo = Path("/workspace/LatentSync")
        if workspace_repo.exists():
            logger.error(f"💡 Found repository at: {workspace_repo}")
        else:
            logger.error(f"💡 Repository not found at: {workspace_repo}")
        
        # Don't create placeholder - fail explicitly
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
    
    def _composite_face_back(
        self,
        original_video: str,
        synced_face_video: str,
        output_video: str,
        bbox_track: List[Optional[Tuple[int, int, int, int, float]]]
    ) -> bool:
        """
        Composite the lip-synced face crop back onto the original video.
        
        Args:
            original_video: Path to original WAN video
            synced_face_video: Path to LatentSync output (512x512 face crop)
            output_video: Path to final composited output
            bbox_track: List of bboxes from face alignment (one per frame)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            import cv2
            import numpy as np
            
            cap_orig = cv2.VideoCapture(original_video)
            cap_synced = cv2.VideoCapture(synced_face_video)
            
            if not cap_orig.isOpened() or not cap_synced.isOpened():
                logger.error("❌ Could not open input videos for compositing")
                return False
            
            # Get video properties
            fps = int(cap_orig.get(cv2.CAP_PROP_FPS))
            width = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Setup output video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
            
            if not out.isOpened():
                logger.error(f"❌ Could not create output video: {output_video}")
                return False
            
            frame_idx = 0
            
            while True:
                ret_orig, frame_orig = cap_orig.read()
                ret_synced, frame_synced = cap_synced.read()
                
                if not ret_orig:
                    break
                
                # Get bbox for this frame
                if frame_idx < len(bbox_track) and bbox_track[frame_idx]:
                    bbox = bbox_track[frame_idx]
                    x, y, w, h, _ = bbox
                    
                    # Resize synced face to match original crop size
                    if ret_synced:
                        # Resize synced face from 512x512 to original crop size
                        face_resized = cv2.resize(frame_synced, (w, h), interpolation=cv2.INTER_LINEAR)
                        
                        # Create feathered alpha mask for smooth blending
                        mask = np.ones((h, w), dtype=np.float32)
                        feather_size = min(w, h) // 10  # 10% feather
                        
                        # Create gradient mask
                        for i in range(feather_size):
                            alpha = i / feather_size
                            mask[i, :] *= alpha  # Top
                            mask[-i-1, :] *= alpha  # Bottom
                            mask[:, i] *= alpha  # Left
                            mask[:, -i-1] *= alpha  # Right
                        
                        # Clamp bbox to frame bounds
                        x = max(0, min(x, width - 1))
                        y = max(0, min(y, height - 1))
                        w = min(w, width - x)
                        h = min(h, height - y)
                        
                        # Adjust face_resized if needed
                        if face_resized.shape[0] != h or face_resized.shape[1] != w:
                            face_resized = cv2.resize(face_resized, (w, h))
                            mask = cv2.resize(mask, (w, h))
                        
                        # Composite face back onto original frame
                        roi = frame_orig[y:y+h, x:x+w].astype(np.float32)
                        face_float = face_resized.astype(np.float32)
                        mask_3d = np.stack([mask] * 3, axis=2)
                        
                        # Blend: face * mask + original * (1 - mask)
                        blended = face_float * mask_3d + roi * (1 - mask_3d)
                        frame_orig[y:y+h, x:x+w] = blended.astype(np.uint8)
                
                out.write(frame_orig)
                frame_idx += 1
            
            cap_orig.release()
            cap_synced.release()
            out.release()
            
            logger.info(f"✅ Composited {frame_idx} frames")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error compositing face back: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False


def get_latentsync_runner() -> Optional[LatentSyncRunner]:
    """
    Factory function to create LatentSyncRunner from environment/config.
    
    Supports both local paths and Hugging Face model IDs.
    Models will be automatically downloaded from Hugging Face on first use.
    
    Returns:
        LatentSyncRunner instance if enabled and available, None otherwise
    """
    enabled = os.getenv("LATENTSYNC_ENABLED", "false").lower() == "true"
    
    if not enabled:
        logger.info("ℹ️ LatentSync disabled (LATENTSYNC_ENABLED=false)")
        return None
    
    model_path = os.getenv("LATENTSYNC_MODEL_PATH", "")
    if not model_path:
        # Try default Hugging Face model ID
        logger.info("💡 LATENTSYNC_MODEL_PATH not set, using default: ByteDance/LatentSync-1.6")
        model_path = "ByteDance/LatentSync-1.6"
    
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

