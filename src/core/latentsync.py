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
        
        # Also check if model_dir is the Hugging Face cache, look for repo in /workspace/LatentSync
        workspace_repo = Path("/workspace/LatentSync")
        if ("huggingface" in str(model_dir) or "cache" in str(model_dir)) and workspace_repo.exists():
            inference_scripts.extend([
                workspace_repo / "inference.py",
                workspace_repo / "infer.py",
                workspace_repo / "scripts" / "inference.py",
                workspace_repo / "scripts" / "infer.py",
                workspace_repo / "scripts" / "interface.py",
                workspace_repo / "scripts" / "run_inference.py",
            ])
            logger.info(f"💡 Model in cache, checking repository at: {workspace_repo}")
        
        inference_script = None
        script_dir = None
        repo_root = None
        for script_path in inference_scripts:
            if script_path.exists():
                inference_script = script_path
                script_dir = script_path.parent
                # Determine repository root (parent of scripts/ or the directory containing the script)
                if "scripts" in str(script_dir):
                    repo_root = script_dir.parent
                elif workspace_repo.exists() and script_path.is_relative_to(workspace_repo):
                    repo_root = workspace_repo
                else:
                    repo_root = script_dir
                break
        
        if inference_script:
            try:
                logger.info(f"📝 Found LatentSync script at: {inference_script}")
                
                # Use custom Python executable if specified, otherwise use sys.executable
                python_exe = self.python_executable
                if python_exe != sys.executable:
                    logger.info(f"🐍 Using custom Python executable: {python_exe}")
                
                # Build command - LatentSync uses specific argument names
                # Based on error message: --video_path, --audio_path, --video_out_path, --inference_ckpt_path
                cmd = [
                    python_exe,
                    str(inference_script),
                    "--video_path", video_in,
                    "--audio_path", audio_in,
                    "--video_out_path", video_out,
                ]
                
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
                elif latentsync_repo_root:
                    # Check repository for checkpoint
                    repo_ckpt_paths = [
                        latentsync_repo_root / "checkpoints" / "latentsync_unet.pt",
                        latentsync_repo_root / "latentsync_unet.pt",
                        latentsync_repo_root / "checkpoints" / "unet.pt",
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
                
                # Add optional parameters if script supports them
                # Note: LatentSync script may not support all these, but we'll try
                if self.fp16:
                    # Check if script supports --fp16, if not it might be automatic
                    pass  # Some scripts don't have explicit fp16 flag
                
                # Device is typically handled automatically by PyTorch based on CUDA availability
                # The script might not have a --device flag
                
                if character_reference and Path(character_reference).exists():
                    # Check if script supports reference image
                    # This might be --reference_image or similar
                    pass  # Add if script supports it
                
                # Use repository root as working directory (not scripts/ subdirectory)
                # This ensures relative paths in the script (like configs/unet.yaml) work correctly
                working_dir = repo_root if repo_root else script_dir
                
                logger.info(f"🔧 Running LatentSync: {' '.join(cmd)}")
                logger.info(f"📁 Working directory: {working_dir}")
                logger.info(f"📁 Script location: {inference_script}")
                
                # Create log file for LatentSync output
                output_video_path = Path(video_out)
                log_file = output_video_path.parent / f"{output_video_path.stem}_latentsync.log"
                
                logger.info(f"📝 Saving LatentSync logs to: {log_file}")
                
                # Prepare environment variables for subprocess
                # Add LatentSync repository to PYTHONPATH so imports work
                env = os.environ.copy()
                
                # Use repo_root for PYTHONPATH
                latentsync_repo_root = repo_root
                if latentsync_repo_root:
                    pythonpath = str(latentsync_repo_root)
                    # Add to existing PYTHONPATH if it exists
                    if "PYTHONPATH" in env:
                        env["PYTHONPATH"] = f"{pythonpath}:{env['PYTHONPATH']}"
                    else:
                        env["PYTHONPATH"] = pythonpath
                    logger.info(f"🐍 Setting PYTHONPATH to include: {latentsync_repo_root}")
                
                # Check for config file and add if script supports it
                if repo_root:
                    config_paths = [
                        repo_root / "configs" / "unet.yaml",
                        repo_root / "configs" / "unet.yml",
                        repo_root / "config" / "unet.yaml",
                    ]
                    for config_path in config_paths:
                        if config_path.exists():
                            # Add --unet_config_path parameter if script supports it
                            if "--unet_config_path" not in cmd:
                                # Insert before the last argument (output path)
                                cmd.insert(-1, "--unet_config_path")
                                cmd.insert(-1, str(config_path))
                                logger.info(f"📋 Found and added config file: {config_path}")
                            break
                    else:
                        logger.warning(f"⚠️ Config file not found in {repo_root}/configs/ - script may fail")
                
                # Run subprocess and capture output
                try:
                    with open(log_file, 'w', encoding='utf-8') as log_f:
                        # Write header
                        log_f.write(f"LatentSync Execution Log\n")
                        log_f.write(f"{'='*60}\n")
                        log_f.write(f"Command: {' '.join(cmd)}\n")
                        log_f.write(f"Working Directory: {script_dir}\n")
                        log_f.write(f"Python Executable: {python_exe}\n")
                        if latentsync_repo_root:
                            log_f.write(f"PYTHONPATH: {env.get('PYTHONPATH', 'Not set')}\n")
                        log_f.write(f"Input Video: {video_in}\n")
                        log_f.write(f"Input Audio: {audio_in}\n")
                        log_f.write(f"Output Video: {video_out}\n")
                        log_f.write(f"{'='*60}\n\n")
                        log_f.flush()
                        
                        # Run subprocess with real-time logging
                        # Use repo_root as working directory so relative paths work
                        result = subprocess.run(
                            cmd,
                            check=False,  # Don't raise exception, handle return code manually
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,  # Combine stderr into stdout
                            text=True,
                            cwd=str(working_dir),  # Use repo root, not scripts directory
                            env=env,  # Pass environment with PYTHONPATH
                            bufsize=1,  # Line buffered
                            universal_newlines=True
                        )
                        
                        # Write output to log file
                        if result.stdout:
                            log_f.write("STDOUT/STDERR:\n")
                            log_f.write(result.stdout)
                            log_f.write(f"\n{'='*60}\n")
                            log_f.write(f"Exit Code: {result.returncode}\n")
                        
                        # Also log to our logger
                        if result.stdout:
                            # Log important lines (filter verbose output)
                            lines = result.stdout.split('\n')
                            for line in lines:
                                line = line.strip()
                                if not line:
                                    continue
                                # Log errors and important messages
                                if any(keyword in line.lower() for keyword in ['error', 'failed', 'exception', 'traceback', 'warning']):
                                    logger.warning(f"LatentSync: {line}")
                                elif any(keyword in line.lower() for keyword in ['success', 'complete', 'done', 'finished']):
                                    logger.info(f"LatentSync: {line}")
                                # Log first few lines and last few lines for context
                                elif len(lines) < 20 or lines.index(line) < 5 or lines.index(line) >= len(lines) - 5:
                                    logger.debug(f"LatentSync: {line}")
                    
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

