#!/usr/bin/env python3
"""
WAN 2.1 Text-to-Video Generator Module
Handles text-to-video generation using Wan-AI/Wan2.1-T2V-14B-Diffusers
"""

import os
import logging

# Set CUDA allocator config early (before torch import if possible, but setting here still helps)
# This helps with memory fragmentation, especially for large models like 14B
# Use new PYTORCH_ALLOC_CONF (PYTORCH_CUDA_ALLOC_CONF is deprecated)
if 'PYTORCH_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import torch
from pathlib import Path
from typing import Optional, Dict, Any, Union
import gc
import subprocess
import tempfile
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Global singleton instance
_wan_pipeline = None
_wan_vae = None


def round_frames_to_valid_count(num_frames: int) -> int:
    """
    Round num_frames to satisfy WAN requirement: (num_frames - 1) must be divisible by 4.
    Valid counts: 49, 53, 57, 61, 65, 69, 73, 77, etc. (1 mod 4)
    
    Args:
        num_frames: Original frame count
        
    Returns:
        Rounded frame count that satisfies (num_frames - 1) % 4 == 0
    """
    if num_frames <= 0:
        return 49  # Minimum valid count
    
    # (num_frames - 1) must be divisible by 4
    # So num_frames must be 1 mod 4 (i.e., num_frames % 4 == 1)
    remainder = num_frames % 4
    
    if remainder == 1:
        # Already valid
        return num_frames
    elif remainder == 0:
        # num_frames % 4 == 0, so (num_frames - 1) % 4 == 3, need to add 1
        return num_frames + 1
    elif remainder == 2:
        # num_frames % 4 == 2, so (num_frames - 1) % 4 == 1, need to subtract 1
        return num_frames - 1
    else:  # remainder == 3
        # num_frames % 4 == 3, so (num_frames - 1) % 4 == 2, need to add 2
        return num_frames + 2


def get_cache_dir() -> Optional[str]:
    """
    Get the appropriate cache directory for Hugging Face models.
    Prefers /workspace if available (for RunPod and similar environments with attached disks).
    
    Returns:
        Cache directory path as string, or None to use default
    """
    workspace_cache = Path("/workspace/.cache/huggingface")
    if Path("/workspace").exists():
        workspace_cache.mkdir(parents=True, exist_ok=True)
        # Set HF_HOME environment variable as well
        os.environ["HF_HOME"] = str(workspace_cache)
        return str(workspace_cache)
    return None


def get_wan_pipeline(device: str = None, force_reload: bool = False):
    """
    Get or initialize the global WAN pipeline (singleton pattern).
    
    Args:
        device: Device to run on ('cuda' or 'cpu'). Auto-detected if None.
        force_reload: Force reload of the pipeline even if already loaded.
        
    Returns:
        WanPipeline instance or None if loading fails
    """
    global _wan_pipeline, _wan_vae
    
    if _wan_pipeline is not None and not force_reload:
        logger.info("♻️ Reusing existing WAN pipeline (singleton) - model already loaded")
        return _wan_pipeline
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        from diffusers import AutoencoderKLWan, WanPipeline
        
        model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
        
        # Configure Hugging Face cache directory to use /workspace if available
        # This is important for RunPod and similar environments with attached disks
        cache_dir = get_cache_dir()
        if cache_dir:
            logger.info(f"📁 Using Hugging Face cache directory: {cache_dir}")
        else:
            # Use default cache
            default_cache = Path.home() / ".cache" / "huggingface"
            logger.info(f"📁 Using default Hugging Face cache: {default_cache}")
        
        logger.info(f"🔄 Loading WAN 2.1 T2V model: {model_id}")
        logger.info(f"💻 Device: {device}")
        
        # Determine torch dtype based on device
        if device == 'cuda' and torch.cuda.is_available():
            # Use bfloat16 on CUDA for better performance and memory efficiency
            torch_dtype = torch.bfloat16
            vae_dtype = torch.float32  # VAE typically uses float32
            logger.info("✅ Using bfloat16 on CUDA for optimal performance")
        else:
            torch_dtype = torch.float32
            vae_dtype = torch.float32
            if device == 'cpu':
                logger.warning("⚠️ Running on CPU - this will be very slow. Consider using GPU.")
        
        # Clear GPU cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Load VAE
        logger.info("📦 Loading WAN VAE...")
        _wan_vae = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=vae_dtype,
            cache_dir=cache_dir
        )
        
        # Load pipeline
        logger.info("📦 Loading WAN pipeline...")
        _wan_pipeline = WanPipeline.from_pretrained(
            model_id,
            vae=_wan_vae,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir
        )
        
        # Move to device
        _wan_pipeline = _wan_pipeline.to(device)
        
        # Enable memory optimizations if available
        if device == 'cuda':
            # Enable VAE slicing and tiling for memory efficiency during decode
            try:
                if hasattr(_wan_pipeline, 'enable_vae_slicing'):
                    _wan_pipeline.enable_vae_slicing()
                    logger.info("✅ Enabled VAE slicing for memory efficiency")
            except Exception as e:
                logger.warning(f"⚠️ Could not enable VAE slicing: {e}")
            
            try:
                if hasattr(_wan_pipeline, 'enable_vae_tiling'):
                    _wan_pipeline.enable_vae_tiling()
                    logger.info("✅ Enabled VAE tiling for memory efficiency")
            except Exception as e:
                logger.warning(f"⚠️ Could not enable VAE tiling: {e}")
            
            try:
                if hasattr(_wan_pipeline, 'enable_memory_efficient_attention'):
                    _wan_pipeline.enable_memory_efficient_attention()
                    logger.info("✅ Enabled memory efficient attention")
            except Exception as e:
                logger.warning(f"⚠️ Could not enable memory efficient attention: {e}")
            
            try:
                if hasattr(_wan_pipeline, 'enable_xformers_memory_efficient_attention'):
                    _wan_pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("✅ Enabled xformers memory efficient attention")
            except Exception as e:
                logger.info("ℹ️ xFormers not available; continuing without it")
            
            # Configure CUDA allocator for expandable segments (helps with fragmentation)
            # Use new PYTORCH_ALLOC_CONF (PYTORCH_CUDA_ALLOC_CONF is deprecated)
            if 'PYTORCH_ALLOC_CONF' not in os.environ:
                os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
                logger.info("✅ Set PYTORCH_ALLOC_CONF=expandable_segments:True for better memory management")
            else:
                logger.info(f"ℹ️ PYTORCH_ALLOC_CONF already set: {os.environ.get('PYTORCH_ALLOC_CONF')}")
        
        logger.info("✅ WAN 2.1 T2V pipeline loaded successfully")
        logger.info("📦 WAN pipeline initialized ONCE - will be reused for all subsequent generations")
        return _wan_pipeline
        
    except ImportError as e:
        logger.error(f"❌ diffusers library not available or WAN not supported: {e}")
        logger.info("💡 Install with: pip install diffusers>=0.34.0 transformers>=4.45.0")
        return None
    except Exception as e:
        logger.error(f"❌ Error loading WAN model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


class WanT2VGenerator:
    """Handles text-to-video generation using WAN 2.1."""
    
    def __init__(self, 
                 width: int = 832,
                 height: int = 480,
                 num_frames: int = 49,
                 fps: int = 12,
                 num_inference_steps: int = 30,
                 guidance_scale: float = 6.0,
                 negative_prompt: str = "text, subtitles, watermark, blurry, low quality, cartoon, anime, manga, illustration, painting, drawing, sketch, bad anatomy, distorted, deformed, ugly",
                 device: str = None):
        """
        Initialize the WAN T2V generator.
        
        Args:
            width: Video width (default: 832)
            height: Video height (default: 480)
            num_frames: Number of frames to generate (default: 49)
            fps: Frames per second for output video (default: 12)
            num_inference_steps: Number of denoising steps (default: 30)
            guidance_scale: Guidance scale for prompt adherence (default: 6.0)
            negative_prompt: Negative prompt (default excludes cartoon/anime/illustration for realistic videos)
            device: Device to run on ('cuda' or 'cpu'). Auto-detected if None.
        """
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.fps = fps
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.negative_prompt = negative_prompt
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.pipeline = None
        
        # Load pipeline (singleton, shared across instances)
        self.pipeline = get_wan_pipeline(device=self.device)
        if self.pipeline is None:
            logger.warning("⚠️ WAN pipeline not available")
    
    def generate_video(self, 
                      prompt: str, 
                      output_path: str,
                      seed: Optional[int] = None,
                      negative_prompt: Optional[str] = None,
                      duration: Optional[float] = None,
                      scene_id: Optional[str] = None,
                      visual_reference: Optional[str] = None,
                      slug: Optional[str] = None,
                      best_frame_filename: Optional[str] = None) -> Union[str, Dict[str, Any]]:
        """
        Generate a video from a text prompt.
        
        Args:
            prompt: Text prompt describing the video
            output_path: Path to save the output MP4 file
            seed: Random seed for reproducibility (optional)
            negative_prompt: Override default negative prompt (optional)
            duration: Target duration in seconds. If provided, num_frames will be calculated from this.
                      If None, uses the default num_frames from initialization.
            scene_id: Scene identifier for frame extraction (optional)
            visual_reference: Visual reference description from storyboard (optional)
            slug: Slug for best frame filename (optional, used if best_frame_filename not provided)
            best_frame_filename: Explicit filename for best frame (optional, takes precedence over slug/scene_id)
            
        Returns:
            Dictionary with:
            - video_path: Path to the generated video file
            - best_frame_path: Path to the best frame PNG (if scene_id provided)
            - scene_id: Scene identifier
            - visual_reference: Visual reference from storyboard
            If scene_id not provided, returns just video_path as string (backward compatible)
        """
        if self.pipeline is None:
            raise RuntimeError("WAN pipeline not available. Cannot generate video.")
        
        # Calculate num_frames from duration if provided
        num_frames_to_use = self.num_frames
        if duration is not None and duration > 0:
            # Calculate frames needed: duration * fps, rounded up to ensure we cover the full duration
            calculated_frames = int(duration * self.fps) + 1
            num_frames_to_use = calculated_frames
            logger.info(f"📏 Target duration: {duration:.2f}s")
            logger.info(f"🎞️ Calculated frames: {calculated_frames} @ {self.fps}fps (~{calculated_frames/self.fps:.2f}s)")
        else:
            logger.info(f"🎞️ Using default frames: {num_frames_to_use} @ {self.fps}fps (~{num_frames_to_use/self.fps:.1f}s)")
        
        # Round frames to satisfy WAN requirement: (num_frames - 1) must be divisible by 4
        original_frames = num_frames_to_use
        num_frames_to_use = round_frames_to_valid_count(num_frames_to_use)
        if num_frames_to_use != original_frames:
            logger.info(f"🔄 Rounded frames from {original_frames} to {num_frames_to_use} (WAN requires (num_frames-1) divisible by 4)")
        else:
            logger.info(f"✅ Using {num_frames_to_use} frames (satisfies WAN requirement: (num_frames-1) % 4 == 0)")
        
        try:
            logger.info(f"🎬 Generating video with WAN 2.1 T2V...")
            logger.info(f"📝 Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
            logger.info(f"📐 Dimensions: {self.width}x{self.height}")
            logger.info(f"⚙️ Steps: {self.num_inference_steps}, Guidance: {self.guidance_scale}")
            
            # Use provided negative prompt or default
            neg_prompt = negative_prompt or self.negative_prompt
            if neg_prompt:
                logger.info(f"🚫 Negative prompt: {neg_prompt[:100]}{'...' if len(neg_prompt) > 100 else ''}")
            
            # Set seed if provided
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                logger.info(f"🎲 Using seed: {seed}")
            
            # Clear GPU cache before generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Generate video - request latents to avoid VAE decode on GPU
            logger.info("🎬 Running inference...")
            try:
                # Try to get latents output (avoids VAE decode on GPU)
                output = self.pipeline(
                    prompt=prompt,
                    negative_prompt=neg_prompt,
                    height=self.height,
                    width=self.width,
                    num_frames=num_frames_to_use,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    output_type="latent",
                    return_dict=True
                )
                has_latents = True
                logger.info("✅ Pipeline returned latents (will decode on CPU)")
            except (TypeError, ValueError) as e:
                # Fallback: pipeline doesn't support output_type="latent"
                logger.warning(f"⚠️ Pipeline doesn't support output_type='latent', using default output: {e}")
                output = self.pipeline(
                    prompt=prompt,
                    negative_prompt=neg_prompt,
                    height=self.height,
                    width=self.width,
                    num_frames=num_frames_to_use,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale
                )
                has_latents = False
            
            # Clear GPU memory after inference, before VAE decode
            logger.info("🧹 Clearing GPU cache before VAE decode...")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gc.collect()
            
            # Handle VAE decode based on output type
            if has_latents and hasattr(output, 'latents') and output.latents is not None:
                try:
                    # Decode VAE on CPU to avoid OOM
                    logger.info("🔄 Decoding VAE on CPU to prevent GPU OOM...")
                    latents = output.latents
                    
                    # Ensure latents is a tensor
                    if not isinstance(latents, torch.Tensor):
                        if isinstance(latents, (list, tuple)) and len(latents) > 0:
                            latents = latents[0]
                        else:
                            raise ValueError("Unexpected latents format")
                    
                    # Move latents to CPU
                    if latents.is_cuda:
                        latents = latents.cpu()
                    
                    # Get VAE from pipeline
                    vae = self.pipeline.vae
                    
                    # Decode on CPU with no_grad to save memory
                    with torch.no_grad():
                        # Move VAE to CPU temporarily for decode
                        vae_device = next(vae.parameters()).device
                        vae_on_cpu = vae_device.type == 'cpu'
                        
                        if not vae_on_cpu:
                            logger.info("📦 Moving VAE to CPU for decode...")
                            vae = vae.to('cpu')
                        
                        # Decode latents (output shape: [B, C, T, H, W])
                        vae_output = vae.decode(latents)
                        # Handle both BaseOutput and direct tensor returns
                        if hasattr(vae_output, 'sample'):
                            frames_tensor = vae_output.sample
                        else:
                            frames_tensor = vae_output
                        
                        # Move VAE back to original device if needed
                        if not vae_on_cpu:
                            logger.info("📦 Moving VAE back to GPU...")
                            vae = vae.to(vae_device)
                            self.pipeline.vae = vae
                    
                    # Convert tensor to numpy frames format expected by export_to_video
                    # VAE output: [B, C, T, H, W] with values in [-1, 1]
                    # Need: list of [H, W, C] arrays with values in [0, 255] uint8
                    frames_tensor = frames_tensor.cpu()  # Ensure on CPU
                    # Permute from [B, C, T, H, W] to [B, T, H, W, C]
                    frames_tensor = frames_tensor.permute(0, 2, 3, 4, 1)
                    # Convert to numpy and scale from [-1, 1] to [0, 255]
                    frames_np = ((frames_tensor.float() + 1.0) / 2.0 * 255.0).clamp(0, 255).byte().numpy()
                    # Convert to list of frames: [B, T, H, W, C] -> list of [H, W, C]
                    # Use .copy() to ensure each frame is a proper numpy array (not a view)
                    frames = [np.array(frames_np[0, t], dtype=np.uint8).copy() for t in range(frames_np.shape[1])]
                    
                    # Clean up latents
                    del latents
                    del frames_tensor
                    del frames_np
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                    logger.info("✅ VAE decode completed on CPU")
                except Exception as e:
                    logger.warning(f"⚠️ CPU VAE decode failed: {e}, falling back to default output")
                    # Fallback: use frames from output (already decoded)
                    frames = output.frames[0] if hasattr(output, 'frames') else output[0]
                    # Ensure frames is a list of numpy arrays
                    if isinstance(frames, np.ndarray):
                        # If it's a single array with shape [T, H, W, C], convert to list
                        if len(frames.shape) == 4:
                            frames = [np.array(frames[t], dtype=np.uint8).copy() for t in range(frames.shape[0])]
                        else:
                            frames = [frames]
                    elif not isinstance(frames, list):
                        frames = [frames]
            else:
                # Fallback: use frames from output (already decoded)
                logger.info("ℹ️ Using pre-decoded frames from pipeline output")
                frames = output.frames[0] if hasattr(output, 'frames') else output[0]
                # Ensure frames is a list of numpy arrays
                if isinstance(frames, np.ndarray):
                    # If it's a single array with shape [T, H, W, C], convert to list
                    if len(frames.shape) == 4:
                        frames = [np.array(frames[t], dtype=np.uint8).copy() for t in range(frames.shape[0])]
                    else:
                        frames = [frames]
                elif not isinstance(frames, list):
                    frames = [frames]
            
            # Clean up output object
            del output
            output = None
            
            # Export to video
            from diffusers.utils import export_to_video
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Ensure all frames are proper numpy arrays with uint8 dtype
            frames = [np.array(frame, dtype=np.uint8) if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 else frame for frame in frames]
            
            logger.info(f"💾 Saving video to: {output_path}...")
            export_to_video(frames, str(output_path), fps=self.fps)
            
            # Explicitly delete frames to free memory after video export
            del frames
            frames = None
            
            # Clear GPU cache after video export
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Wait for all GPU operations to complete
                torch.cuda.empty_cache()
            gc.collect()  # Force Python garbage collection
            
            logger.info(f"✅ Video generated successfully: {output_path}")
            
            # Extract best frame if scene_id is provided
            result = {"video_path": str(output_path)}
            if scene_id is not None:
                try:
                    best_frame_path = self._extract_best_frame(
                        video_path=str(output_path),
                        scene_id=scene_id,
                        slug=slug,
                        best_frame_filename=best_frame_filename,
                        output_dir=output_path.parent
                    )
                    result.update({
                        "best_frame_path": best_frame_path,
                        "scene_id": scene_id,
                        "visual_reference": visual_reference
                    })
                    logger.info(f"✅ Best frame extracted: {best_frame_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to extract best frame: {e}")
                    # Continue without best frame - video generation succeeded
            
            # Return dict if scene_id provided, otherwise string for backward compatibility
            return result if scene_id is not None else str(output_path)
            
        except Exception as e:
            logger.error(f"❌ Error generating video: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _extract_best_frame(self, 
                           video_path: str,
                           scene_id: str,
                           slug: Optional[str] = None,
                           best_frame_filename: Optional[str] = None,
                           output_dir: Optional[Path] = None,
                           extraction_fps: float = 9.0) -> str:
        """
        Extract frames from video and select the best frame using Laplacian variance.
        
        Args:
            video_path: Path to the video file
            scene_id: Scene identifier
            slug: Slug for filename (optional, used if best_frame_filename not provided)
            best_frame_filename: Explicit filename for best frame (optional, takes precedence)
            output_dir: Directory to save the best frame (optional, uses video parent dir)
            extraction_fps: FPS for frame extraction (6-12 fps range, default 9)
            
        Returns:
            Path to the saved best frame PNG
        """
        video_path_obj = Path(video_path)
        if output_dir is None:
            output_dir = video_path_obj.parent
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create temporary directory for extracted frames
        temp_frames_dir = tempfile.mkdtemp(prefix="wan_frames_")
        temp_frames_dir_path = Path(temp_frames_dir)
        
        try:
            # Extract frames using ffmpeg at specified FPS
            logger.info(f"📸 Extracting frames from video at {extraction_fps} fps...")
            frame_pattern = str(temp_frames_dir_path / "frame_%04d.png")
            
            cmd = [
                'ffmpeg', '-y',
                '-i', str(video_path),
                '-vf', f'fps={extraction_fps}',
                frame_pattern
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Find all extracted frames
            frame_files = sorted(temp_frames_dir_path.glob("frame_*.png"))
            if not frame_files:
                raise ValueError("No frames extracted from video")
            
            logger.info(f"📸 Extracted {len(frame_files)} frames")
            
            # Calculate Laplacian variance for each frame (sharpness metric)
            best_frame_path = None
            best_sharpness = -1
            
            for frame_file in frame_files:
                try:
                    # Read frame as grayscale
                    img = cv2.imread(str(frame_file), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    
                    # Calculate Laplacian variance (sharpness metric)
                    laplacian = cv2.Laplacian(img, cv2.CV_64F)
                    sharpness = laplacian.var()
                    
                    if sharpness > best_sharpness:
                        best_sharpness = sharpness
                        best_frame_path = frame_file
                except Exception as e:
                    logger.warning(f"⚠️ Error processing frame {frame_file}: {e}")
                    continue
            
            if best_frame_path is None:
                raise ValueError("Could not find a valid frame")
            
            logger.info(f"✅ Best frame selected (sharpness: {best_sharpness:.2f})")
            
            # Use best_frame_filename from scene if provided, otherwise generate from slug/scene_id
            if best_frame_filename:
                output_filename = best_frame_filename
            elif slug:
                output_filename = f"{slug}__{scene_id}.png"
            else:
                output_filename = f"{scene_id}.png"
            
            output_frame_path = output_dir / output_filename
            
            # Copy best frame to output location
            import shutil
            shutil.copy2(best_frame_path, output_frame_path)
            logger.info(f"💾 Saved best frame: {output_frame_path}")
            
            return str(output_frame_path)
            
        finally:
            # Clean up temporary frames directory
            try:
                import shutil
                shutil.rmtree(temp_frames_dir, ignore_errors=True)
                logger.debug(f"🧹 Cleaned up temporary frames directory: {temp_frames_dir}")
            except Exception as e:
                logger.warning(f"⚠️ Could not clean up temp directory: {e}")
    
    def is_available(self) -> bool:
        """Check if WAN pipeline is available."""
        return self.pipeline is not None


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test WAN generator
    generator = WanT2VGenerator(
        width=832,
        height=480,
        num_frames=49,
        fps=12
    )
    
    if generator.is_available():
        print("✅ WAN generator initialized successfully")
        
        # Test generation
        test_prompt = "A cat walks on the grass, realistic"
        output_path = generator.generate_video(
            prompt=test_prompt,
            output_path="test_wan_output.mp4"
        )
        print(f"✅ Test video generated: {output_path}")
    else:
        print("❌ WAN generator not available")
