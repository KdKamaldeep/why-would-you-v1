#!/usr/bin/env python3
"""
WAN 2.1 Text-to-Video Generator Module
Handles text-to-video generation using Wan-AI/Wan2.1-T2V-1.3B-Diffusers
"""

import os
import logging
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
        
        model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        
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
                      slug: Optional[str] = None) -> Union[str, Dict[str, Any]]:
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
            slug: Slug for best frame filename (optional)
            
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
            num_frames_to_use = int(duration * self.fps) + 1
            logger.info(f"📏 Target duration: {duration:.2f}s")
            logger.info(f"🎞️ Calculated frames: {num_frames_to_use} @ {self.fps}fps (~{num_frames_to_use/self.fps:.2f}s)")
        else:
            logger.info(f"🎞️ Using default frames: {num_frames_to_use} @ {self.fps}fps (~{num_frames_to_use/self.fps:.1f}s)")
        
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
            
            # Generate video
            logger.info("🎬 Running inference...")
            output = self.pipeline(
                prompt=prompt,
                negative_prompt=neg_prompt,
                height=self.height,
                width=self.width,
                num_frames=num_frames_to_use,  # Use calculated frames based on duration
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale
            )
            
            # Extract frames
            frames = output.frames[0]
            
            # Export to video
            from diffusers.utils import export_to_video
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"💾 Saving video to: {output_path}")
            export_to_video(frames, str(output_path), fps=self.fps)
            
            # Explicitly delete frames and output to free memory
            del frames
            del output
            frames = None
            output = None
            
            # Clear GPU cache after generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for all GPU operations to complete
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
                           output_dir: Optional[Path] = None,
                           extraction_fps: float = 9.0) -> str:
        """
        Extract frames from video and select the best frame using Laplacian variance.
        
        Args:
            video_path: Path to the video file
            scene_id: Scene identifier
            slug: Slug for filename (optional, uses scene_id if not provided)
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
            
            # Generate output filename: <slug>__<scene_id>.png
            if slug:
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
