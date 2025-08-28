#!/usr/bin/env python3
"""
SVD (Stable Video Diffusion) Animator Module - Optimized for Performance
Handles motion animation using SVD models directly without ComfyUI
"""

import os
import logging
import subprocess
import torch
import time
from pathlib import Path
from typing import Optional, List
import json
import tempfile
import shutil
import numpy as np
from PIL import Image
import requests
import gc

logger = logging.getLogger(__name__)

class SVDAnimator:
    """Handles SVD-based motion animation for images using models directly with performance optimizations."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        self.device = device
        self.model_path = model_path or self._get_default_model_path()
        self.fps = 10  # Reduced from 15 to 10 for slower playback
        self.num_frames = 25  # Default SVD frame count
        self.model = None
        self.pipeline = None
        
        # Performance optimization settings
        self.enable_memory_efficient_attention = True
        self.enable_xformers = True
        self.use_fp16 = True
        self.enable_model_cpu_offload = False  # Set to True if you have memory issues
        
        # Load model with optimizations
        self._load_model()
        
    def _get_default_model_path(self) -> str:
        """Get the default SVD model path."""
        # Check for common SVD model locations
        possible_paths = [
            "models/svd.safetensors",
            "models/svd_xt.safetensors", 
            "models/svd_xt_1_1.safetensors",
            "models/stable-video-diffusion-img2vid-xt.safetensors"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return path
                
        # Return a default path for download
        return "models/svd_xt_1_1.safetensors"
    
    def _load_model(self):
        """Load SVD model with performance optimizations."""
        try:
            from diffusers import StableVideoDiffusionPipeline
            from diffusers.utils import load_image
            
            logger.info(f"🔄 Loading SVD model with optimizations: {self.model_path}")
            
            # Check if model file exists
            if not Path(self.model_path).exists():
                logger.warning(f"⚠️ SVD model not found at {self.model_path}")
                logger.info("📥 Attempting to download SVD model...")
                self._download_svd_model()
            
            # Performance optimization: Clear GPU cache before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Load the pipeline with optimizations
            logger.info("🎯 Loading SVD pipeline with performance optimizations...")
            
            # Use fp16 for better performance on modern GPUs
            torch_dtype = torch.float16 if self.use_fp16 else torch.float32
            
            self.pipeline = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt",
                torch_dtype=torch_dtype,
                variant="fp16" if self.use_fp16 else None
            )
            
            # Apply performance optimizations
            if torch.cuda.is_available():
                self.pipeline = self.pipeline.to("cuda")
                
                # Enable memory efficient attention if available
                if self.enable_memory_efficient_attention:
                    try:
                        self.pipeline.enable_attention_slicing()
                        logger.info("✅ Enabled attention slicing for memory efficiency")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not enable attention slicing: {e}")
                
                # Enable xformers if available (significant speed boost)
                if self.enable_xformers:
                    try:
                        self.pipeline.enable_xformers_memory_efficient_attention()
                        logger.info("✅ Enabled xformers memory efficient attention")
                    except ImportError:
                        logger.warning("⚠️ xformers not available - install with: pip install xformers")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not enable xformers: {e}")
                
                # Model CPU offload for memory-constrained systems
                if self.enable_model_cpu_offload:
                    try:
                        self.pipeline.enable_model_cpu_offload()
                        logger.info("✅ Enabled model CPU offload")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not enable model CPU offload: {e}")
                
                # Compile model for additional speed boost (PyTorch 2.0+)
                try:
                    if hasattr(torch, 'compile'):
                        self.pipeline.unet = torch.compile(self.pipeline.unet, mode="reduce-overhead")
                        logger.info("✅ Compiled UNet for performance boost")
                except Exception as e:
                    logger.warning(f"⚠️ Could not compile model: {e}")
            
            logger.info("✅ SVD model loaded successfully with optimizations")
            
        except ImportError as e:
            logger.error(f"❌ diffusers library not available: {e}")
            logger.info("💡 Install with: pip install diffusers transformers accelerate")
            self.pipeline = None
        except Exception as e:
            logger.error(f"❌ Error loading SVD model: {e}")
            self.pipeline = None
    
    def _download_svd_model(self):
        """Download SVD model if not available."""
        try:
            from huggingface_hub import snapshot_download
            
            logger.info("📥 Downloading SVD model from Hugging Face...")
            
            # Create models directory
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            
            # Download the model
            snapshot_download(
                repo_id="stabilityai/stable-video-diffusion-img2vid-xt",
                local_dir="models/svd_xt_1_1",
                local_dir_use_symlinks=False
            )
            
            logger.info("✅ SVD model downloaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error downloading SVD model: {e}")
            raise
    
    def animate_image(self, image_path: str, output_dir: str, num_frames: int = 25,
                     motion_bucket_id: int = 127, fps_id: int = 6, cond_aug: float = 0.02, 
                     seed: Optional[int] = None) -> str:
        """
        Animate an image using SVD with performance optimizations.
        
        Args:
            image_path: Path to source image
            output_dir: Directory for output frames
            num_frames: Number of frames to generate (max 25 for SVD)
            motion_bucket_id: Motion intensity (0-255, higher = more motion)
            fps_id: FPS setting (0-7, higher = faster motion)
            cond_aug: Conditioning augmentation (0.0-1.0)
            seed: Random seed for reproducibility
            
        Returns:
            Path to generated frames directory
        """
        try:
            # Ensure output directory exists
            frames_dir = Path(output_dir)
            frames_dir.mkdir(parents=True, exist_ok=True)
            
            # Validate input
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Input image not found: {image_path}")
            
            # Clamp num_frames to SVD limit
            num_frames = min(num_frames, 25)
            
            logger.info(f"🎬 Starting SVD animation: {num_frames} frames")
            logger.info(f"🎬 Input: {image_path}")
            logger.info(f"🎬 Output: {output_dir}")
            
            # Clear GPU cache before generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Use direct SVD pipeline if available
            if self.pipeline is not None:
                start_time = time.time()
                result = self._create_direct_svd_animation(
                    image_path, output_dir, num_frames,
                    motion_bucket_id, fps_id, cond_aug, seed
                )
                generation_time = time.time() - start_time
                logger.info(f"✅ SVD generation completed in {generation_time:.1f}s ({num_frames/generation_time:.1f} fps)")
                return result
            else:
                logger.warning("⚠️ SVD pipeline not available, falling back to FFmpeg animation")
                return self._create_ffmpeg_fallback(image_path, output_dir, num_frames)
            
        except Exception as e:
            logger.error(f"Error in SVD animation: {e}")
            # Fallback to FFmpeg animation
            return self._create_ffmpeg_fallback(image_path, output_dir, num_frames)
    
    def _create_direct_svd_animation(self, image_path: str, output_dir: str, num_frames: int,
                                   motion_bucket_id: int, fps_id: int, cond_aug: float, 
                                   seed: Optional[int]) -> str:
        """Create SVD animation using optimized direct pipeline."""
        try:
            from diffusers.utils import load_image
            
            logger.info("🎬 Creating optimized SVD animation...")
            logger.info(f"🎬 Parameters: motion_bucket_id={motion_bucket_id}, fps_id={fps_id}, cond_aug={cond_aug}, seed={seed}")
            
            # Load the input image
            image = load_image(image_path)
            
            # Set random seed if provided for reproducibility
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                logger.info(f"🎬 Set random seed: {seed}")
            
            # Performance optimization: Use smaller decode_chunk_size for faster generation
            # A4000 has good memory, so we can use a balanced approach
            decode_chunk_size = 4  # Reduced from 8 for faster generation on A4000
            
            logger.info(f"🎬 Using decode_chunk_size={decode_chunk_size} for optimal A4000 performance")
            
            # Generate video frames with optimized parameters
            with torch.no_grad():  # Disable gradient computation for inference
                video_frames = self.pipeline(
                    image,
                    decode_chunk_size=decode_chunk_size,
                    motion_bucket_id=motion_bucket_id,
                    fps=fps_id + 1,  # Convert fps_id to actual fps
                    noise_aug_strength=cond_aug,
                    num_frames=num_frames
                ).frames[0]
            
            # Save frames efficiently
            frames_dir = Path(output_dir)
            frames_dir.mkdir(parents=True, exist_ok=True)
            
            for i, frame in enumerate(video_frames):
                frame_path = frames_dir / f"frame_{i:04d}.png"
                frame.save(frame_path)
            
            # Clear GPU cache after generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            logger.info(f"✅ Optimized SVD animation completed: {output_dir}")
            return output_dir
            
        except Exception as e:
            logger.error(f"Error in optimized SVD animation: {e}")
            return self._create_ffmpeg_fallback(image_path, output_dir, num_frames)
    
    def _create_ffmpeg_fallback(self, image_path: str, output_dir: str, num_frames: int) -> str:
        """Create fallback animation using FFmpeg when SVD is not available."""
        try:
            logger.info("📹 Creating FFmpeg fallback animation...")
            
            # Create output directory
            frames_dir = Path(output_dir)
            frames_dir.mkdir(parents=True, exist_ok=True)
            
            # Use zoompan effect for smooth animation
            vf = (
                f"scale={768*1.2}:{1024*1.2}:force_original_aspect_ratio=decrease,"
                f"pad={768*1.2}:{1024*1.2}:(ow-iw)/2:(oh-ih)/2,"
                f"zoompan=z=1+on*0.002:d={num_frames}:"
                "x=iw/2-(iw/zoom/2)+sin(on*0.1)*30:"
                "y=ih/2-(ih/zoom/2)+cos(on*0.1)*20:"
                f"s=768x1024"
            )
            
            cmd = [
                'ffmpeg', '-y',
                '-loop', '1',
                '-i', image_path,
                '-vf', vf,
                '-r', str(self.fps),
                '-frames:v', str(num_frames),
                '-f', 'image2',
                f'{output_dir}/frame_%04d.png'
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"✅ FFmpeg fallback animation completed: {output_dir}")
            return output_dir
            
        except Exception as e:
            logger.error(f"Error in FFmpeg fallback: {e}")
            return self._create_static_frames(image_path, output_dir, num_frames)
    
    def _create_static_frames(self, image_path: str, output_dir: str, num_frames: int) -> str:
        """Create static frames as final fallback."""
        try:
            logger.info("🖼️ Creating static frames as fallback...")
            
            frames_dir = Path(output_dir)
            frames_dir.mkdir(parents=True, exist_ok=True)
            
            # Load and resize image
            image = Image.open(image_path)
            image = image.resize((768, 1024), Image.Resampling.LANCZOS)
            
            # Save multiple copies as frames
            for i in range(num_frames):
                frame_path = frames_dir / f"frame_{i:04d}.png"
                image.save(frame_path)
            
            logger.info(f"✅ Static frames created: {output_dir}")
            return output_dir
            
        except Exception as e:
            logger.error(f"Error creating static frames: {e}")
            raise
    
    def get_supported_motion_levels(self) -> List[dict]:
        """Get supported motion levels for SVD animation."""
        return [
            {"id": 0, "name": "Very Low Motion", "description": "Subtle movement"},
            {"id": 63, "name": "Low Motion", "description": "Gentle movement"},
            {"id": 127, "name": "Medium Motion", "description": "Normal movement"},
            {"id": 191, "name": "High Motion", "description": "Active movement"},
            {"id": 255, "name": "Very High Motion", "description": "Intense movement"}
        ]
    
    def get_supported_fps_levels(self) -> List[dict]:
        """Get supported FPS levels for SVD animation."""
        return [
            {"id": 0, "name": "Very Slow", "fps": 6},
            {"id": 1, "name": "Slow", "fps": 8},
            {"id": 2, "name": "Normal", "fps": 10},
            {"id": 3, "name": "Fast", "fps": 12},
            {"id": 4, "name": "Very Fast", "fps": 14},
            {"id": 5, "name": "Ultra Fast", "fps": 16},
            {"id": 6, "name": "Maximum", "fps": 18}
        ]
