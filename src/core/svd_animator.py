#!/usr/bin/env python3
"""
SVD (Stable Video Diffusion) Animator Module
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

logger = logging.getLogger(__name__)

class SVDAnimator:
    """Handles SVD-based motion animation for images using models directly."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        self.device = device
        self.model_path = model_path or self._get_default_model_path()
        self.fps = 15
        self.num_frames = 25  # Default SVD frame count
        self.model = None
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
        """Load SVD model directly using diffusers library."""
        try:
            from diffusers import StableVideoDiffusionPipeline
            from diffusers.utils import load_image
            
            logger.info(f"🔄 Loading SVD model: {self.model_path}")
            
            # Check if model file exists
            if not Path(self.model_path).exists():
                logger.warning(f"⚠️ SVD model not found at {self.model_path}")
                logger.info("📥 Attempting to download SVD model...")
                self._download_svd_model()
            
            # Load the pipeline
            self.pipeline = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt",
                torch_dtype=torch.float16,
                variant="fp16"
            )
            
            if torch.cuda.is_available():
                self.pipeline = self.pipeline.to("cuda")
            
            logger.info("✅ SVD model loaded successfully")
            
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
    
    def animate_image(self, image_path: str, output_dir: str, num_frames: int = 24, 
                     motion_bucket_id: int = 127, fps_id: int = 6, 
                     cond_aug: float = 0.02, seed: Optional[int] = None) -> str:
        """
        Animate an image using SVD motion animation directly.
        
        Args:
            image_path: Path to source image
            output_dir: Directory for output frames
            num_frames: Number of frames to generate (SVD HARD LIMIT: 24 frames max for chunking)
            motion_bucket_id: Motion intensity (0-255, higher = more motion)
            fps_id: FPS setting (0-7, higher = faster motion)
            cond_aug: Conditioning augmentation (0.0-1.0)
            seed: Random seed for reproducibility
            
        Returns:
            Path to generated frames directory
            
        Note:
            SVD has a hard limit of 24 frames for chunked generation. For longer sequences, 
            the AnimationGenerator will use overlapping chunk generation for visual consistency.
        """
        try:
            logger.info(f"🎬 SVD Animating image: {image_path}")
            logger.info(f"📊 Target frames: {num_frames} (motion_bucket_id: {motion_bucket_id}, fps_id: {fps_id})")
            
            # Enforce SVD frame limit for chunking
            if num_frames > 24:
                logger.warning(f"⚠️ SVD frame limit exceeded: {num_frames} > 24. Clamping to 24 frames for chunking.")
                num_frames = 24
            
            # Create output directory
            frames_dir = Path(output_dir)
            frames_dir.mkdir(parents=True, exist_ok=True)
            
            # Use direct SVD pipeline for animation
            if self.pipeline is not None:
                return self._create_direct_svd_animation(
                    image_path, str(frames_dir), num_frames, 
                    motion_bucket_id, fps_id, cond_aug, seed
                )
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
        """Create SVD animation using direct pipeline without ComfyUI."""
        try:
            from diffusers.utils import load_image
            
            logger.info("🎬 Creating SVD animation using direct pipeline...")
            logger.info(f"🎬 Parameters: motion_bucket_id={motion_bucket_id}, fps_id={fps_id}, cond_aug={cond_aug}, seed={seed}")
            
            # Load the input image
            image = load_image(image_path)
            
            # Set random seed if provided for reproducibility
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                logger.info(f"🎬 Set random seed: {seed}")
            
            # Generate video frames with consistent parameters
            video_frames = self.pipeline(
                image,
                decode_chunk_size=8,
                motion_bucket_id=motion_bucket_id,
                fps=fps_id + 1,  # Convert fps_id to actual fps
                noise_aug_strength=cond_aug,
                num_frames=num_frames
            ).frames[0]
            
            # Save frames
            frames_dir = Path(output_dir)
            for i, frame in enumerate(video_frames):
                frame_path = frames_dir / f"frame_{i:04d}.png"
                frame.save(frame_path)
            
            logger.info(f"✅ SVD animation completed: {output_dir}")
            return output_dir
            
        except Exception as e:
            logger.error(f"Error in direct SVD animation: {e}")
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
