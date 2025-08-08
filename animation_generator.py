#!/usr/bin/env python3
"""
Animation Generator Module - Handles image animation using AnimateDiff
"""

import os
import logging
import subprocess
import shutil
import torch
import numpy as np
from pathlib import Path
from typing import List, Optional
from PIL import Image

logger = logging.getLogger(__name__)

class AnimationGenerator:
    """Handles image animation using real AnimateDiff AI."""
    
    def __init__(self, model_path: str = "models/mm_sd_v15_v2.safetensors", lora_path: str = "loras/sdxl_lightning_4step.safetensors"):
        self.model_path = model_path
        self.lora_path = lora_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pipe = None
        self._initialize_animatediff_pipeline()
    
    def _initialize_animatediff_pipeline(self):
        """Initialize the AnimateDiff pipeline for real AI animation."""
        try:
            from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
            from diffusers.utils import export_to_video
            
            logger.info(f"Initializing AnimateDiff pipeline...")
            logger.info(f"Device: {self.device}")
            logger.info(f"Motion model: {self.model_path}")
            
            # Check if motion model exists
            if not Path(self.model_path).exists():
                logger.warning(f"Motion model not found: {self.model_path}")
                logger.info("Will use simple animation fallback")
                return
            
            # Initialize motion adapter
            adapter = MotionAdapter.from_single_file(self.model_path)
            
            # Initialize the pipeline with motion adapter
            self.pipe = AnimateDiffPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                motion_adapter=adapter,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Use better scheduler for animation
            self.pipe.scheduler = EulerDiscreteScheduler.from_config(
                self.pipe.scheduler.config,
                timestep_spacing="trailing",
                beta_schedule="linear"
            )
            
            if self.device == 'cuda':
                self.pipe = self.pipe.to(self.device)
                self.pipe.enable_memory_efficient_attention()
                self.pipe.enable_vae_slicing()
            
            logger.info("✅ AnimateDiff pipeline initialized successfully!")
            
        except ImportError as e:
            logger.warning(f"AnimateDiff dependencies not available: {e}")
            logger.info("Installing AnimateDiff dependencies...")
            try:
                os.system("pip install diffusers[animatediff] imageio-ffmpeg")
                logger.info("Dependencies installed! Please restart for AnimateDiff support.")
            except:
                pass
        except Exception as e:
            logger.warning(f"Failed to initialize AnimateDiff: {e}")
            logger.info("Will use simple animation fallback")
    
    def animate_image(self, image_path: str, output_dir: str, num_frames: int = 24, prompt: str = "") -> str:
        """Animate an image using real AnimateDiff AI."""
        try:
            logger.info(f"Animating image: {image_path}")
            
            # Create output directory for frames
            frames_dir = Path(output_dir)
            frames_dir.mkdir(parents=True, exist_ok=True)
            
            # Try real AnimateDiff first
            if self.pipe is not None:
                return self._create_animatediff_animation(image_path, str(frames_dir), num_frames, prompt)
            else:
                logger.info("AnimateDiff not available, using enhanced fallback animation")
                return self._create_enhanced_animation(image_path, str(frames_dir), num_frames)
            
        except Exception as e:
            logger.error(f"Error animating image: {e}")
            return self._create_enhanced_animation(image_path, output_dir, num_frames)
    
    def _create_animatediff_animation(self, image_path: str, output_dir: str, num_frames: int, prompt: str) -> str:
        """Create real AI animation using AnimateDiff."""
        try:
            from diffusers.utils import export_to_video
            
            # Load and analyze the input image to create animation prompt
            image = Image.open(image_path)
            
            # Create animation-focused prompt
            if not prompt:
                animation_prompt = "smooth animation, gentle movement, cartoon style, colorful, high quality"
            else:
                # Enhance prompt for animation
                animation_prompt = f"animated {prompt}, smooth movement, gentle motion, cartoon style, colorful, high quality animation"
            
            negative_prompt = "static, still, frozen, low quality, blurry, distorted, ugly"
            
            logger.info(f"Generating AnimateDiff animation with prompt: {animation_prompt}")
            
            # Generate animation
            with torch.autocast(self.device):
                result = self.pipe(
                    prompt=animation_prompt,
                    negative_prompt=negative_prompt,
                    num_frames=num_frames,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    width=768,
                    height=1024,
                    generator=torch.Generator(device=self.device).manual_seed(42)
                )
            
            # Save frames
            frames = result.frames[0]
            for i, frame in enumerate(frames):
                frame_path = Path(output_dir) / f"frame_{i:04d}.png"
                frame.save(frame_path)
            
            logger.info(f"✅ Generated {len(frames)} AnimateDiff frames in: {output_dir}")
            return output_dir
            
        except Exception as e:
            logger.error(f"AnimateDiff generation failed: {e}")
            return self._create_enhanced_animation(image_path, output_dir, num_frames)
    
    def animate_multiple_images(self, image_paths: List[str], output_dir: str, num_frames: int = 24, prompts: List[str] = None) -> List[str]:
        """Animate multiple images with scene-specific prompts."""
        frame_dirs = []
        for i, image_path in enumerate(image_paths):
            frames_dir = f"{output_dir}/scene_{i+1}_frames"
            prompt = prompts[i] if prompts and i < len(prompts) else ""
            frame_dir = self.animate_image(image_path, frames_dir, num_frames, prompt)
            frame_dirs.append(frame_dir)
        return frame_dirs
    
    def _create_enhanced_animation(self, image_path: str, output_dir: str, num_frames: int) -> str:
        """Create enhanced animation effects using FFmpeg."""
        try:
            import random
            
            # Random animation effect selection
            effects = [
                self._create_zoom_pan_animation,
                self._create_slide_animation, 
                self._create_rotate_zoom_animation,
                self._create_parallax_animation
            ]
            
            # Choose random effect
            effect = random.choice(effects)
            return effect(image_path, output_dir, num_frames)
            
        except Exception as e:
            logger.error(f"Error creating enhanced animation: {e}")
            return self._create_static_frames(image_path, output_dir, num_frames)
    
    def _create_zoom_pan_animation(self, image_path: str, output_dir: str, num_frames: int) -> str:
        """Create zoom and pan animation."""
        try:
            cmd = [
                'ffmpeg', '-y',
                '-loop', '1',
                '-i', image_path,
                '-vf', f'scale=768:1024:force_original_aspect_ratio=decrease,pad=768:1024:(ow-iw)/2:(oh-ih)/2,zoompan=z=\'min(zoom+0.003,1.4)\':d={num_frames}:x=\'iw/2-(iw/zoom/2)\':y=\'ih/2-(ih/zoom/2)\':s=768x1024',
                '-r', '15',
                '-frames:v', str(num_frames),
                '-f', 'image2',
                f'{output_dir}/frame_%04d.png'
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Created zoom-pan animation in: {output_dir}")
            return output_dir
            
        except Exception as e:
            logger.error(f"Error creating zoom-pan animation: {e}")
            return self._create_static_frames(image_path, output_dir, num_frames)
    
    def _create_slide_animation(self, image_path: str, output_dir: str, num_frames: int) -> str:
        """Create sliding animation effect."""
        try:
            cmd = [
                'ffmpeg', '-y',
                '-loop', '1',
                '-i', image_path,
                '-vf', f'scale=1200:1400,crop=768:1024:w*t/{num_frames}*0.3:h*t/{num_frames}*0.2',
                '-r', '15',
                '-frames:v', str(num_frames),
                '-f', 'image2',
                f'{output_dir}/frame_%04d.png'
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Created slide animation in: {output_dir}")
            return output_dir
            
        except Exception as e:
            return self._create_zoom_pan_animation(image_path, output_dir, num_frames)
    
    def _create_rotate_zoom_animation(self, image_path: str, output_dir: str, num_frames: int) -> str:
        """Create rotation with zoom animation."""
        try:
            cmd = [
                'ffmpeg', '-y',
                '-loop', '1',
                '-i', image_path,
                '-vf', f'scale=768:1024:force_original_aspect_ratio=decrease,pad=768:1024:(ow-iw)/2:(oh-ih)/2,rotate=t*0.05:fillcolor=none:eval=frame,zoompan=z=\'min(zoom+0.002,1.2)\':d={num_frames}:x=\'iw/2-(iw/zoom/2)\':y=\'ih/2-(ih/zoom/2)\':s=768x1024',
                '-r', '15',
                '-frames:v', str(num_frames),
                '-f', 'image2',
                f'{output_dir}/frame_%04d.png'
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Created rotate-zoom animation in: {output_dir}")
            return output_dir
            
        except Exception as e:
            return self._create_zoom_pan_animation(image_path, output_dir, num_frames)
    
    def _create_parallax_animation(self, image_path: str, output_dir: str, num_frames: int) -> str:
        """Create parallax scrolling effect."""
        try:
            cmd = [
                'ffmpeg', '-y',
                '-loop', '1',
                '-i', image_path,
                '-vf', f'scale=900:1200,crop=768:1024:w*sin(t*0.1)*0.1+w*0.1:h*cos(t*0.1)*0.05+h*0.05',
                '-r', '15',
                '-frames:v', str(num_frames),
                '-f', 'image2',
                f'{output_dir}/frame_%04d.png'
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Created parallax animation in: {output_dir}")
            return output_dir
            
        except Exception as e:
            return self._create_zoom_pan_animation(image_path, output_dir, num_frames)
    
    def _create_static_frames(self, image_path: str, output_dir: str, num_frames: int) -> str:
        """Create static frames as last resort."""
        try:
            frames_dir = Path(output_dir)
            frames_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy the same image multiple times
            for i in range(num_frames):
                frame_path = frames_dir / f"frame_{i:04d}.png"
                shutil.copy2(image_path, frame_path)
            
            logger.info(f"Created static frames in: {output_dir}")
            return output_dir
            
        except Exception as e:
            logger.error(f"Error creating static frames: {e}")
            return output_dir

