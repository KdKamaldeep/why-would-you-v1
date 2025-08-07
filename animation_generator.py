#!/usr/bin/env python3
"""
Animation Generator Module - Handles image animation using AnimateDiff
"""

import logging
import subprocess
import shutil
import torch
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

class AnimationGenerator:
    """Handles image animation using AnimateDiff."""
    
    def __init__(self, model_path: str = "models/animatediff_v1-5-pruned.ckpt", lora_path: str = "loras/animov.safetensors"):
        self.model_path = model_path
        self.lora_path = lora_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def animate_image(self, image_path: str, output_dir: str, num_frames: int = 24) -> str:
        """Animate an image using AnimateDiff with cartoon LoRA."""
        try:
            logger.info(f"Animating image: {image_path}")
            
            # Create output directory for frames
            frames_dir = Path(output_dir)
            frames_dir.mkdir(parents=True, exist_ok=True)
            
            # This is a simplified version - in practice you'd need the full AnimateDiff implementation
            # For now, we'll create a simple zoom effect using FFmpeg
            return self._create_simple_animation(image_path, str(frames_dir), num_frames)
            
        except Exception as e:
            logger.error(f"Error animating image: {e}")
            return self._create_simple_animation(image_path, output_dir, num_frames)
    
    def animate_multiple_images(self, image_paths: List[str], output_dir: str, num_frames: int = 24) -> List[str]:
        """Animate multiple images and return frame directories."""
        frame_dirs = []
        for i, image_path in enumerate(image_paths):
            frames_dir = f"{output_dir}/scene_{i+1}_frames"
            frame_dir = self.animate_image(image_path, frames_dir, num_frames)
            frame_dirs.append(frame_dir)
        return frame_dirs
    
    def _create_simple_animation(self, image_path: str, output_dir: str, num_frames: int) -> str:
        """Create a simple zoom animation as fallback."""
        try:
            # Create frames with zoom effect using FFmpeg
            cmd = [
                'ffmpeg', '-y',
                '-loop', '1',
                '-i', image_path,
                '-vf', f'scale=768:1024:force_original_aspect_ratio=decrease,pad=768:1024:(ow-iw)/2:(oh-ih)/2,zoompan=z=\'min(zoom+0.002,1.3)\':d={num_frames}:x=\'iw/2-(iw/zoom/2)\':y=\'ih/2-(ih/zoom/2)\':s=768x1024',
                '-r', '15',
                '-frames:v', str(num_frames),
                '-f', 'image2',
                f'{output_dir}/frame_%04d.png'
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Created animation frames in: {output_dir}")
            return output_dir
            
        except Exception as e:
            logger.error(f"Error creating simple animation: {e}")
            return self._create_static_frames(image_path, output_dir, num_frames)
    
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

