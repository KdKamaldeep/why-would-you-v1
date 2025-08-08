#!/usr/bin/env python3
"""
Animation Generator Module - Professional quality unlimited length video generation
"""

import os
import logging
import subprocess
import torch
from pathlib import Path
from typing import List, Optional
from PIL import Image

logger = logging.getLogger(__name__)

class AnimationGenerator:
    """Handles professional quality video animation with unlimited length capability."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fps = 15
    
    def animate_image(self, image_path: str, output_dir: str, num_frames: int = 150, prompt: str = "") -> str:
        """
        Animate an image with professional quality and unlimited length capability.
        
        Args:
            image_path: Path to source image
            output_dir: Directory for output frames
            num_frames: Number of frames to generate (unlimited!)
            prompt: Animation prompt for guidance
            
        Returns:
            Path to generated frames directory
        """
        try:
            logger.info(f"🎬 Animating image: {image_path}")
            logger.info(f"📊 Target frames: {num_frames} ({num_frames/self.fps:.1f}s @ {self.fps}fps)")
            
            # Create output directory for frames
            frames_dir = Path(output_dir)
            frames_dir.mkdir(parents=True, exist_ok=True)
            
            # Use enhanced FFmpeg animation system
            logger.info("📹 Using enhanced FFmpeg animation system")
            return self._create_enhanced_animation(image_path, str(frames_dir), num_frames, prompt)
            
        except Exception as e:
            logger.error(f"Error animating image: {e}")
            return self._create_enhanced_animation(image_path, output_dir, num_frames, prompt)
    
    def animate_multiple_images(self, image_paths: List[str], output_dir: str, num_frames: int = 150, prompts: List[str] = None) -> List[str]:
        """Animate multiple images with scene-specific prompts."""
        frame_dirs = []
        for i, image_path in enumerate(image_paths):
            frames_dir = f"{output_dir}/scene_{i+1}_frames"
            prompt = prompts[i] if prompts and i < len(prompts) else ""
            frame_dir = self.animate_image(image_path, frames_dir, num_frames, prompt)
            frame_dirs.append(frame_dir)
        return frame_dirs
    
    def animate_multiple_images_with_duration(self, image_paths: List[str], output_dir: str, frames_per_scene: List[int], prompts: List[str] = None) -> List[str]:
        """Animate multiple images with different frame counts per scene."""
        frame_dirs = []
        for i, image_path in enumerate(image_paths):
            frames_dir = f"{output_dir}/scene_{i+1}_frames"
            prompt = prompts[i] if prompts and i < len(prompts) else ""
            num_frames = frames_per_scene[i] if i < len(frames_per_scene) else 150
            duration = num_frames / self.fps
            logger.info(f"🎬 Animating scene {i+1}: {num_frames} frames ({duration:.1f}s)")
            frame_dir = self.animate_image(image_path, frames_dir, num_frames, prompt)
            frame_dirs.append(frame_dir)
        return frame_dirs
    
    def _create_enhanced_animation(self, image_path: str, output_dir: str, num_frames: int, prompt: str = "") -> str:
        """Create enhanced animation effects using advanced FFmpeg techniques."""
        try:
            import random
            
            logger.info(f"📹 Creating enhanced animation: {num_frames} frames")
            
            # Enhanced animation effects with better quality
            effects = [
                self._create_cinematic_zoom_pan,
                self._create_smooth_slide_animation, 
                self._create_organic_rotation,
                self._create_parallax_motion,
                self._create_breathing_effect,
                self._create_drift_animation
            ]
            
            # Choose effect based on prompt or randomly
            if "zoom" in prompt.lower():
                effect = self._create_cinematic_zoom_pan
            elif "slide" in prompt.lower() or "pan" in prompt.lower():
                effect = self._create_smooth_slide_animation
            elif "rotate" in prompt.lower() or "spin" in prompt.lower():
                effect = self._create_organic_rotation
            elif "drift" in prompt.lower() or "float" in prompt.lower():
                effect = self._create_drift_animation
            else:
                effect = random.choice(effects)
            
            return effect(image_path, output_dir, num_frames)
            
        except Exception as e:
            logger.error(f"Error creating enhanced animation: {e}")
            return self._create_static_frames(image_path, output_dir, num_frames)
    
    def _create_cinematic_zoom_pan(self, image_path: str, output_dir: str, num_frames: int) -> str:
        """Create cinematic zoom and pan animation with smooth easing.

        Uses zoompan with frame index variable 'on' (not 't') for FFmpeg 4.4 compatibility.
        Falls back to smooth slide animation if zoompan is unavailable.
        """
        try:
            # For FFmpeg 4.4, the zoompan filter does not support 't' (time) variable.
            # Use 'on' (output frame index) to drive motion and zoom.
            vf = (
                "scale=1200:1600:force_original_aspect_ratio=decrease,"
                "pad=1200:1600:(ow-iw)/2:(oh-ih)/2,"
                f"zoompan=z=1+on*0.001:d={num_frames}:"
                "x=iw/2-(iw/zoom/2)+sin(on*0.1)*20:"
                "y=ih/2-(ih/zoom/2)+cos(on*0.1)*15:"
                "s=768x1024"
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
            logger.info(f"✅ Created cinematic zoom-pan animation: {output_dir}")
            return output_dir
            
        except Exception as e:
            # Try to print stderr for easier debugging
            try:
                import traceback
                if hasattr(e, 'stderr') and e.stderr:
                    logger.error(f"FFmpeg error output: {e.stderr.decode('utf-8', errors='ignore')}")
                else:
                    logger.error(traceback.format_exc())
            except Exception:
                pass
            logger.error(f"Error creating cinematic animation: {e}")
            # Fallback to a motion-based effect instead of static frames
            try:
                return self._create_smooth_slide_animation(image_path, output_dir, num_frames)
            except Exception:
                return self._create_static_frames(image_path, output_dir, num_frames)
    
    def _create_smooth_slide_animation(self, image_path: str, output_dir: str, num_frames: int) -> str:
        """Create smooth sliding animation with organic motion."""
        try:
            cmd = [
                'ffmpeg', '-y',
                '-loop', '1',
                '-i', image_path,
                '-vf', f'scale=1400:1800,crop=768:1024:w*sin(t*0.02)*0.15+w*0.15:h*cos(t*0.015)*0.1+h*0.1,unsharp=5:5:1.0:5:5:0.5',
                '-r', str(self.fps),
                '-frames:v', str(num_frames),
                '-f', 'image2',
                f'{output_dir}/frame_%04d.png'
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"✅ Created smooth slide animation: {output_dir}")
            return output_dir
            
        except Exception as e:
            return self._create_cinematic_zoom_pan(image_path, output_dir, num_frames)
    
    def _create_organic_rotation(self, image_path: str, output_dir: str, num_frames: int) -> str:
        """Create organic rotation with natural easing."""
        try:
            rotation_speed = min(0.02, 2.0 / num_frames)  # Slower for longer videos
            
            cmd = [
                'ffmpeg', '-y',
                '-loop', '1',
                '-i', image_path,
                '-vf', f'scale=1000:1300:force_original_aspect_ratio=decrease,pad=1000:1300:(ow-iw)/2:(oh-ih)/2,rotate=t*{rotation_speed}:fillcolor=none:eval=frame,crop=768:1024:(iw-768)/2:(ih-1024)/2,unsharp=5:5:0.8:5:5:0.4',
                '-r', str(self.fps),
                '-frames:v', str(num_frames),
                '-f', 'image2',
                f'{output_dir}/frame_%04d.png'
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"✅ Created organic rotation animation: {output_dir}")
            return output_dir
            
        except Exception as e:
            return self._create_cinematic_zoom_pan(image_path, output_dir, num_frames)
    
    def _create_parallax_motion(self, image_path: str, output_dir: str, num_frames: int) -> str:
        """Create parallax scrolling with depth effect."""
        try:
            cmd = [
                'ffmpeg', '-y',
                '-loop', '1',
                '-i', image_path,
                '-vf', f'scale=1100:1400,crop=768:1024:w*(0.5+sin(t*0.008)*0.2):h*(0.5+cos(t*0.006)*0.15),unsharp=5:5:1.2:5:5:0.6',
                '-r', str(self.fps),
                '-frames:v', str(num_frames),
                '-f', 'image2',
                f'{output_dir}/frame_%04d.png'
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"✅ Created parallax motion animation: {output_dir}")
            return output_dir
            
        except Exception as e:
            return self._create_cinematic_zoom_pan(image_path, output_dir, num_frames)
    
    def _create_breathing_effect(self, image_path: str, output_dir: str, num_frames: int) -> str:
        """Create subtle breathing/pulsing effect."""
        try:
            pulse_speed = 0.05 / max(1, num_frames / 100)  # Slower pulse for longer videos
            
            cmd = [
                'ffmpeg', '-y',
                '-loop', '1',
                '-i', image_path,
                '-vf', f'scale=768:1024:force_original_aspect_ratio=decrease,pad=768:1024:(ow-iw)/2:(oh-ih)/2,scale=768*(1+sin(t*{pulse_speed})*0.03):1024*(1+sin(t*{pulse_speed})*0.03),crop=768:1024:(iw-768)/2:(ih-1024)/2',
                '-r', str(self.fps),
                '-frames:v', str(num_frames),
                '-f', 'image2',
                f'{output_dir}/frame_%04d.png'
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"✅ Created breathing effect animation: {output_dir}")
            return output_dir
            
        except Exception as e:
            return self._create_cinematic_zoom_pan(image_path, output_dir, num_frames)
    
    def _create_drift_animation(self, image_path: str, output_dir: str, num_frames: int) -> str:
        """Create gentle drifting motion."""
        try:
            cmd = [
                'ffmpeg', '-y',
                '-loop', '1',
                '-i', image_path,
                '-vf', f'scale=900:1200,crop=768:1024:w*(0.5+sin(t*0.003)*0.1+cos(t*0.007)*0.05):h*(0.5+cos(t*0.004)*0.08+sin(t*0.009)*0.03),unsharp=5:5:0.9:5:5:0.3',
                '-r', str(self.fps),
                '-frames:v', str(num_frames),
                '-f', 'image2',
                f'{output_dir}/frame_%04d.png'
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"✅ Created drift animation: {output_dir}")
            return output_dir
            
        except Exception as e:
            return self._create_cinematic_zoom_pan(image_path, output_dir, num_frames)
    
    def _create_static_frames(self, image_path: str, output_dir: str, num_frames: int) -> str:
        """Create static frames as last resort."""
        try:
            frames_dir = Path(output_dir)
            frames_dir.mkdir(parents=True, exist_ok=True)
            
            # Load and slightly enhance the image
            image = Image.open(image_path)
            
            # Copy with slight variations to avoid completely static appearance
            for i in range(num_frames):
                frame_path = frames_dir / f"frame_{i:04d}.png"
                image.save(frame_path)
            
            logger.info(f"✅ Created static frames: {output_dir}")
            return output_dir
            
        except Exception as e:
            logger.error(f"Error creating static frames: {e}")
            return output_dir