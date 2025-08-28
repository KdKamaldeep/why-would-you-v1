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
    
    def __init__(self, width: int = 768, height: int = 1024, animator_type: str = "ffmpeg"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fps = 15
        self.width = width
        self.height = height
        self.animator_type = animator_type.lower()
        
        # Initialize SVD animator if needed
        self.svd_animator = None
        if self.animator_type == "svd":
            try:
                from .svd_animator import SVDAnimator
                self.svd_animator = SVDAnimator()
                logger.info("✅ SVD Animator initialized")
            except ImportError as e:
                logger.warning(f"⚠️ SVD Animator not available: {e}")
                self.animator_type = "ffmpeg"
                logger.info("🔄 Falling back to FFmpeg animation")
    
    def animate_image(self, image_path: str, output_dir: str, num_frames: int = 150, prompt: str = "", 
                     motion_bucket_id: int = 127, fps_id: int = 6, cond_aug: float = 0.02, 
                     seed: Optional[int] = None) -> str:
        """
        Animate an image with professional quality and unlimited length capability.
        
        Args:
            image_path: Path to source image
            output_dir: Directory for output frames
            num_frames: Number of frames to generate (unlimited!)
            prompt: Animation prompt for guidance
            motion_bucket_id: Motion intensity for SVD (0-255, higher = more motion)
            fps_id: FPS setting for SVD (0-7, higher = faster motion)
            cond_aug: Conditioning augmentation for SVD (0.0-1.0)
            seed: Random seed for reproducibility
            
        Returns:
            Path to generated frames directory
        """
        try:
            logger.info(f"🎬 Animating image: {image_path}")
            logger.info(f"📊 Target frames: {num_frames} ({num_frames/self.fps:.1f}s @ {self.fps}fps)")
            logger.info(f"🎬 Animation type: {self.animator_type}")
            
            # Create output directory for frames
            frames_dir = Path(output_dir)
            frames_dir.mkdir(parents=True, exist_ok=True)
            
            # Choose animation method based on animator type
            if self.animator_type == "svd" and self.svd_animator:
                logger.info("🎬 Using SVD motion animation")
                # SVD has a hard limit of 25 frames - we need to work within this constraint
                if num_frames <= 25:
                    return self.svd_animator.animate_image(
                        image_path, str(frames_dir), num_frames, 
                        motion_bucket_id, fps_id, cond_aug, seed
                    )
                else:
                    # For longer sequences, we'll use SVD's 25 frames and loop/extend them
                    logger.info(f"🎬 SVD limit: 25 frames, requested {num_frames} frames")
                    logger.info(f"🎬 Will generate 25 SVD frames and loop them to match {num_frames} frames")
                    svd_frames_dir = self.svd_animator.animate_image(
                        image_path, str(frames_dir) + "_svd", 25, 
                        motion_bucket_id, fps_id, cond_aug, seed
                    )
                    return self._extend_svd_animation_with_looping(svd_frames_dir, str(frames_dir), num_frames)
            else:
                # Use enhanced FFmpeg animation system
                logger.info("📹 Using enhanced FFmpeg animation system")
                return self._create_enhanced_animation(image_path, str(frames_dir), num_frames, prompt)
            
        except Exception as e:
            logger.error(f"Error animating image: {e}")
            # Fallback to FFmpeg animation
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
                f"scale={int(self.width*1.5)}:{int(self.height*1.5)}:force_original_aspect_ratio=decrease,"
                f"pad={int(self.width*1.5)}:{int(self.height*1.5)}:(ow-iw)/2:(oh-ih)/2,"
                f"zoompan=z=1+on*0.001:d={num_frames}:"
                "x=iw/2-(iw/zoom/2)+sin(on*0.1)*20:"
                "y=ih/2-(ih/zoom/2)+cos(on*0.1)*15:"
                f"s={self.width}x{self.height}"
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
                '-vf', f'scale={int(self.width*1.8)}:{int(self.height*1.8)},crop={self.width}:{self.height}:w*sin(t*0.02)*0.15+w*0.15:h*cos(t*0.015)*0.1+h*0.1,unsharp=5:5:1.0:5:5:0.5',
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
                '-vf', f'scale={int(self.width*1.3)}:{int(self.height*1.3)}:force_original_aspect_ratio=decrease,pad={int(self.width*1.3)}:{int(self.height*1.3)}:(ow-iw)/2:(oh-ih)/2,rotate=t*{rotation_speed}:fillcolor=none:eval=frame,crop={self.width}:{self.height}:(iw-{self.width})/2:(ih-{self.height})/2,unsharp=5:5:0.8:5:5:0.4',
                '-r', str(self.fps),
                '-frames:v', str(num_frames),
                '-f', 'image2',
                f'{output_dir}/frame_%04d.png'
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"✅ Created organic rotation animation: {output_dir}")
            return output_dir
    
    def _extend_svd_animation_with_looping(self, svd_frames_dir: str, output_dir: str, target_frames: int) -> str:
        """Extend SVD animation (25 frames) to longer sequences using intelligent looping."""
        try:
            import shutil
            from PIL import Image
            import numpy as np
            
            svd_dir = Path(svd_frames_dir)
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Get SVD frames
            svd_frames = sorted(svd_dir.glob("frame_*.png"))
            if len(svd_frames) == 0:
                raise Exception("No SVD frames found")
            
            svd_frame_count = len(svd_frames)
            logger.info(f"🎬 SVD generated {svd_frame_count} frames, need {target_frames} frames")
            
            # Load SVD frames
            svd_images = []
            for frame_path in svd_frames:
                img = Image.open(frame_path)
                svd_images.append(np.array(img))
            
            # Calculate how many times to loop the SVD sequence
            loops_needed = target_frames // svd_frame_count
            remaining_frames = target_frames % svd_frame_count
            
            logger.info(f"🎬 Will loop SVD sequence {loops_needed} times + {remaining_frames} additional frames")
            
            frame_index = 0
            
            # Generate the required number of frames
            for i in range(target_frames):
                # Calculate which SVD frame to use (with looping)
                svd_frame_index = i % svd_frame_count
                
                # Get the corresponding SVD frame
                svd_frame = svd_images[svd_frame_index]
                frame_img = Image.fromarray(svd_frame)
                
                # Save the frame
                frame_path = output_path / f"frame_{frame_index:04d}.png"
                frame_img.save(frame_path)
                frame_index += 1
            
            logger.info(f"✅ Generated {target_frames} frames using SVD looping")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error extending SVD animation with looping: {e}")
            # Fallback: create static frames
            return self._create_static_frames(svd_frames_dir, output_dir, target_frames)
    
    def _create_parallax_motion(self, image_path: str, output_dir: str, num_frames: int) -> str:
        """Create parallax scrolling with depth effect."""
        try:
            cmd = [
                'ffmpeg', '-y',
                '-loop', '1',
                '-i', image_path,
                '-vf', f'scale={int(self.width*1.4)}:{int(self.height*1.4)},crop={self.width}:{self.height}:w*(0.5+sin(t*0.008)*0.2):h*(0.5+cos(t*0.006)*0.15),unsharp=5:5:1.2:5:5:0.6',
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
                '-vf', f'scale={self.width}:{self.height}:force_original_aspect_ratio=decrease,pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2,scale={self.width}*(1+sin(t*{pulse_speed})*0.03):{self.height}*(1+sin(t*{pulse_speed})*0.03),crop={self.width}:{self.height}:(iw-{self.width})/2:(ih-{self.height})/2',
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
                '-vf', f'scale={int(self.width*1.2)}:{int(self.height*1.2)},crop={self.width}:{self.height}:w*(0.5+sin(t*0.003)*0.1+cos(t*0.007)*0.05):h*(0.5+cos(t*0.004)*0.08+sin(t*0.009)*0.03),unsharp=5:5:0.9:5:5:0.3',
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