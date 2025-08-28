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
    
    def __init__(self, width: int = 768, height: int = 1024, animator_type: str = "ffmpeg", svd_chunked_generation: bool = True, svd_overlap_frames: int = 6):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fps = 10  # Reduced from 15 to 10 for slower playback
        self.width = width
        self.height = height
        self.animator_type = animator_type.lower()
        self.svd_chunked_generation = svd_chunked_generation
        self.svd_overlap_frames = max(4, min(8, svd_overlap_frames))  # Clamp to 4-8 range
        
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
                # SVD has a hard limit of 24 frames - we need to work within this constraint
                if num_frames <= 24:
                    return self.svd_animator.animate_image(
                        image_path, str(frames_dir), num_frames, 
                        motion_bucket_id, fps_id, cond_aug, seed
                    )
                else:
                    # For longer sequences, choose between chunked generation and looping
                    logger.info(f"🎬 SVD limit: 24 frames, requested {num_frames} frames")
                    svd_frames_dir = self.svd_animator.animate_image(
                        image_path, str(frames_dir) + "_svd", 24, 
                        motion_bucket_id, fps_id, cond_aug, seed
                    )
                    
                    if self.svd_chunked_generation:
                        logger.info(f"🎬 Using chunked SVD generation for extended sequences")
                        return self._extend_svd_animation_with_chunked_generation(
                            svd_frames_dir, str(frames_dir), num_frames,
                            motion_bucket_id, fps_id, cond_aug, seed
                        )
                    else:
                        logger.info(f"🎬 Using SVD looping for extended sequences")
                        return self._extend_svd_animation_with_looping(
                            svd_frames_dir, str(frames_dir), num_frames
                        )
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
            
        except Exception as e:
            return self._create_cinematic_zoom_pan(image_path, output_dir, num_frames)
    
    def _extend_svd_animation_with_chunked_generation(self, svd_frames_dir: str, output_dir: str, target_frames: int,
                                                    motion_bucket_id: int, fps_id: int, cond_aug: float, 
                                                    seed: Optional[int] = None) -> str:
        """Extend SVD animation using overlapping chunked generation for visual consistency.
        
        This method implements overlapping chunk generation with 4-8 frame overlap between chunks
        to ensure smooth transitions and visual consistency across the entire sequence.
        Includes quality degradation detection and recovery for long sequences.
        """
        try:
            import shutil
            from PIL import Image
            import numpy as np
            
            svd_dir = Path(svd_frames_dir)
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Create a persistent directory for chunk processing
            chunks_dir = output_path / "chunks_processing"
            chunks_dir.mkdir(exist_ok=True)
            
            # Get initial SVD frames (first 24 frames)
            svd_frames = sorted(svd_dir.glob("frame_*.png"))
            if len(svd_frames) == 0:
                raise Exception("No SVD frames found")
            
            svd_frame_count = len(svd_frames)
            logger.info(f"🎬 Initial SVD generated {svd_frame_count} frames, need {target_frames} frames total")
            
            # Copy all initial SVD frames to output (these are the original 24 frames)
            frame_index = 0
            for frame_path in svd_frames:
                dest_path = output_path / f"frame_{frame_index:04d}.png"
                shutil.copy2(frame_path, dest_path)
                frame_index += 1
            
            # Calculate additional frames needed beyond the initial 24
            additional_frames_needed = target_frames - svd_frame_count
            if additional_frames_needed <= 0:
                logger.info(f"✅ No additional frames needed, using only initial {svd_frame_count} SVD frames")
                # Clean up chunks directory
                if chunks_dir.exists():
                    shutil.rmtree(chunks_dir)
                return str(output_path)
            
            # Overlapping chunk configuration
            frames_per_chunk = 24  # SVD limit (using 24 for overlap calculation)
            overlap_frames = self.svd_overlap_frames  # Configurable overlap frames (4-8 range)
            effective_frames_per_chunk = frames_per_chunk - overlap_frames  # New frames per chunk
            
            # Calculate chunks needed for additional frames
            chunks_needed = (additional_frames_needed + effective_frames_per_chunk - 1) // effective_frames_per_chunk
            logger.info(f"🎬 Need {additional_frames_needed} additional frames")
            logger.info(f"🎬 Using overlapping chunks: {frames_per_chunk} frames per chunk, {overlap_frames} frame overlap")
            logger.info(f"🎬 Effective new frames per chunk: {effective_frames_per_chunk}, will generate {chunks_needed} chunks")
            
            # Fixed parameters for consistency across all chunks
            fixed_seed = seed if seed is not None else 42
            fixed_motion_bucket_id = motion_bucket_id
            fixed_fps_id = fps_id
            fixed_cond_aug = cond_aug
            
            logger.info(f"🎬 Fixed parameters for consistency: seed={fixed_seed}, motion_bucket_id={fixed_motion_bucket_id}, fps_id={fixed_fps_id}, cond_aug={fixed_cond_aug}")
            
            # Quality degradation detection
            max_chunks_before_fallback = 6  # After 6 chunks, use fallback strategy
            use_original_frames_for_long_sequences = chunks_needed > max_chunks_before_fallback
            
            if use_original_frames_for_long_sequences:
                logger.info(f"🎬 Long sequence detected ({chunks_needed} chunks > {max_chunks_before_fallback}), using enhanced quality strategy")
            
            # Get the last frame from initial SVD generation as starting point for next chunk
            last_frame_path = svd_frames[-1]
            current_input_frame = last_frame_path
            
            # Track quality metrics
            consecutive_failures = 0
            max_consecutive_failures = 3
            
            # Generate additional chunks with overlap
            for chunk_idx in range(chunks_needed):
                logger.info(f"🎬 Generating overlapping chunk {chunk_idx + 1}/{chunks_needed}")
                logger.info(f"🎬 Input frame: {current_input_frame}")
                
                # For long sequences, periodically use original frames to maintain quality
                if use_original_frames_for_long_sequences and chunk_idx > 0 and chunk_idx % 3 == 0:
                    # Use a frame from the original SVD sequence instead of the last generated frame
                    original_frame_idx = min(chunk_idx * 2, len(svd_frames) - 1)
                    current_input_frame = svd_frames[original_frame_idx]
                    logger.info(f"🎬 Using original frame {original_frame_idx} for quality maintenance")
                
                # Create persistent directory for this chunk
                chunk_output_dir = chunks_dir / f"chunk_{chunk_idx + 1}"
                chunk_output_dir.mkdir(exist_ok=True)
                
                # Generate new SVD chunk using the current input frame
                chunk_seed = fixed_seed + chunk_idx
                try:
                    chunk_frames_dir = self.svd_animator.animate_image(
                        str(current_input_frame), 
                        str(chunk_output_dir), 
                        frames_per_chunk,  # Generate full 24 frames
                        fixed_motion_bucket_id, fixed_fps_id, fixed_cond_aug, 
                        chunk_seed
                    )
                    
                    # Copy chunk frames to main output with overlap handling
                    chunk_frames = sorted(Path(chunk_frames_dir).glob("frame_*.png"))
                    if len(chunk_frames) > 0:
                        # For first chunk, skip first frame to avoid duplication
                        # For subsequent chunks, use overlap frames for smooth transition
                        if chunk_idx == 0:
                            start_idx = 1  # Skip first frame to avoid duplication
                        else:
                            start_idx = overlap_frames  # Use overlap frames for smooth transition
                        
                        frames_added = 0
                        for i in range(start_idx, len(chunk_frames)):
                            if frame_index >= target_frames:
                                break
                            
                            src_path = chunk_frames[i]
                            dest_path = output_path / f"frame_{frame_index:04d}.png"
                            shutil.copy2(src_path, dest_path)
                            frame_index += 1
                            frames_added += 1
                        
                        # Update the last frame for next iteration
                        current_input_frame = chunk_frames[-1]
                        
                        logger.info(f"✅ Overlapping chunk {chunk_idx + 1} completed: {frames_added} frames added (started from frame {start_idx})")
                        consecutive_failures = 0  # Reset failure counter on success
                    else:
                        raise Exception("No frames generated")
                        
                except Exception as chunk_error:
                    consecutive_failures += 1
                    logger.warning(f"⚠️ Overlapping chunk {chunk_idx + 1} failed: {chunk_error}")
                    
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error(f"❌ Too many consecutive failures ({consecutive_failures}), switching to fallback strategy")
                        # Use looping strategy for remaining frames
                        remaining_frames = target_frames - frame_index
                        if remaining_frames > 0:
                            logger.info(f"🔄 Using looping strategy for remaining {remaining_frames} frames")
                            return self._extend_svd_animation_with_looping(svd_frames_dir, output_dir, target_frames)
                        break
                    
                    # Fallback: create some frames from the last known good frame
                    logger.info(f"🔄 Using fallback for chunk {chunk_idx + 1}: duplicating last good frame")
                    for i in range(effective_frames_per_chunk):
                        if frame_index >= target_frames:
                            break
                        dest_path = output_path / f"frame_{frame_index:04d}.png"
                        shutil.copy2(current_input_frame, dest_path)
                        frame_index += 1
            
        except Exception as e:
            logger.error(f"Error in overlapping chunked SVD generation: {e}")
            # Clean up chunks directory on error
            try:
                if chunks_dir.exists():
                    shutil.rmtree(chunks_dir)
            except Exception:
                pass
            # Fallback: create static frames
            return self._create_static_frames(svd_frames_dir, output_dir, target_frames)
    
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