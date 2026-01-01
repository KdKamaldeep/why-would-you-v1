#!/usr/bin/env python3
"""
Google Veo Text-to-Video Generator Module
Handles text-to-video generation using Google Veo via genai package
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
import time
import subprocess
import tempfile
import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logger.warning("google-genai package not available. Install with: pip install google-genai")


class VeoGenerator:
    """Handles text-to-video generation using Google Veo."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 width: int = 768,
                 height: int = 1024,
                 duration: int = 5,
                 negative_prompt: str = "text, subtitles, watermark, blurry, low quality, cartoon, anime, manga, illustration, painting, drawing, sketch, bad anatomy, distorted, deformed, ugly"):
        """
        Initialize the Veo generator.
        
        Args:
            api_key: Google Gemini API key. If None, will try to get from GOOGLE_API_KEY env var.
            width: Video width (default: 768 for shorts)
            height: Video height (default: 1024 for shorts)
            duration: Video duration in seconds (default: 5)
            negative_prompt: Negative prompt (default excludes cartoon/anime/illustration for realistic videos)
        """
        if not GENAI_AVAILABLE:
            raise ImportError("google-genai package not available. Install with: pip install google-genai")
        
        self.width = width
        self.height = height
        self.duration = duration
        self.negative_prompt = negative_prompt
        
        # Get API key from parameter or environment
        api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("Google Gemini API key is required. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize the genai client
        try:
            self.client = genai.Client(api_key=api_key)
            logger.info("✅ Google Veo client initialized successfully")
        except Exception as e:
            logger.error(f"❌ Error initializing Veo client: {e}")
            raise
    
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
            seed: Random seed for reproducibility (optional, not always supported by Veo)
            negative_prompt: Override default negative prompt (optional)
            duration: Target duration in seconds. If provided, will be used for generation.
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
        try:
            # Use provided duration or default
            target_duration = duration or self.duration
            if target_duration:
                logger.info(f"📏 Target duration: {target_duration:.2f}s")
            
            logger.info(f"🎬 Generating video with Google Veo 2...")
            logger.info(f"📝 Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
            logger.info(f"📐 Dimensions: {self.width}x{self.height}")
            
            # Use provided negative prompt or default
            neg_prompt = negative_prompt or self.negative_prompt
            if neg_prompt:
                logger.info(f"🚫 Negative prompt: {neg_prompt[:100]}{'...' if len(neg_prompt) > 100 else ''}")
            
            # Combine prompt with negative prompt if needed
            # Note: Effects (speed, zoom, transitions) are already in the prompt, don't add them
            full_prompt = prompt
            if neg_prompt:
                # Some models support negative prompts in the format, adjust based on Veo's API
                full_prompt = f"{prompt}. Avoid: {neg_prompt}"
            
            # Generate video using Veo
            logger.info("🎬 Running inference...")
            
            # Determine aspect ratio from dimensions
            aspect_ratio = "9:16" if self.height > self.width else "16:9"
            resolution = "1080p" if max(self.width, self.height) >= 1080 else "720p"
            
            # Generate video using Veo API
            operation = self.client.models.generate_videos(
                model="veo-2",
                prompt=full_prompt,
                config=genai.types.GenerateVideosConfig(
                    aspect_ratio=aspect_ratio,
                    resolution=resolution,
                    duration=int(target_duration) if target_duration else self.duration
                ),
            )
            
            # Wait for the video to be generated
            logger.info("⏳ Waiting for video generation to complete...")
            result = operation.result()
            
            # Check if we have generated videos
            if not result.generated_videos or len(result.generated_videos) == 0:
                raise RuntimeError("Veo did not return any videos in the response")
            
            # Get the first generated video
            generated_video = result.generated_videos[0]
            
            # Save video to file
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"💾 Downloading video to: {output_path}")
            
            # Download the video file
            self.client.files.download(file=generated_video.video, path=str(output_path))
            
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
        temp_frames_dir = tempfile.mkdtemp(prefix="veo_frames_")
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
        """Check if Veo is available."""
        return GENAI_AVAILABLE and hasattr(self, 'client') and self.client is not None


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test Veo generator
    try:
        generator = VeoGenerator(
            width=768,
            height=1024,
            duration=5
        )
        
        if generator.is_available():
            print("✅ Veo generator initialized successfully")
            
            # Test generation
            test_prompt = "A cat walks on the grass, realistic"
            output_path = generator.generate_video(
                prompt=test_prompt,
                output_path="test_veo_output.mp4"
            )
            print(f"✅ Test video generated: {output_path}")
        else:
            print("❌ Veo generator not available")
    except Exception as e:
        print(f"❌ Error: {e}")

