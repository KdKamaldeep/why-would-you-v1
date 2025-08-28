#!/usr/bin/env python3
"""
SVD (Stable Video Diffusion) Animator Module
Handles motion animation using SVD models for realistic video generation
"""

import os
import logging
import subprocess
import torch
from pathlib import Path
from typing import Optional, List
import json
import tempfile
import shutil

logger = logging.getLogger(__name__)

class SVDAnimator:
    """Handles SVD-based motion animation for images."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        self.device = device
        self.model_path = model_path or self._get_default_model_path()
        self.fps = 15
        self.num_frames = 25  # Default SVD frame count
        
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
    
    def animate_image(self, image_path: str, output_dir: str, num_frames: int = 25, 
                     motion_bucket_id: int = 127, fps_id: int = 6, 
                     cond_aug: float = 0.02, seed: Optional[int] = None) -> str:
        """
        Animate an image using SVD motion animation.
        
        Args:
            image_path: Path to source image
            output_dir: Directory for output frames
            num_frames: Number of frames to generate (SVD HARD LIMIT: 25 frames max)
            motion_bucket_id: Motion intensity (0-255, higher = more motion)
            fps_id: FPS setting (0-7, higher = faster motion)
            cond_aug: Conditioning augmentation (0.0-1.0)
            seed: Random seed for reproducibility
            
        Returns:
            Path to generated frames directory
            
        Note:
            SVD has a hard limit of 25 frames. For longer sequences, the AnimationGenerator
            will automatically loop these 25 frames to match the required duration.
        """
        try:
            logger.info(f"🎬 SVD Animating image: {image_path}")
            logger.info(f"📊 Target frames: {num_frames} (motion_bucket_id: {motion_bucket_id}, fps_id: {fps_id})")
            
            # Enforce SVD frame limit
            if num_frames > 25:
                logger.warning(f"⚠️ SVD frame limit exceeded: {num_frames} > 25. Clamping to 25 frames.")
                num_frames = 25
            
            # Create output directory
            frames_dir = Path(output_dir)
            frames_dir.mkdir(parents=True, exist_ok=True)
            
            # Use ComfyUI SVD workflow for animation
            return self._create_svd_animation(
                image_path, str(frames_dir), num_frames, 
                motion_bucket_id, fps_id, cond_aug, seed
            )
            
        except Exception as e:
            logger.error(f"Error in SVD animation: {e}")
            # Fallback to static frames
            return self._create_static_frames(image_path, output_dir, num_frames)
    
    def _create_svd_animation(self, image_path: str, output_dir: str, num_frames: int,
                             motion_bucket_id: int, fps_id: int, cond_aug: float, 
                             seed: Optional[int]) -> str:
        """Create SVD animation using ComfyUI workflow."""
        try:
            # Create temporary workflow file
            workflow = self._create_svd_workflow(
                image_path, num_frames, motion_bucket_id, fps_id, cond_aug, seed
            )
            
            # Run ComfyUI workflow
            output_video = self._run_comfyui_workflow(workflow, output_dir)
            
            # Extract frames from video
            self._extract_frames_from_video(output_video, output_dir, num_frames)
            
            logger.info(f"✅ SVD animation completed: {output_dir}")
            return output_dir
            
        except Exception as e:
            logger.error(f"Error creating SVD animation: {e}")
            return self._create_static_frames(image_path, output_dir, num_frames)
    
    def _create_svd_workflow(self, image_path: str, num_frames: int, 
                           motion_bucket_id: int, fps_id: int, cond_aug: float, 
                           seed: Optional[int]) -> dict:
        """Create ComfyUI workflow for SVD animation."""
        
        # Generate random seed if not provided
        if seed is None:
            import random
            seed = random.randint(1, 1000000)
        
        workflow = {
            "1": {
                "class_type": "LoadImage",
                "inputs": {
                    "image": image_path,
                    "choose file to upload": "image"
                }
            },
            "2": {
                "class_type": "SVD_img2vid_Conditioning",
                "inputs": {
                    "clip_vision": ["1", 1],
                    "vae": ["1", 2],
                    "width": 768,
                    "height": 1024,
                    "video_frames": num_frames,
                    "motion_bucket_id": motion_bucket_id,
                    "fps_id": fps_id,
                    "cond_aug": cond_aug
                }
            },
            "3": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": "svd_xt_1_1.safetensors"
                }
            },
            "4": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": 20,
                    "cfg": 1.0,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["3", 0],
                    "positive": ["2", 0],
                    "negative": ["2", 1],
                    "latent_image": ["2", 2]
                }
            },
            "5": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["4", 0],
                    "vae": ["3", 2]
                }
            },
            "6": {
                "class_type": "SaveVideo",
                "inputs": {
                    "images": ["5", 0],
                    "filename_prefix": "svd_output",
                    "fps": self.fps,
                    "crf": 20
                }
            }
        }
        
        return workflow
    
    def _run_comfyui_workflow(self, workflow: dict, output_dir: str) -> str:
        """Run ComfyUI workflow and return output video path."""
        try:
            # Create temporary workflow file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(workflow, f, indent=2)
                workflow_path = f.name
            
            # Run ComfyUI API
            import requests
            
            # Queue the workflow
            queue_response = requests.post(
                "http://127.0.0.1:8188/prompt",
                json={"prompt": workflow}
            )
            
            if queue_response.status_code != 200:
                raise Exception(f"Failed to queue workflow: {queue_response.text}")
            
            prompt_id = queue_response.json()["prompt_id"]
            
            # Wait for completion
            while True:
                history_response = requests.get(f"http://127.0.0.1:8188/history/{prompt_id}")
                if history_response.status_code == 200:
                    history = history_response.json()
                    if prompt_id in history:
                        # Workflow completed
                        outputs = history[prompt_id]["outputs"]
                        if "6" in outputs:  # SaveVideo node
                            video_info = outputs["6"]["images"][0]
                            video_path = video_info["filename"]
                            return video_path
                        break
                
                import time
                time.sleep(1)
            
            raise Exception("Workflow did not complete successfully")
            
        except Exception as e:
            logger.error(f"Error running ComfyUI workflow: {e}")
            # Fallback: create a simple video using FFmpeg
            return self._create_fallback_video(workflow, output_dir)
        finally:
            # Clean up temporary file
            if 'workflow_path' in locals():
                os.unlink(workflow_path)
    
    def _create_fallback_video(self, workflow: dict, output_dir: str) -> str:
        """Create a fallback video when ComfyUI is not available."""
        try:
            # Extract image path from workflow
            image_path = None
            for node_id, node in workflow.items():
                if node.get("class_type") == "LoadImage":
                    image_path = node["inputs"].get("image")
                    break
            
            if not image_path:
                raise Exception("Could not find image path in workflow")
            
            # Create a simple zoom animation as fallback
            output_video = os.path.join(output_dir, "fallback_video.mp4")
            
            cmd = [
                'ffmpeg', '-y',
                '-loop', '1',
                '-i', image_path,
                '-vf', f'scale=768:1024,zoompan=z=1+0.001*on:d=25:x=iw/2-(iw/zoom/2):y=ih/2-(ih/zoom/2):s=768x1024',
                '-r', str(self.fps),
                '-frames:v', '25',
                '-c:v', 'libx264',
                '-crf', '20',
                output_video
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            return output_video
            
        except Exception as e:
            logger.error(f"Error creating fallback video: {e}")
            raise
    
    def _extract_frames_from_video(self, video_path: str, output_dir: str, num_frames: int):
        """Extract frames from video to individual PNG files."""
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-vf', f'fps={self.fps}',
                '-frames:v', str(num_frames),
                '-f', 'image2',
                os.path.join(output_dir, 'frame_%04d.png')
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"✅ Extracted {num_frames} frames from video")
            
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            # Create static frames as fallback
            self._create_static_frames(video_path, output_dir, num_frames)
    
    def _create_static_frames(self, image_path: str, output_dir: str, num_frames: int) -> str:
        """Create static frames as fallback when animation fails."""
        try:
            from PIL import Image
            
            # Load the image
            img = Image.open(image_path)
            img = img.resize((768, 1024), Image.Resampling.LANCZOS)
            
            # Create frames directory
            frames_dir = Path(output_dir)
            frames_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the same image multiple times
            for i in range(num_frames):
                frame_path = frames_dir / f"frame_{i:04d}.png"
                img.save(frame_path)
            
            logger.info(f"✅ Created {num_frames} static frames as fallback")
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
