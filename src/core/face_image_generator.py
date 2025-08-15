#!/usr/bin/env python3
"""
Face-Based Image Generator Module
Handles generating images using faces from existing images with diffusion models
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional, Tuple, Union
from pathlib import Path
import logging
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", message=".*CLIPFeatureExtractor.*")
warnings.filterwarnings("ignore", message=".*Some weights of the model checkpoint were not used.*")

logger = logging.getLogger(__name__)

class FaceImageGenerator:
    """Handles face-based image generation using diffusion models."""
    
    def __init__(self, 
                 model_path: str = "models/toonyou_beta6.safetensors",
                 controlnet_path: Optional[str] = None,
                 ip_adapter_path: Optional[str] = None,
                 device: str = None):
        """
        Initialize the face-based image generator.
        
        Args:
            model_path: Path to the base diffusion model
            controlnet_path: Path to ControlNet model for face control
            ip_adapter_path: Path to IP-Adapter model for image prompting
            device: Device to run on ('cuda' or 'cpu')
        """
        self.model_path = model_path
        self.controlnet_path = controlnet_path
        self.ip_adapter_path = ip_adapter_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.face_detector = None
        self.face_landmark_detector = None
        self.pipe = None
        self.controlnet = None
        self.ip_adapter = None
        
        self._initialize_face_detection()
        self._initialize_diffusion_pipeline()
    
    def _initialize_face_detection(self):
        """Initialize face detection and landmark detection."""
        try:
            # Try to use mediapipe for face detection (no CMake required)
            import mediapipe as mp
            self.face_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )
            self.face_landmark_detector = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("✅ Face detection initialized with MediaPipe")
            
        except ImportError:
            try:
                # Fallback to OpenCV face detection
                import cv2
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_detector = cv2.CascadeClassifier(cascade_path)
                logger.info("✅ Face detection initialized with OpenCV")
            except Exception as e:
                logger.warning(f"⚠️ Face detection not available: {e}")
                self.face_detector = None
    
    def _initialize_diffusion_pipeline(self):
        """Initialize the diffusion pipeline with optional ControlNet and IP-Adapter."""
        try:
            from diffusers import StableDiffusionPipeline
            from diffusers.utils import load_image
            import safetensors
            
            # Try to import ControlNetPipeline, but don't fail if not available
            try:
                from diffusers import ControlNetPipeline
                self.controlnet_available = True
            except ImportError:
                self.controlnet_available = False
                logger.warning("⚠️ ControlNetPipeline not available in this version of diffusers")
            
            logger.info(f"🚀 Initializing diffusion pipeline...")
            logger.info(f"💻 Device: {self.device}")
            
            # Load base model
            if Path(self.model_path).exists():
                self.pipe = StableDiffusionPipeline.from_single_file(
                    self.model_path,
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False
                )
            else:
                # Fallback to pretrained model
                repo_id = os.getenv("SD_REPO_ID", "runwayml/stable-diffusion-v1-5")
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    repo_id,
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False
                )
            
            if self.device == 'cuda':
                self.pipe = self.pipe.to(self.device)
                if hasattr(self.pipe, 'enable_memory_efficient_attention'):
                    try:
                        self.pipe.enable_memory_efficient_attention()
                    except Exception:
                        pass
            
            # Load ControlNet if available
            if self.controlnet_path and Path(self.controlnet_path).exists() and self.controlnet_available:
                try:
                    from diffusers import ControlNetModel
                    self.controlnet = ControlNetModel.from_pretrained(
                        self.controlnet_path,
                        torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
                    )
                    if self.device == 'cuda':
                        self.controlnet = self.controlnet.to(self.device)
                    
                    # Create ControlNet pipeline
                    self.pipe = ControlNetPipeline.from_pretrained(
                        self.pipe,
                        controlnet=self.controlnet,
                        torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
                    )
                    logger.info("✅ ControlNet loaded successfully")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load ControlNet: {e}")
            elif self.controlnet_path and Path(self.controlnet_path).exists() and not self.controlnet_available:
                logger.warning("⚠️ ControlNet path exists but ControlNetPipeline is not available")
            
            # Load IP-Adapter if available
            if self.ip_adapter_path and Path(self.ip_adapter_path).exists():
                try:
                    from diffusers.utils import load_image
                    from ip_adapter import IPAdapter
                    
                    self.ip_adapter = IPAdapter(
                        self.pipe,
                        self.ip_adapter_path,
                        self.device
                    )
                    logger.info("✅ IP-Adapter loaded successfully")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load IP-Adapter: {e}")
            
            logger.info("✅ Diffusion pipeline initialized successfully")
            
        except ImportError as e:
            logger.error(f"❌ Failed to import required libraries: {e}")
            logger.info("💡 Please install: pip install diffusers transformers accelerate safetensors")
        except Exception as e:
            logger.error(f"❌ Failed to initialize diffusion pipeline: {e}")
    
    def detect_face(self, image: Union[str, Image.Image, np.ndarray]) -> Optional[Tuple[np.ndarray, List]]:
        """
        Detect face and landmarks in an image.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            
        Returns:
            Tuple of (face_region, landmarks) or None if no face detected
        """
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        
        if self.face_detector is None:
            logger.warning("⚠️ Face detector not available")
            return None
        
        try:
            # Use MediaPipe face detection
            if hasattr(self.face_detector, 'process'):
                results = self.face_detector.process(image)
                if results.detections:
                    detection = results.detections[0]
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = image.shape
                    
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    # Extract face region with padding
                    padding = int(min(width, height) * 0.2)
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(w, x + width + padding)
                    y2 = min(h, y + height + padding)
                    
                    face_region = image[y1:y2, x1:x2]
                    
                    # Get landmarks
                    landmark_results = self.face_landmark_detector.process(image)
                    landmarks = []
                    if landmark_results.multi_face_landmarks:
                        face_landmarks = landmark_results.multi_face_landmarks[0]
                        landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
                    
                    return face_region, landmarks
            
            # Fallback to OpenCV face detection
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                faces = self.face_detector.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    # Add padding
                    padding = int(min(w, h) * 0.2)
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(image.shape[1], x + w + padding)
                    y2 = min(image.shape[0], y + h + padding)
                    
                    face_region = image[y1:y2, x1:x2]
                    return face_region, []
        
        except Exception as e:
            logger.error(f"❌ Face detection failed: {e}")
            return None
    
    def create_face_control_image(self, face_image: np.ndarray, target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        """
        Create a control image from face landmarks for ControlNet.
        
        Args:
            face_image: Face image as numpy array
            target_size: Target size for the control image
            
        Returns:
            Control image as numpy array
        """
        try:
            # Resize face image
            face_resized = cv2.resize(face_image, target_size)
            
            # Convert to grayscale for edge detection
            gray = cv2.cvtColor(face_resized, cv2.COLOR_RGB2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Create control image (white edges on black background)
            control_image = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            control_image[edges > 0] = [255, 255, 255]
            
            return control_image
            
        except Exception as e:
            logger.error(f"❌ Failed to create control image: {e}")
            return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    
    def generate_with_face(self, 
                          prompt: str,
                          face_image_path: str,
                          negative_prompt: str = "",
                          num_inference_steps: int = 30,
                          guidance_scale: float = 7.5,
                          strength: float = 0.8,
                          output_path: Optional[str] = None) -> Optional[Image.Image]:
        """
        Generate an image using a face from an existing image.
        
        Args:
            prompt: Text prompt for generation
            face_image_path: Path to the image containing the face
            negative_prompt: Negative prompt
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for generation
            strength: Strength of face influence (0.0 to 1.0)
            output_path: Path to save the generated image
            
        Returns:
            Generated image as PIL Image or None if failed
        """
        if self.pipe is None:
            logger.error("❌ Diffusion pipeline not initialized")
            return None
        
        try:
            # Load and detect face
            face_data = self.detect_face(face_image_path)
            if face_data is None:
                logger.error("❌ No face detected in the input image")
                return None
            
            face_region, landmarks = face_data
            
            # Create control image if ControlNet is available
            control_image = None
            if self.controlnet is not None:
                control_image = self.create_face_control_image(face_region)
                control_image = Image.fromarray(control_image)
            
            # Prepare generation parameters
            generator = torch.Generator(device=self.device).manual_seed(42)
            
            # Generate image
            if control_image is not None:
                # Use ControlNet pipeline
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=control_image,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    controlnet_conditioning_scale=strength
                )
            elif self.ip_adapter is not None:
                # Use IP-Adapter
                face_pil = Image.fromarray(face_region)
                result = self.ip_adapter.generate(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=face_pil,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    ip_adapter_scale=strength
                )
            else:
                # Use base pipeline with enhanced prompt
                enhanced_prompt = f"{prompt}, face from reference image, detailed facial features"
                result = self.pipe(
                    prompt=enhanced_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator
                )
            
            generated_image = result.images[0]
            
            # Save if output path provided
            if output_path:
                generated_image.save(output_path)
                logger.info(f"✅ Generated image saved to: {output_path}")
            
            return generated_image
            
        except Exception as e:
            logger.error(f"❌ Image generation failed: {e}")
            return None
    
    def generate_face_swap_video(self,
                                prompt: str,
                                face_image_path: str,
                                background_prompt: str = "",
                                num_frames: int = 30,
                                output_path: str = "face_swap_video.mp4") -> bool:
        """
        Generate a video with face swapping using diffusion.
        
        Args:
            prompt: Prompt for the face generation
            face_image_path: Path to the face reference image
            background_prompt: Prompt for the background
            num_frames: Number of frames to generate
            output_path: Output video path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from moviepy.editor import ImageSequenceClip
            
            frames = []
            
            for i in range(num_frames):
                logger.info(f"🎬 Generating frame {i+1}/{num_frames}")
                
                # Generate frame with face
                frame = self.generate_with_face(
                    prompt=f"{prompt}, frame {i+1}",
                    face_image_path=face_image_path,
                    negative_prompt=background_prompt
                )
                
                if frame is None:
                    logger.error(f"❌ Failed to generate frame {i+1}")
                    return False
                
                frames.append(np.array(frame))
            
            # Create video
            clip = ImageSequenceClip(frames, fps=15)
            clip.write_videofile(output_path, codec='libx264')
            
            logger.info(f"✅ Face swap video saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Video generation failed: {e}")
            return False

# Example usage and testing
if __name__ == "__main__":
    # Initialize the face image generator
    generator = FaceImageGenerator()
    
    # Example: Generate image with face from reference
    result = generator.generate_with_face(
        prompt="A cartoon character in a magical forest",
        face_image_path="path/to/face_reference.jpg",
        output_path="generated_with_face.png"
    )
    
    if result:
        print("✅ Image generated successfully!")
    else:
        print("❌ Image generation failed!")
