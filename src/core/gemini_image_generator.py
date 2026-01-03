#!/usr/bin/env python3
"""
Gemini Image Generator Module - Handles image generation using Google Gemini API
"""

import os
import logging
from pathlib import Path
from typing import Optional
from PIL import Image

logger = logging.getLogger(__name__)

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("⚠️ google-genai package not installed. Install with: pip install google-genai")


class GeminiImageGenerator:
    """Handles image generation using Google Gemini API."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """
        Initialize the Gemini Image Generator.
        
        Args:
            api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
            model_name: Gemini model name (defaults to GEMINI_IMAGE_MODEL env var or "gemini-2.5-flash-image")
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name or os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
        
        if not GEMINI_AVAILABLE:
            logger.warning("⚠️ google-genai package not available - image generation will be disabled")
            self.available = False
            self.client = None
        elif not self.api_key:
            logger.warning("⚠️ GEMINI_API_KEY not set - image generation will be disabled")
            self.available = False
            self.client = None
        else:
            try:
                self.client = genai.Client(api_key=self.api_key)
                self.available = True
                logger.info(f"✅ Gemini Image Generator initialized with model: {self.model_name}")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Gemini client: {e}")
                self.available = False
                self.client = None
    
    def generate_image(self, prompt: str, output_path: str, width: int = 1280, height: int = 704) -> Optional[str]:
        """
        Generate an image from a text prompt using Gemini API.
        
        Args:
            prompt: Text prompt describing the image
            output_path: Path to save the generated image
            width: Image width (default: 1280)
            height: Image height (default: 704)
            
        Returns:
            Path to the generated image file, or None if generation fails
        """
        if not self.available or self.client is None:
            logger.warning("⚠️ Gemini image generation not available")
            return None
        
        try:
            logger.info(f"🎨 Generating image with Gemini: {prompt[:100]}...")
            
            # Generate image using Gemini API
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            
            # Extract image from response
            # The response structure may vary - check for image data
            if hasattr(response, 'candidates') and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        # Check if part contains image data
                        if hasattr(part, 'inline_data'):
                            image_data = part.inline_data.data
                            mime_type = part.inline_data.mime_type
                            
                            # Decode base64 image
                            import base64
                            import io
                            image_bytes = base64.b64decode(image_data)
                            
                            # Load image and resize
                            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                            image = image.resize((width, height), Image.Resampling.LANCZOS)
                            
                            # Save to output path
                            output_path_obj = Path(output_path)
                            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
                            image.save(output_path_obj, "PNG")
                            
                            logger.info(f"✅ Image generated and saved: {output_path}")
                            return str(output_path_obj)
            
            # Alternative: response might contain image URL or different structure
            # Try to extract from text response if it's a URL
            if hasattr(response, 'text'):
                text_response = response.text
                logger.warning(f"⚠️ Gemini returned text instead of image: {text_response[:200]}")
            
            logger.warning("⚠️ No image data in Gemini response")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error generating image with Gemini: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

