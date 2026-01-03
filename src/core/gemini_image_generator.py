#!/usr/bin/env python3
"""
Gemini Image Generator Module - Handles image generation using Google Gemini API
"""

import os
import logging
import base64
import io
import json
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


def extract_gemini_image(response_json):
    """
    Extract image from Gemini JSON response.
    
    Args:
        response_json: Dictionary containing Gemini API response
        
    Returns:
        PIL Image in RGB format
        
    Raises:
        RuntimeError: If image extraction fails
    """
    try:
        parts = response_json["candidates"][0]["content"]["parts"]

        for part in parts:
            if "inline_data" in part:
                data = part["inline_data"]["data"]
                mime = part["inline_data"].get("mime_type", "")

                if not mime.startswith("image/"):
                    raise ValueError(f"Unexpected mime type: {mime}")

                img_bytes = base64.b64decode(data)
                return Image.open(io.BytesIO(img_bytes)).convert("RGB")

        raise ValueError("No inline image data found in Gemini response")

    except Exception as e:
        raise RuntimeError(f"Gemini image extraction failed: {e}")


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
            
            # Convert response to JSON/dict for parsing
            # The google-genai library response needs to be converted to dict
            try:
                # Try multiple methods to convert response to dict
                if isinstance(response, dict):
                    response_json = response
                elif hasattr(response, 'to_dict'):
                    response_json = response.to_dict()
                elif hasattr(response, '_raw_response'):
                    # Some SDKs store raw response
                    response_json = response._raw_response
                else:
                    # Use JSON serialization as fallback
                    # Convert response object to dict via JSON
                    response_str = json.dumps(response, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o))
                    response_json = json.loads(response_str)
            except Exception as e:
                logger.error(f"❌ Failed to convert response to JSON: {e}")
                raise RuntimeError(f"Cannot parse Gemini response: {type(response)} - {e}")
            
            # Extract image using helper function
            try:
                image = extract_gemini_image(response_json)
                
                # Resize and save image
                image = image.resize((width, height), Image.Resampling.LANCZOS)
                
                output_path_obj = Path(output_path)
                output_path_obj.parent.mkdir(parents=True, exist_ok=True)
                image.save(output_path_obj, "PNG")
                
                logger.info(f"✅ Image generated and saved: {output_path}")
                return str(output_path_obj)
                
            except RuntimeError as e:
                logger.error(f"❌ {e}")
                return None
            
        except Exception as e:
            logger.error(f"❌ Error generating image with Gemini: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

