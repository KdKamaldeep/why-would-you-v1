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
        # Navigate to parts
        if "candidates" not in response_json:
            raise ValueError("No 'candidates' key in response")
        
        if len(response_json["candidates"]) == 0:
            raise ValueError("Empty candidates array")
        
        candidate = response_json["candidates"][0]
        if "content" not in candidate:
            raise ValueError("No 'content' key in candidate")
        
        if "parts" not in candidate["content"]:
            raise ValueError("No 'parts' key in content")
        
        parts = candidate["content"]["parts"]
        
        if not parts:
            raise ValueError("Empty parts array")

        for i, part in enumerate(parts):
            logger.debug(f"Part {i}: {list(part.keys()) if isinstance(part, dict) else type(part)}")
            
            # Check both snake_case (Python) and camelCase (API response) - matches Node.js implementation
            inline_data = None
            if "inlineData" in part:
                inline_data = part["inlineData"]
            elif "inline_data" in part:
                inline_data = part["inline_data"]
            
            if inline_data:
                logger.debug(f"Inline data keys: {list(inline_data.keys()) if isinstance(inline_data, dict) else type(inline_data)}")
                
                # Check both snake_case and camelCase for data and mime_type
                data = inline_data.get("data") or inline_data.get("data")
                mime = inline_data.get("mimeType") or inline_data.get("mime_type") or ""

                if not data:
                    logger.warning("⚠️ inline_data found but 'data' is empty")
                    continue

                logger.info(f"📦 Found inline_data: mime={mime}, data_type={type(data)}, data_length={len(data) if isinstance(data, str) else 'unknown'}")

                if not mime.startswith("image/"):
                    raise ValueError(f"Unexpected mime type: {mime}")

                # Check if data is already bytes or needs base64 decoding
                if isinstance(data, bytes):
                    img_bytes = data
                    logger.info("📦 Data is already bytes, skipping base64 decode")
                elif isinstance(data, str):
                    # Decode base64
                    try:
                        img_bytes = base64.b64decode(data)
                        logger.info(f"📦 Base64 decoded: {len(data)} chars -> {len(img_bytes)} bytes")
                    except Exception as e:
                        raise ValueError(f"Failed to base64 decode image data: {e}")
                else:
                    raise ValueError(f"Unexpected data type: {type(data)}")

                # Validate we have image data
                if not img_bytes or len(img_bytes) < 100:
                    logger.error(f"⚠️ Image data too small: {len(img_bytes) if img_bytes else 0} bytes")
                    logger.error(f"⚠️ First 50 bytes (hex): {img_bytes[:50].hex() if img_bytes else 'N/A'}")
                    logger.error(f"⚠️ First 50 bytes (ascii): {img_bytes[:50] if img_bytes else 'N/A'}")
                    raise ValueError(f"Invalid image data: {len(img_bytes) if img_bytes else 0} bytes (expected at least 100 bytes for valid image)")

                # Check if it's valid image data by checking magic bytes
                # Common image formats: PNG, JPEG, WebP
                is_valid_image = False
                if img_bytes.startswith(b'\x89PNG\r\n\x1a\n'):  # PNG
                    is_valid_image = True
                elif img_bytes.startswith(b'\xff\xd8\xff'):  # JPEG
                    is_valid_image = True
                elif img_bytes.startswith(b'RIFF') and b'WEBP' in img_bytes[:12]:  # WebP
                    is_valid_image = True
                
                if not is_valid_image:
                    # Log first bytes for debugging
                    hex_preview = img_bytes[:50].hex() if len(img_bytes) >= 50 else img_bytes.hex()
                    logger.warning(f"⚠️ Image data doesn't match known formats. First bytes (hex): {hex_preview}")
                    # Try to open anyway - PIL might still recognize it
                
                # Try to open the image
                try:
                    image = Image.open(io.BytesIO(img_bytes))
                    # Verify it's actually an image
                    image.verify()
                    # Reopen for actual use (verify() closes the image)
                    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    return image
                except Exception as e:
                    raise ValueError(f"PIL cannot open image data: {e}. Data length: {len(img_bytes)} bytes, mime: {mime}")

        raise ValueError("No inline image data found in Gemini response")

    except KeyError as e:
        raise RuntimeError(f"Gemini image extraction failed: Missing key in response - {e}")
    except ValueError as e:
        raise RuntimeError(f"Gemini image extraction failed: {e}")
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
    
    def _to_aspect_ratio(self, width: int, height: int) -> str:
        """Convert width/height to aspect ratio string (matches Node.js implementation)."""
        ratio = width / height
        
        if abs(ratio - 16/9) < 0.15:
            return "16:9"
        elif abs(ratio - 9/16) < 0.15:
            return "9:16"
        elif abs(ratio - 4/3) < 0.15:
            return "4:3"
        elif abs(ratio - 3/4) < 0.15:
            return "3:4"
        else:
            return "1:1"
    
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
            # IMPORTANT: Must pass imageConfig with aspectRatio for image generation (matches Node.js implementation)
            aspect_ratio = self._to_aspect_ratio(width, height)
            logger.info(f"📐 Using aspect ratio: {aspect_ratio} for {width}x{height}")
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "imageConfig": {
                        "aspectRatio": aspect_ratio
                    }
                }
            )
            
            # Convert response to JSON/dict for parsing
            # The google-genai library response needs to be converted to dict
            response_json = None
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
                    # Try to access response attributes directly as dict
                    # The google-genai library might use protobuf or similar
                    # Try accessing as dict-like object
                    try:
                        # Check if response has candidates attribute
                        if hasattr(response, 'candidates'):
                            # Build dict manually from response object
                            candidates = []
                            for cand in response.candidates:
                                if hasattr(cand, 'content') and hasattr(cand.content, 'parts'):
                                    parts = []
                                    for part in cand.content.parts:
                                        part_dict = {}
                                        if hasattr(part, 'inline_data') and part.inline_data:
                                            inline = part.inline_data
                                            part_dict['inline_data'] = {
                                                'data': getattr(inline, 'data', None),
                                                'mime_type': getattr(inline, 'mime_type', 'image/png')
                                            }
                                        parts.append(part_dict)
                                    candidates.append({
                                        'content': {'parts': parts}
                                    })
                            response_json = {'candidates': candidates}
                        else:
                            # Use JSON serialization as fallback
                            response_str = json.dumps(response, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o))
                            response_json = json.loads(response_str)
                    except Exception as inner_e:
                        logger.warning(f"⚠️ Failed to build dict from response object: {inner_e}")
                        # Last resort: try JSON serialization
                        response_str = json.dumps(response, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o))
                        response_json = json.loads(response_str)
            except Exception as e:
                logger.error(f"❌ Failed to convert response to JSON: {e}")
                logger.error(f"Response type: {type(response)}")
                logger.error(f"Response attributes: {[attr for attr in dir(response) if not attr.startswith('_')]}")
                raise RuntimeError(f"Cannot parse Gemini response: {type(response)} - {e}")
            
            # Log response structure for debugging
            logger.info(f"📋 Response type: {type(response_json)}")
            if isinstance(response_json, dict):
                logger.info(f"📋 Response keys: {list(response_json.keys())}")
                if 'candidates' in response_json:
                    logger.info(f"📋 Candidates count: {len(response_json['candidates'])}")
                    if len(response_json['candidates']) > 0:
                        candidate = response_json['candidates'][0]
                        logger.info(f"📋 Candidate type: {type(candidate)}")
                        if isinstance(candidate, dict):
                            logger.info(f"📋 Candidate keys: {list(candidate.keys())}")
                            if 'content' in candidate:
                                content = candidate['content']
                                logger.info(f"📋 Content type: {type(content)}")
                                if isinstance(content, dict):
                                    logger.info(f"📋 Content keys: {list(content.keys())}")
                                    if 'parts' in content:
                                        logger.info(f"📋 Parts count: {len(content['parts'])}")
                                        for i, part in enumerate(content['parts']):
                                            logger.info(f"📋 Part {i} type: {type(part)}")
                                            if isinstance(part, dict):
                                                logger.info(f"📋 Part {i} keys: {list(part.keys())}")
                                                if 'inline_data' in part:
                                                    inline = part['inline_data']
                                                    logger.info(f"📋 Inline data type: {type(inline)}")
                                                    if isinstance(inline, dict):
                                                        logger.info(f"📋 Inline data keys: {list(inline.keys())}")
                                                        logger.info(f"📋 Inline data values: {[(k, type(v).__name__, len(str(v)) if isinstance(v, (str, bytes)) else 'N/A') for k, v in inline.items()]}")
            
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

