#!/usr/bin/env python3
"""
Image Generator Module - Handles image generation using Stable Diffusion (legacy, not used in main pipeline)
"""

import os
import time
import logging
import warnings
import torch
from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional
from pathlib import Path
from .prompt_enhancer import PromptEnhancer

# Suppress deprecation warnings for CLIP classes
warnings.filterwarnings("ignore", message=".*CLIPFeatureExtractor.*")
warnings.filterwarnings("ignore", message=".*Some weights of the model checkpoint were not used.*")

logger = logging.getLogger(__name__)

class ImageGenerator:
    """Handles cartoon image generation using Stable Diffusion.

    Supports optional LoRA for style adaptation.
    """
    
    def __init__(self, model_path: str = "models/toonyou_beta6.safetensors", lora_path: Optional[str] = None, lora_scale: float = 0.8, enable_prompt_enhancement: bool = True, width: int = 768, height: int = 1024):
        self.model_path = model_path
        self.lora_path = lora_path
        self.lora_scale = lora_scale
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pipe = None
        self.sd_available = False
        self.enable_prompt_enhancement = enable_prompt_enhancement
        self.prompt_enhancer = None
        self.width = width
        self.height = height
        
        # Initialize prompt enhancer if enabled
        if self.enable_prompt_enhancement:
            self.prompt_enhancer = PromptEnhancer()
        
        self._initialize_pipeline()
        
    def _initialize_pipeline(self):
        """Initialize the Stable Diffusion pipeline."""
        # Import dependencies first, handle ImportError narrowly
        try:
            from diffusers import StableDiffusionPipeline
            import safetensors  # noqa: F401
        except ImportError as e:
            logger.warning(f"⚠️ Diffusers stack not available: {e}")
            logger.info("📦 Installing required packages...")
            try:
                import subprocess
                subprocess.run([
                    "pip", "install",
                    "diffusers>=0.25.0",
                    "transformers>=4.30.0",
                    "accelerate>=0.20.0",
                    "safetensors>=0.3.0"
                ], check=True)
                logger.info("✅ Packages installed! Please restart the script.")
            except Exception as install_error:
                logger.error(f"❌ Failed to install packages: {install_error}")
                logger.info("💡 Please install manually: pip install diffusers transformers accelerate safetensors")
            return

        try:
            logger.info(f"🚀 Initializing Stable Diffusion pipeline...")
            logger.info(f"💻 Device: {self.device}")
            logger.info(f"📁 Model path: {self.model_path}")

            # Check if model file exists; if not, try loading from Hugging Face repo id
            if not Path(self.model_path).exists():
                logger.warning(f"⚠️ Model file not found: {self.model_path}")
                repo_id = os.getenv("SD_REPO_ID")
                if not repo_id:
                    # Default to SD 1.5 base to ensure images can still be generated
                    repo_id = "runwayml/stable-diffusion-v1-5"
                    logger.info(f"💡 Falling back to pretrained model: {repo_id}")
                else:
                    logger.info(f"💡 Loading pretrained model from repo id: {repo_id}")
                try:
                    from diffusers import StableDiffusionPipeline
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
                        if hasattr(self.pipe, 'enable_xformers_memory_efficient_attention'):
                            try:
                                self.pipe.enable_xformers_memory_efficient_attention()
                            except Exception:
                                logger.info("ℹ️ xFormers not available; continuing without it")
                    # Optionally load LoRA after from_pretrained as well
                    if self.lora_path and Path(self.lora_path).exists():
                        try:
                            logger.info(f"🎭 Loading LoRA: {self.lora_path}")
                            load_ok = False
                            if hasattr(self.pipe, 'load_lora_weights'):
                                self.pipe.load_lora_weights(self.lora_path)
                                load_ok = True
                                if hasattr(self.pipe, 'fuse_lora'):
                                    try:
                                        self.pipe.fuse_lora(lora_scale=self.lora_scale)
                                    except Exception:
                                        pass
                                elif hasattr(self.pipe, 'set_adapters'):
                                    try:
                                        self.pipe.set_adapters(["default"], adapter_weights=[self.lora_scale])
                                    except Exception:
                                        pass
                            if load_ok:
                                logger.info(f"✅ LoRA loaded with scale ~ {self.lora_scale}")
                        except Exception as le:
                            logger.warning(f"⚠️ Failed to load LoRA '{self.lora_path}': {le}")
                    
                    # Configure tokenizer to avoid attention mask warnings
                    if hasattr(self.pipe, 'tokenizer') and self.pipe.tokenizer is not None:
                        if hasattr(self.pipe.tokenizer, 'pad_token') and self.pipe.tokenizer.pad_token is None:
                            self.pipe.tokenizer.pad_token = self.pipe.tokenizer.eos_token
                            logger.info("🔧 Configured tokenizer pad_token to avoid attention mask warnings")
                        
                        # Also configure the text encoder tokenizer if available
                        if hasattr(self.pipe, 'text_encoder') and hasattr(self.pipe.text_encoder, 'config'):
                            if hasattr(self.pipe.text_encoder.config, 'pad_token_id') and self.pipe.text_encoder.config.pad_token_id is None:
                                self.pipe.text_encoder.config.pad_token_id = self.pipe.tokenizer.eos_token_id
                                logger.info("🔧 Configured text encoder pad_token_id")
                    
                    self.sd_available = True
                    logger.info("✅ Stable Diffusion pipeline initialized from pretrained repo!")
                    return
                except Exception as e:
                    logger.warning(f"❌ Failed to load pretrained pipeline '{repo_id}': {e}")
                    logger.info("💡 Will use placeholder images instead. To enable SD, set SD_REPO_ID or place a .safetensors model.")
                    return

            # Load the pipeline with the custom model
            logger.info("📦 Loading Stable Diffusion model...")
            self.pipe = StableDiffusionPipeline.from_single_file(
                self.model_path,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
                use_safetensors=True
            )

            if self.device == 'cuda':
                self.pipe = self.pipe.to(self.device)
                # Enable memory optimizations (best-effort, never fail)
                if hasattr(self.pipe, 'enable_memory_efficient_attention'):
                    try:
                        self.pipe.enable_memory_efficient_attention()
                    except Exception:
                        pass
                if hasattr(self.pipe, 'enable_xformers_memory_efficient_attention'):
                    try:
                        self.pipe.enable_xformers_memory_efficient_attention()
                    except Exception:
                        logger.info("ℹ️ xFormers not available; continuing without it")

            # Optionally load a LoRA for style adaptation
            if self.lora_path and Path(self.lora_path).exists():
                try:
                    logger.info(f"🎭 Loading LoRA: {self.lora_path}")
                    load_ok = False
                    # Newer diffusers API
                    if hasattr(self.pipe, 'load_lora_weights'):
                        self.pipe.load_lora_weights(self.lora_path)
                        load_ok = True
                        # Try to fuse or set scale depending on API
                        if hasattr(self.pipe, 'fuse_lora'):
                            try:
                                self.pipe.fuse_lora(lora_scale=self.lora_scale)
                            except Exception:
                                pass
                        elif hasattr(self.pipe, 'set_adapters'):
                            try:
                                self.pipe.set_adapters(["default"], adapter_weights=[self.lora_scale])
                            except Exception:
                                pass
                    if load_ok:
                        logger.info(f"✅ LoRA loaded with scale ~ {self.lora_scale}")
                except Exception as le:
                    logger.warning(f"⚠️ Failed to load LoRA '{self.lora_path}': {le}")

            # Configure tokenizer to avoid attention mask warnings
            if hasattr(self.pipe, 'tokenizer') and self.pipe.tokenizer is not None:
                if hasattr(self.pipe.tokenizer, 'pad_token') and self.pipe.tokenizer.pad_token is None:
                    self.pipe.tokenizer.pad_token = self.pipe.tokenizer.eos_token
                    logger.info("🔧 Configured tokenizer pad_token to avoid attention mask warnings")
                
                # Also configure the text encoder tokenizer if available
                if hasattr(self.pipe, 'text_encoder') and hasattr(self.pipe.text_encoder, 'config'):
                    if hasattr(self.pipe.text_encoder.config, 'pad_token_id') and self.pipe.text_encoder.config.pad_token_id is None:
                        self.pipe.text_encoder.config.pad_token_id = self.pipe.tokenizer.eos_token_id
                        logger.info("🔧 Configured text encoder pad_token_id")

            self.sd_available = True
            logger.info("✅ Stable Diffusion pipeline initialized successfully!")
            logger.info("🎨 Ready to generate professional images!")

        except Exception as e:
            logger.warning(f"❌ Failed to initialize SD pipeline: {e}")
            logger.info("💡 Will use placeholder images instead")
            self.sd_available = False
    
    def generate_cartoon_image(self, prompt: str, output_path: str, subtitle: str = None, negative_prompt: str = None, character_faces: dict = None) -> str:
        """Generate a cartoon-style image using Stable Diffusion with optional subtitle, negative prompt, and character faces."""
        try:
            logger.info(f"🎨 Generating image for prompt: {prompt}")
            if negative_prompt:
                logger.info(f"🎨 Using negative prompt: {negative_prompt}")
            if character_faces:
                logger.info(f"👥 Using character faces: {list(character_faces.keys())}")
            
            # If we have character faces and face-based generation is available, use it
            if character_faces and self._can_use_face_generation():
                result_path = self._generate_face_based_image(prompt, output_path, negative_prompt, character_faces)
            # If we have a working pipeline, use it
            elif self.sd_available and self.pipe is not None:
                result_path = self._generate_sd_image(prompt, output_path, negative_prompt)
            else:
                # Fallback to placeholder
                logger.info("⚠️ Using placeholder image (SD pipeline not available)")
                result_path = self._generate_placeholder_image(prompt, output_path, subtitle)
            
            # Add subtitle if provided
            if subtitle:
                result_path = self._add_subtitle_to_image(result_path, subtitle)
                
            return result_path
                
        except Exception as e:
            logger.error(f"❌ Error generating image: {e}")
            return self._generate_placeholder_image(prompt, output_path, subtitle)
    
    def _can_use_face_generation(self) -> bool:
        """Check if face-based generation is available."""
        try:
            # Check if face detection libraries are available
            import mediapipe
            return True
        except ImportError:
            try:
                import cv2
                return True
            except ImportError:
                return False
    
    def _generate_face_based_image(self, prompt: str, output_path: str, negative_prompt: str = None, character_faces: dict = None) -> str:
        """Generate image using face-based generation for characters."""
        # Note: Face-based generation removed - using standard SD generation instead
        logger.warning("⚠️ Face-based generation not available, using standard image generation")
        return self._generate_sd_image(prompt, output_path, negative_prompt)
    
    def _find_best_character_face(self, prompt: str, character_faces: dict) -> str:
        """Find the best matching character face for the given prompt."""
        if not character_faces:
            return None
        
        # Simple keyword matching - can be enhanced with more sophisticated NLP
        prompt_lower = prompt.lower()
        
        for character_name, face_path in character_faces.items():
            character_lower = character_name.lower()
            
            # Check if character name appears in the prompt
            if character_lower in prompt_lower:
                logger.info(f"🎭 Found character '{character_name}' in prompt")
                return face_path
            
            # Check for common variations
            if character_lower.replace(' ', '') in prompt_lower.replace(' ', ''):
                logger.info(f"🎭 Found character '{character_name}' (variation) in prompt")
                return face_path
        
        # If no exact match, try to find the most relevant character
        # This is a simple heuristic - can be improved
        for character_name, face_path in character_faces.items():
            character_lower = character_name.lower()
            
            # Check for common character types
            character_types = ['lion', 'robot', 'princess', 'king', 'queen', 'wizard', 'dragon', 'cat', 'dog', 'bear']
            for char_type in character_types:
                if char_type in character_lower and char_type in prompt_lower:
                    logger.info(f"🎭 Found character type '{char_type}' for '{character_name}'")
                    return face_path
        
        logger.info("🎭 No character face match found, using first available face")
        return list(character_faces.values())[0] if character_faces else None
    
    def _generate_sd_image(self, prompt: str, output_path: str, negative_prompt: str = None) -> str:
        """Generate image using Stable Diffusion with intelligent aspect ratio optimization."""
        try:
            # Apply professional prompt optimization
            if self.enable_prompt_enhancement and self.prompt_enhancer:
                try:
                    # Use the new professional enhancer
                    from .prompt_enhancer import ProfessionalPromptEnhancer
                    enhancer = ProfessionalPromptEnhancer()
                    analysis = enhancer.analyze_prompt(prompt, style="realistic")
                    
                    # Use enhanced prompt if it's significantly better
                    if analysis.clarity_score > 0.6 and analysis.structure_score > 0.5:
                        final_prompt = analysis.enhanced_prompt
                        optimized_negative = analysis.optimized_negative_prompt
                        logger.info(f"🎯 Using enhanced prompt: {final_prompt}")
                        logger.info(f"🎯 Using optimized negative prompt: {optimized_negative}")
                        
                        # Update negative prompt if we have a better one
                        if optimized_negative and not negative_prompt:
                            negative_prompt = optimized_negative
                    else:
                        final_prompt = prompt
                        logger.info(f"🎯 Using original prompt (enhancement not beneficial): {prompt}")
                except Exception as e:
                    logger.warning(f"⚠️ Prompt enhancement failed: {e}")
                    final_prompt = prompt
            else:
                final_prompt = prompt
                logger.info(f"🎯 Using original prompt: {prompt}")
            
            # Determine aspect ratio for intelligent optimization
            is_16_9_format = self.width > self.height and self.width / self.height > 1.5
            
            # Validate dimensions - some models have limitations
            max_dimension = 1024  # Common limit for many SD models
            if self.width > max_dimension or self.height > max_dimension:
                logger.warning(f"⚠️ Dimensions {self.width}x{self.height} exceed recommended maximum of {max_dimension}")
                logger.info(f"🔄 Scaling down to fit within limits while preserving aspect ratio")
                
                # Scale down while preserving aspect ratio
                if self.width > self.height:
                    # Landscape
                    new_width = max_dimension
                    new_height = int((self.height / self.width) * max_dimension)
                else:
                    # Portrait
                    new_height = max_dimension
                    new_width = int((self.width / self.height) * max_dimension)
                
                # Ensure dimensions are multiples of 8 (SD requirement)
                new_width = (new_width // 8) * 8
                new_height = (new_height // 8) * 8
                
                logger.info(f"📐 Scaled dimensions: {new_width}x{new_height}")
                actual_width, actual_height = new_width, new_height
            else:
                actual_width, actual_height = self.width, self.height
            
            # Memory optimization for larger images
            if actual_width * actual_height > 768 * 1024:  # If larger than shorts format
                logger.info(f"🧠 Large image detected, optimizing memory usage")
                # Reduce steps for larger images to save memory
                num_steps = 20  # Reduced from 30
                logger.info(f"📉 Reduced steps to {num_steps} for memory optimization")
            else:
                num_steps = 30
            
            # Use provided negative prompt or create intelligent default
            if negative_prompt is None:
                # Base negative prompt for realistic style
                base_negative_prompt = (
                    "photorealistic, realistic, photo, 3d render, cgi, anime, manga, "
                    "blurry, low quality, dark, scary, violent, adult content, nsfw, "
                    "hyperrealistic, detailed textures, photographic, film grain, "
                    "realistic lighting, realistic shadows, realistic proportions, "
                    "detailed skin, detailed hair, detailed clothing textures"
                )
                
                # Use base negative prompt for all formats - no static composition handling
                negative_prompt = base_negative_prompt
            else:
                # Use user-provided negative prompt as-is - no automatic enhancement
                pass
            
            # Use consistent generation parameters for all formats
            guidance_scale = 7.5
            logger.info(f"🎬 Using standard settings: guidance_scale={guidance_scale}, steps={num_steps}")
            logger.info(f"📐 Generating with dimensions: {actual_width}x{actual_height}")
            
            # Clear CUDA cache before generation if available
            if self.device == 'cuda':
                try:
                    import torch
                    torch.cuda.empty_cache()
                    logger.info("🧹 CUDA cache cleared before generation")
                except Exception:
                    pass
            
            # Generate image with optimized settings
            with torch.autocast(self.device):
                result = self.pipe(
                    prompt=final_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    width=actual_width,
                    height=actual_height,
                    num_images_per_prompt=1,
                    generator=torch.Generator(device=self.device).manual_seed(42),  # Consistent results
                    return_dict=True
                )
            
            # Save the image
            image = result.images[0]
            
            # If we scaled down, resize to original dimensions
            if actual_width != self.width or actual_height != self.height:
                logger.info(f"🔄 Resizing from {actual_width}x{actual_height} to {self.width}x{self.height}")
                image = image.resize((self.width, self.height), Image.Resampling.LANCZOS)
            
            image.save(output_path, quality=95)
            
            # Clear CUDA cache after generation
            if self.device == 'cuda':
                try:
                    import torch
                    torch.cuda.empty_cache()
                    logger.info("🧹 CUDA cache cleared after generation")
                except Exception:
                    pass
            
            logger.info(f"✅ Generated professional SD image: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ SD generation failed: {e}")
            logger.info("🔄 Falling back to placeholder image")
            
            # Clear CUDA cache on error
            if self.device == 'cuda':
                try:
                    import torch
                    torch.cuda.empty_cache()
                    logger.info("🧹 CUDA cache cleared after error")
                except Exception:
                    pass
            
            return self._generate_placeholder_image(prompt, output_path)
    
    def generate_multiple_images(self, prompts: List[str], output_dir: str, subtitles: List[str] = None, negative_prompts: List[str] = None) -> List[str]:
        """Generate multiple cartoon images for a list of prompts with optional subtitles and negative prompts."""
        image_paths = []
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for i, prompt in enumerate(prompts):
            logger.info(f"🎬 Generating scene {i+1}/{len(prompts)}")
            output_path = f"{output_dir}/scene_{i+1}.png"
            
            # Get subtitle if available
            subtitle = None
            if subtitles and i < len(subtitles):
                subtitle = subtitles[i]
            
            # Get negative prompt if available
            negative_prompt = None
            if negative_prompts and i < len(negative_prompts):
                negative_prompt = negative_prompts[i]
            
            image_path = self.generate_cartoon_image(prompt, output_path, subtitle, negative_prompt)
            image_paths.append(image_path)
            
            # Small delay between generations to prevent memory issues
            if i < len(prompts) - 1:
                time.sleep(1)
        
        logger.info(f"✅ Generated {len(image_paths)} images successfully!")
        return image_paths
    
    def _add_subtitle_to_image(self, image_path: str, subtitle: str) -> str:
        """Add subtitle text to an existing image."""
        try:
            # Load the image
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Create a copy to draw on
            img_with_subtitle = img.copy()
            draw = ImageDraw.Draw(img_with_subtitle)
            
            # Try to load a good font for subtitles
            try:
                # Try different font options
                font_options = [
                    "arial.ttf",
                    "Arial.ttf", 
                    "DejaVuSans.ttf",
                    "LiberationSans-Regular.ttf"
                ]
                subtitle_font = None
                for font_path in font_options:
                    try:
                        subtitle_font = ImageFont.truetype(font_path, 32)
                        break
                    except:
                        continue
                
                if subtitle_font is None:
                    subtitle_font = ImageFont.load_default()
            except:
                subtitle_font = ImageFont.load_default()
            
            # Calculate subtitle position (bottom of image)
            img_width, img_height = img.size
            subtitle_y = img_height - 120  # 120 pixels from bottom
            
            # Split subtitle into lines if too long
            words = subtitle.split()
            lines = []
            current_line = []
            
            for word in words:
                current_line.append(word)
                test_line = ' '.join(current_line)
                bbox = draw.textbbox((0, 0), test_line, font=subtitle_font)
                if bbox[2] - bbox[0] > img_width - 40:  # 20px margin on each side
                    if len(current_line) > 1:
                        current_line.pop()
                        lines.append(' '.join(current_line))
                        current_line = [word]
                    else:
                        lines.append(word)
                        current_line = []
            
            if current_line:
                lines.append(' '.join(current_line))
            
            # Limit to 3 lines maximum
            lines = lines[:3]
            
            # Draw subtitle background
            line_height = 40
            total_height = len(lines) * line_height
            bg_y_start = subtitle_y - 10
            bg_y_end = subtitle_y + total_height + 10
            
            # Semi-transparent background
            bg_overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            bg_draw = ImageDraw.Draw(bg_overlay)
            bg_draw.rectangle([0, bg_y_start, img_width, bg_y_end], fill=(0, 0, 0, 180))
            
            # Composite the background
            img_with_subtitle = Image.alpha_composite(img_with_subtitle.convert('RGBA'), bg_overlay).convert('RGB')
            draw = ImageDraw.Draw(img_with_subtitle)
            
            # Draw subtitle text
            for i, line in enumerate(lines):
                bbox = draw.textbbox((0, 0), line, font=subtitle_font)
                text_width = bbox[2] - bbox[0]
                x = (img_width - text_width) // 2
                y = subtitle_y + (i * line_height)
                
                # Draw text shadow
                draw.text((x + 2, y + 2), line, fill='black', font=subtitle_font)
                # Draw main text
                draw.text((x, y), line, fill='white', font=subtitle_font)
            
            # Save the image with subtitle
            img_with_subtitle.save(image_path, quality=95)
            logger.info(f"✅ Added subtitle to image: {subtitle}")
            return image_path
            
        except Exception as e:
            logger.error(f"❌ Error adding subtitle to image: {e}")
            return image_path

    def _generate_placeholder_image(self, prompt: str, output_path: str, subtitle: str = None) -> str:
        """Generate a colorful placeholder image with better design and optional subtitle."""
        try:
            # Create a gradient background instead of solid blue
            img = Image.new('RGB', (self.width, self.height), color='white')
            draw = ImageDraw.Draw(img)
            
            # Create a colorful gradient background
            for y in range(self.height):
                color_r = int(135 + (y / self.height) * 120)  # 135-255
                color_g = int(206 + (y / self.height) * 49)   # 206-255  
                color_b = int(250 - (y / self.height) * 50)   # 250-200
                color = (min(255, color_r), min(255, color_g), min(255, color_b))
                draw.line([(0, y), (self.width, y)], fill=color)
            
            # Add decorative elements
            # Draw some simple shapes for visual appeal
            draw.ellipse([50, 50, 150, 150], fill='yellow', outline='orange', width=3)
            draw.rectangle([600, 100, 700, 200], fill='lightgreen', outline='green', width=3)
            draw.ellipse([100, 800, 200, 900], fill='pink', outline='red', width=3)
            draw.rectangle([550, 850, 650, 950], fill='lightcoral', outline='darkred', width=3)
            
            # Add text with better formatting
            try:
                # Try different font sizes
                title_font = ImageFont.truetype("arial.ttf", 36)
                text_font = ImageFont.truetype("arial.ttf", 24)
            except:
                title_font = ImageFont.load_default()
                text_font = ImageFont.load_default()
            
            # Title
            title = "🎬 Video Scene"
            title_bbox = draw.textbbox((0, 0), title, font=title_font)
            title_width = title_bbox[2] - title_bbox[0]
            title_x = (self.width - title_width) // 2
            
            # Add text shadow
            draw.text((title_x + 2, 302), title, fill='gray', font=title_font)
            draw.text((title_x, 300), title, fill='darkblue', font=title_font)
            
            # Scene description
            words = prompt.split()
            lines = []
            current_line = []
            
            for word in words:
                current_line.append(word)
                test_line = ' '.join(current_line)
                bbox = draw.textbbox((0, 0), test_line, font=text_font)
                if bbox[2] - bbox[0] > self.width - 100:  # Max width with margin
                    if len(current_line) > 1:
                        current_line.pop()
                        lines.append(' '.join(current_line))
                        current_line = [word]
                    else:
                        lines.append(word)
                        current_line = []
            
            if current_line:
                lines.append(' '.join(current_line))
            
            # Limit to 8 lines
            lines = lines[:8]
            
            y_offset = int(self.height * 0.4)  # 40% from top
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=text_font)
                text_width = bbox[2] - bbox[0]
                x = (self.width - text_width) // 2
                
                # Add text shadow
                draw.text((x + 1, y_offset + 1), line, fill='gray', font=text_font)
                draw.text((x, y_offset), line, fill='darkblue', font=text_font)
                y_offset += 35
            
            # Add subtitle if provided
            if subtitle:
                try:
                    subtitle_font = ImageFont.truetype("arial.ttf", 28)
                except:
                    subtitle_font = ImageFont.load_default()
                
                # Draw subtitle background
                subtitle_y = int(self.height * 0.85)  # 85% from top
                bbox = draw.textbbox((0, 0), subtitle, font=subtitle_font)
                text_width = bbox[2] - bbox[0]
                x = (self.width - text_width) // 2
                
                # Semi-transparent background for subtitle
                bg_overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
                bg_draw = ImageDraw.Draw(bg_overlay)
                bg_draw.rectangle([x-10, subtitle_y-10, x+text_width+10, subtitle_y+40], fill=(0, 0, 0, 180))
                
                # Composite the background
                img = Image.alpha_composite(img.convert('RGBA'), bg_overlay).convert('RGB')
                draw = ImageDraw.Draw(img)
                
                # Draw subtitle text
                draw.text((x + 2, subtitle_y + 2), subtitle, fill='black', font=subtitle_font)
                draw.text((x, subtitle_y), subtitle, fill='white', font=subtitle_font)
            
            # Add a note about the placeholder (only if no subtitle)
            if not subtitle:
                note = "⚠️ Placeholder - Install diffusers for AI images"
                note_bbox = draw.textbbox((0, 0), note, font=text_font)
                note_width = note_bbox[2] - note_bbox[0]
                note_x = (768 - note_width) // 2
                draw.text((note_x, 950), note, fill='red', font=text_font)
                
                # Add installation instructions
                install_note = "Run: pip install diffusers transformers accelerate safetensors"
                install_bbox = draw.textbbox((0, 0), install_note, font=text_font)
                install_width = install_bbox[2] - install_bbox[0]
                install_x = (768 - install_width) // 2
                draw.text((install_x, 980), install_note, fill='darkgreen', font=text_font)
            
            img.save(output_path, quality=95)
            logger.info(f"✅ Generated enhanced placeholder image: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Error creating placeholder: {e}")
            # Create a simple fallback image
            img = Image.new('RGB', (self.width, self.height), color='lightblue')
            draw = ImageDraw.Draw(img)
            draw.text((self.width // 2, self.height // 2), f"Scene: {prompt}", fill='black', anchor='mm')
            img.save(output_path)
            return output_path
    
    def is_sd_available(self) -> bool:
        """Check if Stable Diffusion is available."""
        return self.sd_available and self.pipe is not None

    def _is_image_blank_or_poor_quality(self, image_path: str) -> bool:
        """
        Check if an image is blank, mostly empty, or of poor quality.
        Returns True if the image should be regenerated with an adjusted prompt.
        """
        try:
            from PIL import Image, ImageStat
            import numpy as np
            
            # Load the image
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array for analysis
            img_array = np.array(image)
            
            # Check 1: Variance analysis (blank images have low variance)
            stat = ImageStat.Stat(image)
            variance = np.var(img_array)
            
            # Check 2: Check if image is mostly one color (blank/empty)
            unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
            
            # Check 3: Check brightness distribution
            gray = image.convert('L')
            gray_array = np.array(gray)
            brightness_variance = np.var(gray_array)
            
            # Check 4: Check for extreme brightness (all white or all black)
            mean_brightness = np.mean(gray_array)
            
            # Define thresholds
            is_blank = (
                variance < 1000 or  # Very low variance indicates blank image
                unique_colors < 100 or  # Very few unique colors
                brightness_variance < 500 or  # Low brightness variance
                mean_brightness < 10 or  # Too dark
                mean_brightness > 245  # Too bright
            )
            
            if is_blank:
                logger.warning(f"⚠️ Image detected as blank/poor quality: {image_path}")
                logger.info(f"   Variance: {variance:.1f}, Unique colors: {unique_colors}, Brightness: {mean_brightness:.1f}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error analyzing image quality: {e}")
            # If we can't analyze, assume it's fine
            return False

    def _adjust_visual_prompt_for_blank_image(self, original_prompt: str, attempt: int = 1) -> str:
        """
        Adjust the visual prompt using GPT-2 to fix blank image issues.
        Returns an enhanced prompt that should produce better results.
        """
        try:
            if not self.prompt_enhancer or not self.prompt_enhancer.is_available():
                logger.warning("⚠️ Prompt enhancer not available, using fallback adjustments")
                return self._fallback_prompt_adjustment(original_prompt, attempt)
            
            # Create specific adjustment prompts based on attempt number
            if attempt == 1:
                adjustment_prompt = f"Enhance this visual prompt to create a detailed, realistic scene with clear subjects and natural lighting: {original_prompt}"
            elif attempt == 2:
                adjustment_prompt = f"Transform this prompt into a highly detailed, realistic scene with strong visual elements and clear composition: {original_prompt}"
            else:
                adjustment_prompt = f"Create an extremely detailed, realistic scene with multiple visual elements, natural colors, and clear subjects: {original_prompt}"
            
            logger.info(f"🎯 Adjusting prompt (attempt {attempt}): {original_prompt}")
            
            # Use GPT-2 to enhance the prompt
            enhanced_prompt = self.prompt_enhancer.enhance_prompt(
                adjustment_prompt,
                enhancement_type="realistic_detailed",
                max_tokens=77  # Keep within diffusion model limits
            )
            
            # Add specific cartoon enhancement keywords if not present
            enhancement_keywords = [
                "natural colors", "detailed", "clear composition", 
                "rich textures", "natural lighting", "distinct subjects", "photorealistic"
            ]
            
            # Check if any enhancement keywords are missing
            missing_keywords = [kw for kw in enhancement_keywords if kw.lower() not in enhanced_prompt.lower()]
            
            if missing_keywords and attempt <= 2:
                # Add missing keywords
                additional_enhancement = ", ".join(missing_keywords[:3])  # Limit to 3 keywords
                enhanced_prompt = f"{enhanced_prompt}, {additional_enhancement}"
                
                # Ensure we stay within token limits
                enhanced_prompt = self.prompt_enhancer._limit_tokens(enhanced_prompt, 77)
            
            logger.info(f"🎯 Enhanced prompt: {enhanced_prompt}")
            return enhanced_prompt
            
        except Exception as e:
            logger.error(f"Error adjusting prompt with GPT-2: {e}")
            return self._fallback_prompt_adjustment(original_prompt, attempt)

    def _fallback_prompt_adjustment(self, original_prompt: str, attempt: int) -> str:
        """
        Fallback prompt adjustment when GPT-2 is not available.
        """
        base_enhancements = [
            "photorealistic, detailed, natural lighting",
            "realistic scene, rich details, clear subjects",
            "highly detailed, natural colors, strong composition"
        ]
        
        enhancement = base_enhancements[min(attempt - 1, len(base_enhancements) - 1)]
        adjusted_prompt = f"{original_prompt}, {enhancement}"
        
        logger.info(f"🎯 Fallback adjusted prompt: {adjusted_prompt}")
        return adjusted_prompt

    def generate_cartoon_image_with_validation(self, prompt: str, output_path: str, max_attempts: int = 3, negative_prompt: str = None, character_faces: dict = None) -> str:
        """
        Generate an image with validation and automatic prompt adjustment (legacy method).
        Retries with adjusted prompts if the generated image is blank or poor quality.
        """
        original_prompt = prompt
        
        for attempt in range(1, max_attempts + 1):
            logger.info(f"🎨 Generating image (attempt {attempt}/{max_attempts})")
            
            # Generate the image
            result_path = self.generate_cartoon_image(prompt, output_path, negative_prompt=negative_prompt, character_faces=character_faces)
            
            # Validate the generated image
            if not self._is_image_blank_or_poor_quality(result_path):
                logger.info(f"✅ Image validation passed on attempt {attempt}")
                return result_path
            
            logger.warning(f"⚠️ Image validation failed on attempt {attempt}")
            
            # If this is not the last attempt, adjust the prompt and try again
            if attempt < max_attempts:
                logger.info(f"🔄 Adjusting prompt for attempt {attempt + 1}")
                prompt = self._adjust_visual_prompt_for_blank_image(original_prompt, attempt)
                
                # Create a new output path for this attempt
                base_path = Path(output_path)
                new_output_path = base_path.parent / f"{base_path.stem}_attempt_{attempt + 1}{base_path.suffix}"
                output_path = str(new_output_path)
            else:
                logger.warning(f"⚠️ All {max_attempts} attempts failed. Using best available image.")
                return result_path
        
        return output_path
