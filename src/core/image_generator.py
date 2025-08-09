#!/usr/bin/env python3
"""
Image Generator Module - Handles cartoon image generation using Stable Diffusion
"""

import os
import time
import logging
import torch
from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class ImageGenerator:
    """Handles cartoon image generation using Stable Diffusion.

    Supports optional LoRA for style adaptation.
    """
    
    def __init__(self, model_path: str = "models/toonyou_beta6.safetensors", lora_path: Optional[str] = None, lora_scale: float = 0.8):
        self.model_path = model_path
        self.lora_path = lora_path
        self.lora_scale = lora_scale
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pipe = None
        self.sd_available = False
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

            self.sd_available = True
            logger.info("✅ Stable Diffusion pipeline initialized successfully!")
            logger.info("🎨 Ready to generate professional cartoon images!")

        except Exception as e:
            logger.warning(f"❌ Failed to initialize SD pipeline: {e}")
            logger.info("💡 Will use placeholder images instead")
            self.sd_available = False
    
    def generate_cartoon_image(self, prompt: str, output_path: str) -> str:
        """Generate a cartoon-style image using Stable Diffusion."""
        try:
            logger.info(f"🎨 Generating cartoon image for prompt: {prompt}")
            
            # If we have a working pipeline, use it
            if self.sd_available and self.pipe is not None:
                return self._generate_sd_image(prompt, output_path)
            else:
                # Fallback to placeholder
                logger.info("⚠️ Using placeholder image (SD pipeline not available)")
                return self._generate_placeholder_image(prompt, output_path)
                
        except Exception as e:
            logger.error(f"❌ Error generating image: {e}")
            return self._generate_placeholder_image(prompt, output_path)
    
    def _generate_sd_image(self, prompt: str, output_path: str) -> str:
        """Generate image using Stable Diffusion."""
        try:
            # Enhanced prompt for better cartoon results
            enhanced_prompt = (
                f"cartoon style, storybook illustration, matte shading, soft outlines, {prompt}, "
                f"high quality, digital art, vibrant colors, clean lines"
            )
            # Strengthen anti-anime bias for Indian style intent
            negative_prompt = (
                "photorealistic, realistic, photo, 3d render, cgi, anime, manga, "
                "blurry, low quality, dark, scary, violent, adult content, nsfw"
            )
            
            logger.info(f"🎯 Enhanced prompt: {enhanced_prompt}")
            
            # Generate image with optimized settings
            with torch.autocast(self.device):
                result = self.pipe(
                    prompt=enhanced_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=25,  # More steps for better quality
                    guidance_scale=8.0,      # Slightly higher for better adherence
                    width=768,
                    height=1024,
                    num_images_per_prompt=1,
                    generator=torch.Generator(device=self.device).manual_seed(42)  # Consistent results
                )
            
            # Save the image
            image = result.images[0]
            image.save(output_path, quality=95)
            
            logger.info(f"✅ Generated professional SD image: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ SD generation failed: {e}")
            logger.info("🔄 Falling back to placeholder image")
            return self._generate_placeholder_image(prompt, output_path)
    
    def generate_multiple_images(self, prompts: List[str], output_dir: str) -> List[str]:
        """Generate multiple cartoon images for a list of prompts."""
        image_paths = []
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for i, prompt in enumerate(prompts):
            logger.info(f"🎬 Generating scene {i+1}/{len(prompts)}")
            output_path = f"{output_dir}/scene_{i+1}.png"
            image_path = self.generate_cartoon_image(prompt, output_path)
            image_paths.append(image_path)
            
            # Small delay between generations to prevent memory issues
            if i < len(prompts) - 1:
                time.sleep(1)
        
        logger.info(f"✅ Generated {len(image_paths)} images successfully!")
        return image_paths
    
    def _generate_placeholder_image(self, prompt: str, output_path: str) -> str:
        """Generate a colorful placeholder image with better design."""
        try:
            # Create a gradient background instead of solid blue
            img = Image.new('RGB', (768, 1024), color='white')
            draw = ImageDraw.Draw(img)
            
            # Create a colorful gradient background
            for y in range(1024):
                color_r = int(135 + (y / 1024) * 120)  # 135-255
                color_g = int(206 + (y / 1024) * 49)   # 206-255  
                color_b = int(250 - (y / 1024) * 50)   # 250-200
                color = (min(255, color_r), min(255, color_g), min(255, color_b))
                draw.line([(0, y), (768, y)], fill=color)
            
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
            title = "🎬 Cartoon Scene"
            title_bbox = draw.textbbox((0, 0), title, font=title_font)
            title_width = title_bbox[2] - title_bbox[0]
            title_x = (768 - title_width) // 2
            
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
                if bbox[2] - bbox[0] > 600:  # Max width
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
            
            y_offset = 400
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=text_font)
                text_width = bbox[2] - bbox[0]
                x = (768 - text_width) // 2
                
                # Add text shadow
                draw.text((x + 1, y_offset + 1), line, fill='gray', font=text_font)
                draw.text((x, y_offset), line, fill='darkblue', font=text_font)
                y_offset += 35
            
            # Add a note about the placeholder
            note = "⚠️ Placeholder - Install diffusers for AI images"
            note_bbox = draw.textbbox((0, 0), note, font=text_font)
            note_width = note_bbox[2] - note_bbox[0]
            note_x = (768 - note_width) // 2
            draw.text((note_x, 750), note, fill='red', font=text_font)
            
            # Add installation instructions
            install_note = "Run: pip install diffusers transformers accelerate safetensors"
            install_bbox = draw.textbbox((0, 0), install_note, font=text_font)
            install_width = install_bbox[2] - install_bbox[0]
            install_x = (768 - install_width) // 2
            draw.text((install_x, 780), install_note, fill='darkgreen', font=text_font)
            
            img.save(output_path, quality=95)
            logger.info(f"✅ Generated enhanced placeholder image: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Error creating placeholder: {e}")
            # Create a simple fallback image
            img = Image.new('RGB', (768, 1024), color='lightblue')
            draw = ImageDraw.Draw(img)
            draw.text((384, 512), f"Scene: {prompt}", fill='black', anchor='mm')
            img.save(output_path)
            return output_path
    
    def is_sd_available(self) -> bool:
        """Check if Stable Diffusion is available."""
        return self.sd_available and self.pipe is not None
