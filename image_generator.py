#!/usr/bin/env python3
"""
Image Generator Module - Handles cartoon image generation using Stable Diffusion
"""

import os
import time
import logging
import torch
from PIL import Image, ImageDraw, ImageFont
from typing import List
from pathlib import Path

logger = logging.getLogger(__name__)

class ImageGenerator:
    """Handles cartoon image generation using Stable Diffusion."""
    
    def __init__(self, model_path: str = "models/toonyou_beta6.safetensors"):
        self.model_path = model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pipe = None
        self._initialize_pipeline()
        
    def _initialize_pipeline(self):
        """Initialize the Stable Diffusion pipeline."""
        try:
            # Try to import and initialize diffusers
            from diffusers import StableDiffusionPipeline
            import safetensors
            
            logger.info(f"Initializing Stable Diffusion pipeline...")
            logger.info(f"Device: {self.device}")
            logger.info(f"Model path: {self.model_path}")
            
            # Check if model file exists
            if not Path(self.model_path).exists():
                logger.warning(f"Model file not found: {self.model_path}")
                logger.info("Will use placeholder images instead")
                return
            
            # Load the pipeline with the custom model
            self.pipe = StableDiffusionPipeline.from_single_file(
                self.model_path,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            if self.device == 'cuda':
                self.pipe = self.pipe.to(self.device)
                self.pipe.enable_memory_efficient_attention()
            
            logger.info("✅ Stable Diffusion pipeline initialized successfully!")
            
        except ImportError as e:
            logger.warning(f"Diffusers not available: {e}")
            logger.info("Installing diffusers... Please wait.")
            try:
                os.system("pip install diffusers transformers accelerate safetensors")
                logger.info("Diffusers installed! Please restart the script.")
            except:
                pass
        except Exception as e:
            logger.warning(f"Failed to initialize SD pipeline: {e}")
            logger.info("Will use placeholder images instead")
    
    def generate_cartoon_image(self, prompt: str, output_path: str) -> str:
        """Generate a cartoon-style image using Stable Diffusion."""
        try:
            logger.info(f"Generating cartoon image for prompt: {prompt}")
            
            # If we have a working pipeline, use it
            if self.pipe is not None:
                return self._generate_sd_image(prompt, output_path)
            else:
                # Fallback to placeholder
                logger.info("Using placeholder image (SD pipeline not available)")
                return self._generate_placeholder_image(prompt, output_path)
                
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return self._generate_placeholder_image(prompt, output_path)
    
    def _generate_sd_image(self, prompt: str, output_path: str) -> str:
        """Generate image using Stable Diffusion."""
        try:
            # Enhanced prompt for better cartoon results
            enhanced_prompt = f"cartoon style, animated, colorful, cute, {prompt}, high quality, digital art, illustration"
            negative_prompt = "photorealistic, realistic, photo, blurry, low quality, dark, scary, violent"
            
            # Generate image
            with torch.autocast(self.device):
                result = self.pipe(
                    prompt=enhanced_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    width=768,
                    height=1024,
                    num_images_per_prompt=1
                )
            
            # Save the image
            image = result.images[0]
            image.save(output_path)
            
            logger.info(f"✅ Generated SD image: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"SD generation failed: {e}")
            return self._generate_placeholder_image(prompt, output_path)
    
    def generate_multiple_images(self, prompts: List[str], output_dir: str) -> List[str]:
        """Generate multiple cartoon images for a list of prompts."""
        image_paths = []
        for i, prompt in enumerate(prompts):
            output_path = f"{output_dir}/scene_{i+1}.png"
            image_path = self.generate_cartoon_image(prompt, output_path)
            image_paths.append(image_path)
        return image_paths
    
    def _generate_placeholder_image(self, prompt: str, output_path: str) -> str:
        """Generate a colorful placeholder image with better design."""
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
        
        img.save(output_path)
        logger.info(f"Generated enhanced placeholder image: {output_path}")
        return output_path
