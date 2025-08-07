#!/usr/bin/env python3
"""
Image Generator Module - Handles cartoon image generation using Stable Diffusion
"""

import time
import logging
import torch
from PIL import Image, ImageDraw, ImageFont
from typing import List

logger = logging.getLogger(__name__)

class ImageGenerator:
    """Handles cartoon image generation using local Stable Diffusion."""
    
    def __init__(self, model_path: str = "models/toonyou_beta6.safetensors"):
        self.model_path = model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def generate_cartoon_image(self, prompt: str, output_path: str) -> str:
        """Generate a cartoon-style image using local Stable Diffusion."""
        try:
            # This is a simplified version - in practice you'd need the full SD implementation
            # For now, we'll use a placeholder that simulates the process
            logger.info(f"Generating cartoon image for prompt: {prompt}")
            
            # Simulate SD generation time
            time.sleep(2)
            
            # Create a placeholder image with the prompt
            return self._generate_placeholder_image(prompt, output_path)
            
        except Exception as e:
            logger.error(f"Error generating image: {e}")
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
        """Generate a simple placeholder image if SD fails."""
        img = Image.new('RGB', (768, 1024), color='lightblue')
        draw = ImageDraw.Draw(img)
        
        # Add text
        try:
            font = ImageFont.truetype("arial.ttf", 32)
        except:
            font = ImageFont.load_default()
        
        text = f"Cartoon Scene:\n{prompt[:100]}..."
        lines = text.split('\n')
        
        y_offset = 100
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            x = (768 - text_width) // 2
            draw.text((x, y_offset), line, fill='black', font=font)
            y_offset += 50
        
        img.save(output_path)
        logger.info(f"Generated placeholder image: {output_path}")
        return output_path
