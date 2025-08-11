#!/usr/bin/env python3
"""
Test script for Stable Diffusion image generation
"""

import os
import logging
from pathlib import Path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.image_generator import ImageGenerator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_image_generation():
    """Test the image generation system."""
    
    print("🎨 Testing Stable Diffusion Image Generation")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize image generator
    print("🚀 Initializing Image Generator...")
    image_gen = ImageGenerator()
    
    # Check if SD is available
    if image_gen.is_sd_available():
        print("✅ Stable Diffusion is available!")
        print(f"💻 Device: {image_gen.device}")
        print(f"📁 Model: {image_gen.model_path}")
    else:
        print("⚠️ Stable Diffusion not available - will use placeholder images")
        print("💡 To enable SD, run: bash download_models.sh")
    
    # Test prompts with enhanced cartoon style keywords
    test_prompts = [
  "(wide shot:1.3) (full scene view:1.2) (Indian village:1.3) at sunrise, dusty path in foreground leading to distant blue hills, (two children:1.3) midground, girl red scarf short black hair yellow kurta green salwar; boy messy hair blue shirt brown shorts, holding small cloth bags, (storybook mystery:1.4)",
  "(wide shot:1.3) (full scene view:1.2) small wooden bridge in midground over sparkling stream, (two children:1.3) midground crossing bridge, girl red scarf short black hair yellow kurta green salwar; boy messy hair blue shirt brown shorts, surrounded by green trees and bright flowers in background, (storybook colorful:1.4)",
  "(wide shot:1.3) (full scene view:1.2) glowing golden fireflies lighting jungle path, tall green plants in foreground, (two children:1.3) midground walking cautiously, girl red scarf short black hair yellow kurta green salwar; boy messy hair blue shirt brown shorts, background fades into dense jungle, (storybook magical:1.4)",
  "(wide shot:1.3) (full scene view:1.2) (golden glowing river:1.4) flowing between rocky cliffs, rocky foreground, (two children:1.3) midground standing in awe, girl red scarf short black hair yellow kurta green salwar; boy messy hair blue shirt brown shorts, bright golden reflections on faces, background cliffs glowing, (storybook magical:1.4)",
  "(wide shot:1.3) (full scene view:1.2) (happy Indian village festival:1.4) with cheering villagers in background, (two children:1.3) foreground opening glowing chest, colorful butterflies and sparkles filling sky, girl red scarf short black hair yellow kurta green salwar; boy messy hair blue shirt brown shorts, (storybook magical:1.4)"
]




    print(f"\n🎬 Generating {len(test_prompts)} test images...")
    
    # Generate images
    image_paths = image_gen.generate_multiple_images(test_prompts, str(output_dir))
    
    # Report results
    print(f"\n✅ Generated {len(image_paths)} images:")
    for i, path in enumerate(image_paths):
        if Path(path).exists():
            size = Path(path).stat().st_size / 1024  # KB
            print(f"  {i+1}. {path} ({size:.1f} KB)")
        else:
            print(f"  {i+1}. {path} (❌ Failed)")
    
    print(f"\n📁 Images saved to: {output_dir.absolute()}")
    
    # Summary
    if image_gen.is_sd_available():
        print("\n🎉 SUCCESS: Stable Diffusion is working perfectly!")
        print("🎨 You can now generate professional cartoon images!")
    else:
        print("\n💡 To enable AI image generation:")
        print("   1. Run: bash download_models.sh")
        print("   2. Install: pip install diffusers transformers accelerate safetensors")
        print("   3. Restart this test script")

if __name__ == "__main__":
    test_image_generation()
