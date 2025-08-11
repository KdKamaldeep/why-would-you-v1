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
  "(Indian village:1.3) misty dawn, soft fog, green fields, (two children:1.3) girl red scarf short black hair yellow kurta green salwar; boy messy hair blue shirt brown shorts, holding lanterns, looking toward hills, (storybook mystery:1.4), wide shot",
  "(two children:1.3) girl red scarf short black hair yellow kurta green salwar; boy messy hair blue shirt brown shorts, walking on muddy path with faint footprints, tall grass, (storybook mystery:1.4), wide shot",
  "(two children:1.3) girl red scarf short black hair yellow kurta green salwar; boy messy hair blue shirt brown shorts, at rocky cave behind waterfall, golden glow inside, (storybook magical mystery:1.4), wide shot",
  "(ancient chest:1.4) vines glowing symbols, (two children:1.3) girl red scarf short black hair yellow kurta green salwar; boy messy hair blue shirt brown shorts, reaching toward chest, (storybook magical:1.4), wide shot",
  "(happy Indian village festival:1.4) cheering villagers, (two children:1.3) girl red scarf short black hair yellow kurta green salwar; boy messy hair blue shirt brown shorts, opening glowing chest, colorful butterflies sparkles, (storybook magical:1.4), wide shot"
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
