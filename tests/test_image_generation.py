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
  "(Indian village:1.3) at misty dawn, soft fog over green fields, (two children:1.3) small Indian girl, red scarf, short black hair, wearing yellow kurta and green salwar, small Indian boy, messy hair, wearing blue shirt and brown shorts, holding lanterns, looking toward distant hills, (storybook mystery adventure style:1.4), wide shot",
  "(two children:1.3) small Indian girl, red scarf, short black hair, wearing yellow kurta and green salwar, small Indian boy, messy hair, wearing blue shirt and brown shorts, walking on overgrown muddy path with faint footprints, tall grass on both sides, (storybook mystery style:1.4), soft light, wide shot",
  "(two children:1.3) small Indian girl, red scarf, short black hair, wearing yellow kurta and green salwar, small Indian boy, messy hair, wearing blue shirt and brown shorts, standing at entrance of rocky cave hidden behind waterfall, golden light glowing inside, (storybook magical mystery style:1.4), wide shot",
  "(ancient chest:1.4) covered in vines with glowing mysterious symbols, (two children:1.3) small Indian girl, red scarf, short black hair, wearing yellow kurta and green salwar, small Indian boy, messy hair, wearing blue shirt and brown shorts, reaching toward chest, (storybook magical style:1.4), warm glow, wide shot",
  "(happy Indian village festival:1.4) with cheering villagers, (two children:1.3) small Indian girl, red scarf, short black hair, wearing yellow kurta and green salwar, small Indian boy, messy hair, wearing blue shirt and brown shorts, opening glowing ancient chest, hundreds of colorful glowing butterflies flying into the sky, sparkles everywhere, (storybook magical style:1.4), golden evening light, wide shot"
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
