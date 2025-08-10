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
    
    # Test prompts
    test_prompts =[
  "Wide shot – Bright, colorful cartoon-style Indian village at golden hour, dusty lane stretching into the distance, cheerful mud houses with thatched roofs, neem trees swaying gently under a glowing orange sky.",
  "Tracking shot – A smiling character in a vibrant red-yellow lehenga walks along the lane, shiny silver anklet sparkling on their foot, flowing dupatta fluttering like a ribbon, long playful shadows on the ground.",
  "Low angle close-up – The silver anklet slips off the foot in a cute bounce, landing softly on the dusty lane with a tiny puff of dust shaped like a cloud.",
  "Tracking close-up – The anklet rolls slowly and wobbles along the lane toward a lively cartoon street market with colorful stalls, big signs, and bustling friendly characters.",
  "Medium shot – A curious child in an off-white cotton kurta crouches near the anklet, wide-eyed with surprise as it twinkles in the sunlight.",
  "Close-up – The child’s small hands carefully pick up the shiny anklet, its silver pattern sparkling with magical glints, a playful smile spreading on their face.",
  "Medium shot – Later on the lane, two cheerful characters meet; one offers the anklet with a happy gesture, and the other receives it with a big relieved grin.",
  "Close-up – Two pairs of cartoon hands exchanging spiral-shaped golden sweets wrapped in crinkly brown paper, glowing warmly in the soft sunset light.",
  "Wide shot – Peaceful sunset scene of the village lane painted in bright oranges and pinks, the two characters walking away side-by-side as the camera slowly zooms out."
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
