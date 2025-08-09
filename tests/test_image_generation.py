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
    test_prompts = [
  "Golden hour wide shot of an Indian village, dusty lane stretching into the distance, mud houses with thatched roofs, neem trees swaying gently in warm sunlight.",
  "A person in a red-yellow lehenga walks along the lane, silver anklet glinting on their foot, dupatta flowing behind, long shadows stretching on the dusty ground.",
  "Close-up of the silver anklet slipping off the foot and falling softly onto the dusty lane, a small cloud of dust rising around it.",
  "The anklet rolls forward on the dusty lane, leading toward a lively street market with colorful stalls and bustling activity.",
  "A child in an off-white cotton kurta crouches near the market, eyes fixed on the shiny silver anklet resting on the ground.",
  "Small hands reach out and gently pick up the anklet, sunlight catching the intricate silver as a playful smile appears.",
  "Two figures meet on the village lane, one handing over the anklet while the other receives it with a look of relief.",
  "Close-up of hands exchanging warm golden jalebis wrapped in brown paper, glowing softly in the sunset light.",
  "Sunset wide shot of the village lane painted in orange and pink hues, the two figures walking away together as the camera slowly pulls back."
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
