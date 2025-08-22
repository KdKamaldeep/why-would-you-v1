#!/usr/bin/env python3
"""
Test script for Cartoon Style Image Generation
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

def test_cartoon_generation():
    """Test cartoon image generation with optimized settings."""
    
    print("🎨 Testing Cartoon Style Image Generation")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path("cartoon_test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize image generator with cartoon-optimized settings
    print("🚀 Initializing Cartoon Image Generator...")
    
    # Try to use a cartoon model if available
    cartoon_models = [
        "models/toonyou_beta6.safetensors",
        "models/anything-v4.5.safetensors", 
        "models/counterfeit-v3.0.safetensors"
    ]
    
    model_path = None
    for model in cartoon_models:
        if Path(model).exists():
            model_path = model
            print(f"✅ Found cartoon model: {model}")
            break
    
    if not model_path:
        print("⚠️ No cartoon model found, will use default with cartoon prompts")
        # Set environment variable to use a cartoon-friendly model
        os.environ["SD_REPO_ID"] = "runwayml/stable-diffusion-v1-5"
    
    image_gen = ImageGenerator(model_path=model_path) if model_path else ImageGenerator()
    
    # Check if SD is available
    if image_gen.is_sd_available():
        print("✅ Stable Diffusion is available!")
        print(f"💻 Device: {image_gen.device}")
        print(f"📁 Model: {image_gen.model_path}")
    else:
        print("⚠️ Stable Diffusion not available - will use placeholder images")
        print("💡 To enable SD, run: bash download_models.sh")
    
    # Cartoon-optimized test prompts
    cartoon_prompts = [
        "cartoon illustration, 2D animation style, flat colors, simple shapes, bright colorful Indian village at golden hour, dusty lane, cheerful mud houses with thatched roofs, neem trees, glowing orange sky",
        "cartoon illustration, 2D animation style, flat colors, simple shapes, smiling cartoon character in vibrant red-yellow lehenga, shiny silver anklet, flowing dupatta, playful shadows",
        "cartoon illustration, 2D animation style, flat colors, simple shapes, silver anklet bouncing off foot, tiny puff of dust shaped like cloud, cute animation",
        "cartoon illustration, 2D animation style, flat colors, simple shapes, anklet rolling toward lively cartoon street market, colorful stalls, big signs, bustling friendly characters",
        "cartoon illustration, 2D animation style, flat colors, simple shapes, curious cartoon child in off-white cotton kurta, wide-eyed surprise, twinkling anklet",
        "cartoon illustration, 2D animation style, flat colors, simple shapes, small cartoon hands picking up shiny anklet, silver pattern sparkling, playful smile",
        "cartoon illustration, 2D animation style, flat colors, simple shapes, two cheerful cartoon characters meeting, happy gesture, relieved grin",
        "cartoon illustration, 2D animation style, flat colors, simple shapes, cartoon hands exchanging golden sweets, crinkly brown paper, warm sunset light",
        "cartoon illustration, 2D animation style, flat colors, simple shapes, peaceful sunset village lane, bright oranges and pinks, characters walking away"
    ]

    print(f"\n🎬 Generating {len(cartoon_prompts)} cartoon images...")
    
    # Generate images
    image_paths = image_gen.generate_multiple_images(cartoon_prompts, str(output_dir))
    
    # Report results
    print(f"\n✅ Generated {len(image_paths)} cartoon images:")
    for i, path in enumerate(image_paths):
        if Path(path).exists():
            size = Path(path).stat().st_size / 1024  # KB
            print(f"  {i+1}. {path} ({size:.1f} KB)")
        else:
            print(f"  {i+1}. {path} (❌ Failed)")
    
    print(f"\n📁 Cartoon images saved to: {output_dir.absolute()}")
    
    # Summary
    if image_gen.is_sd_available():
        print("\n🎉 SUCCESS: Cartoon image generation is working!")
        print("🎨 Generated professional cartoon style images!")
    else:
        print("\n💡 To enable AI cartoon generation:")
        print("   1. Run: bash download_models.sh")
        print("   2. Install: pip install diffusers transformers accelerate safetensors")
        print("   3. Restart this test script")

if __name__ == "__main__":
    test_cartoon_generation()
