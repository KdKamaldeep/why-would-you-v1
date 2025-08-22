#!/usr/bin/env python3
"""
Test script for Cartoon Style Image Generation
"""

import os
import logging
import argparse
from pathlib import Path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.image_generator import ImageGenerator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_image_generation(style="cartoon"):
    """Test image generation with specified style (cartoon or realistic)."""
    
    print(f"🎨 Testing {style.title()} Style Image Generation")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path(f"{style}_test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize image generator with style-optimized settings
    print(f"🚀 Initializing {style.title()} Image Generator...")
    
    # Model selection based on style
    if style == "cartoon":
        # Cartoon models
        style_models = [
            "models/toonyou_beta6.safetensors",
            "models/anything-v4.5.safetensors", 
            "models/counterfeit-v3.0.safetensors"
        ]
        default_repo = "runwayml/stable-diffusion-v1-5"
    else:  # realistic
        # Realistic models
        style_models = [
            "models/realistic-vision-v5.1.safetensors",
            "models/dreamshaper-v8.safetensors",
            "models/deliberate-v3.safetensors"
        ]
        default_repo = "runwayml/stable-diffusion-v1-5"
    
    model_path = None
    for model in style_models:
        if Path(model).exists():
            model_path = model
            print(f"✅ Found {style} model: {model}")
            break
    
    if not model_path:
        print(f"⚠️ No {style} model found, will use default with {style} prompts")
        # Set environment variable to use a style-friendly model
        os.environ["SD_REPO_ID"] = default_repo
    
    image_gen = ImageGenerator(model_path=model_path) if model_path else ImageGenerator()
    
    # Check if SD is available
    if image_gen.is_sd_available():
        print("✅ Stable Diffusion is available!")
        print(f"💻 Device: {image_gen.device}")
        print(f"📁 Model: {image_gen.model_path}")
    else:
        print("⚠️ Stable Diffusion not available - will use placeholder images")
        print("💡 To enable SD, run: bash download_models.sh")
    
    # Style-optimized test prompts
    if style == "cartoon":
        test_prompts = [
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
    else:  # realistic
        test_prompts = [
            "photorealistic, high quality, detailed, Indian village at golden hour, dusty lane, traditional mud houses with thatched roofs, neem trees, warm orange sky, natural lighting",
            "photorealistic, high quality, detailed, Indian woman in traditional red-yellow lehenga, silver anklet, flowing dupatta, natural lighting, realistic textures",
            "photorealistic, high quality, detailed, silver anklet falling to ground, dust particles, realistic physics, natural lighting",
            "photorealistic, high quality, detailed, traditional Indian street market, colorful stalls, realistic textures, natural lighting, bustling atmosphere",
            "photorealistic, high quality, detailed, Indian child in traditional cotton kurta, surprised expression, natural lighting, realistic skin textures",
            "photorealistic, high quality, detailed, hands picking up silver anklet, realistic skin textures, natural lighting, detailed jewelry",
            "photorealistic, high quality, detailed, two people meeting, happy expressions, natural lighting, realistic clothing textures",
            "photorealistic, high quality, detailed, hands exchanging traditional sweets, crinkly paper, warm lighting, realistic textures",
            "photorealistic, high quality, detailed, peaceful sunset village lane, warm lighting, realistic textures, natural atmosphere"
        ]

    print(f"\n🎬 Generating {len(test_prompts)} {style} images...")
    
    # Generate images
    image_paths = image_gen.generate_multiple_images(test_prompts, str(output_dir))
    
    # Report results
    print(f"\n✅ Generated {len(image_paths)} {style} images:")
    for i, path in enumerate(image_paths):
        if Path(path).exists():
            size = Path(path).stat().st_size / 1024  # KB
            print(f"  {i+1}. {path} ({size:.1f} KB)")
        else:
            print(f"  {i+1}. {path} (❌ Failed)")
    
    print(f"\n📁 {style.title()} images saved to: {output_dir.absolute()}")
    
    # Summary
    if image_gen.is_sd_available():
        print(f"\n🎉 SUCCESS: {style.title()} image generation is working!")
        print(f"🎨 Generated professional {style} style images!")
    else:
        print("\n💡 To enable AI image generation:")
        print("   1. Run: bash download_models.sh")
        print("   2. Install: pip install diffusers transformers accelerate safetensors")
        print("   3. Restart this test script")

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Test image generation with different styles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test cartoon style (default)
  python test_cartoon_images.py
  
  # Test realistic style
  python test_cartoon_images.py --style realistic
  
  # Test both styles
  python test_cartoon_images.py --style cartoon
  python test_cartoon_images.py --style realistic
        """
    )
    
    parser.add_argument(
        "--style", "-s",
        choices=["cartoon", "realistic"],
        default="cartoon",
        help="Visual style to test (cartoon or realistic)"
    )
    
    args = parser.parse_args()
    
    # Run the test with specified style
    test_image_generation(args.style)

if __name__ == "__main__":
    main()
