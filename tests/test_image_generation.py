#!/usr/bin/env python3
"""
Test script for Stable Diffusion image generation
"""

import os
import logging
from pathlib import Path
import sys
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.image_generator import ImageGenerator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_image_generation(prompt=None, negative_prompt=None, video_format="shorts"):
    """Test the image generation system."""
    
    print("🎨 Testing Stable Diffusion Image Generation")
    print("=" * 50)
    
    # Set dimensions based on video format
    if video_format.lower() == "shorts":
        width, height = 768, 1024  # 9:16 aspect ratio
        print(f"📐 Using YouTube Shorts format: {width}x{height} (9:16)")
    elif video_format.lower() == "normal":
        width, height = 1920, 1080  # 16:9 aspect ratio
        print(f"📐 Using Normal video format: {width}x{height} (16:9)")
    else:
        width, height = 768, 1024  # Default to shorts
        print(f"📐 Using default format: {width}x{height}")
    
    # Create output directory
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize image generator with specified dimensions
    print("🚀 Initializing Image Generator...")
    image_gen = ImageGenerator(width=width, height=height)
    
    # Check if SD is available
    if image_gen.is_sd_available():
        print("✅ Stable Diffusion is available!")
        print(f"💻 Device: {image_gen.device}")
        print(f"📁 Model: {image_gen.model_path}")
    else:
        print("⚠️ Stable Diffusion not available - will use placeholder images")
        print("💡 To enable SD, run: bash download_models.sh")
    
    # Use provided prompt or default test prompt (optimized to fit within 77 tokens)
    if prompt is None:
        prompt = "(wide shot:1.2) single little Indian girl with long black hair tied in two braids, wearing pink frock, sitting alone near a pond under full moonlight, soft ripples in water, lotus flowers, storybook style:1.3"
    
    test_prompts = [prompt]
    test_negative_prompts = [negative_prompt] if negative_prompt else None



    print(f"\n🎬 Generating {len(test_prompts)} test images...")
    print(f"📝 Using prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    if negative_prompt:
        print(f"🚫 Using negative prompt: {negative_prompt[:100]}{'...' if len(negative_prompt) > 100 else ''}")
    
    # Generate images
    image_paths = image_gen.generate_multiple_images(test_prompts, str(output_dir), negative_prompts=test_negative_prompts)
    
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

def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test Stable Diffusion image generation with custom prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_image_generation.py
  python test_image_generation.py --prompt "a cute cartoon cat playing in a garden"
  python test_image_generation.py -p "cartoon style, colorful background, happy characters"
  python test_image_generation.py --negative-prompt "dark, scary, realistic"
  python test_image_generation.py --video-format normal --prompt "wide shot of cartoon characters"
  python test_image_generation.py -f shorts -p "vertical cartoon scene" -n "crowded, busy background"
        """
    )
    
    parser.add_argument(
        '-p', '--prompt',
        type=str,
        help='Custom prompt for image generation (optional)',
        default=None
    )
    
    parser.add_argument(
        '-n', '--negative-prompt',
        type=str,
        help='Negative prompt to avoid certain elements (optional)',
        default=None
    )
    
    parser.add_argument(
        '-f', '--video-format',
        choices=['shorts', 'normal'],
        default='shorts',
        help='Video format: "shorts" for 9:16 YouTube Shorts, "normal" for 16:9 standard videos'
    )
    
    args = parser.parse_args()
    
    # Run the test with the provided prompt, negative prompt, and video format
    test_image_generation(args.prompt, args.negative_prompt, args.video_format)

if __name__ == "__main__":
    main()
