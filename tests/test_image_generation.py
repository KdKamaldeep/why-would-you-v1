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

def test_image_generation(prompt=None):
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
    
    # Use provided prompt or default test prompt
    if prompt is None:
        prompt = "(wide shot, cartoon brown monkey wearing red scarf, cartoon brown bear in blue vest, cartoon gray squirrel with green bow sleeping peacefully under tree:1.3), (calm jungle night with stars and grass:1.2), (soft blue moonlight, serene storybook illustration style:1.1)"
    
    test_prompts =[
  "(wide shot) (full scene view) foreground: [Misthi, 9, Indian girl, green salwar kameez, braid with marigold, brass anklet, barefoot:1.4], midground: [calm river, small fishing boat:1.0], background: [distant crumbling fort on hill, warm evening sky:0.9], storybook magical style, soft hand-painted, natural proportions (wide shot)",
  "(wide shot) (full scene view) foreground: [Misthi listening to elders on charpoy:1.4], midground: [clay lamps glowing:1.0], background: [fort silhouette under darkening sky:0.9], storybook magical style, soft hand-painted, natural proportions (wide shot)",
  "(wide shot) (full scene view) foreground: [Misthi holding brass lantern:1.4], midground: [marigold garlands, rangoli at doorsteps:1.0], background: [dirt path to shadowy fort:0.9], storybook magical style, soft hand-painted, natural proportions (wide shot)",
  "(wide shot) (full scene view) foreground: [Misthi brushing tall grass:1.4], midground: [mossy carved stones, glowing fireflies:1.1], background: [dense forest canopy under moonlight:0.9], storybook magical style, soft hand-painted, natural proportions (wide shot)",
  "(wide shot) (full scene view) foreground: [Misthi holding lantern at fort gate:1.4], midground: [carved lock with river and moon:1.1], background: [tall stone walls with vines under moonlight:0.9], storybook magical style, soft hand-painted, natural proportions (wide shot)",
  "(wide shot) (full scene view) foreground: [Misthi leaning over stone well:1.4], midground: [ripples reflecting starlight:1.1], background: [broken archways, banyan branches:0.9], storybook magical style, soft hand-painted, natural proportions (wide shot)",
  "(wide shot) (full scene view) foreground: [Misthi climbing rope ladder:1.4], midground: [torch-lit carved stone walls:1.1], background: [bronze chest in chamber center:0.9], storybook magical style, soft hand-painted, natural proportions (wide shot)",
  "(wide shot) (full scene view) foreground: [Misthi lifting silver-blue flute:1.4], midground: [glowing water rising:1.1], background: [faint river outline above:0.9], storybook magical style, soft hand-painted, natural proportions (wide shot)",
  "(wide shot) (full scene view) foreground: [Misthi playing glowing flute:1.4], midground: [cheering villagers with water pots:1.1], background: [green fields, flowing river:0.9], storybook magical style, soft hand-painted, natural proportions (wide shot)"
]


    print(f"\n🎬 Generating {len(test_prompts)} test images...")
    print(f"📝 Using prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    
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
        """
    )
    
    parser.add_argument(
        '-p', '--prompt',
        type=str,
        help='Custom prompt for image generation (optional)',
        default=None
    )
    
    args = parser.parse_args()
    
    # Run the test with the provided prompt
    test_image_generation(args.prompt)

if __name__ == "__main__":
    main()
