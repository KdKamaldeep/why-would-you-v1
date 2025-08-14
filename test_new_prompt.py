#!/usr/bin/env python3
"""
Test script with the new optimized prompt
"""

import sys
import os
sys.path.append('src')

from src.core.image_generator import ImageGenerator

def test_new_prompt():
    """Test image generation with the new optimized prompt."""
    
    # Initialize image generator with 16:9 dimensions
    image_gen = ImageGenerator(
        width=1920,
        height=1080,
        enable_prompt_enhancement=False
    )
    
    # Test with the new prompt
    print("Test: New optimized prompt")
    result = image_gen.generate_cartoon_image(
        prompt="(wide shot:1.2) little Indian girl with long black hair tied in two braids, wearing pink frock, sitting near a pond under full moonlight, soft ripples in water, lotus flowers, storybook style:1.3",
        output_path="test_new_prompt.png",
        negative_prompt="blurry, low resolution, text, watermark"
    )
    print(f"Result: {result}")
    
    # Test 2: Even simpler version without weights
    print("\nTest 2: Simple version without weights")
    result2 = image_gen.generate_cartoon_image(
        prompt="little Indian girl with long black hair in braids, wearing pink dress, sitting by pond under moonlight, lotus flowers, storybook style",
        output_path="test_new_prompt_simple.png",
        negative_prompt="blurry, low resolution, text, watermark"
    )
    print(f"Result 2: {result2}")

if __name__ == "__main__":
    test_new_prompt()
