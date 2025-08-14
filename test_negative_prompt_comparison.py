#!/usr/bin/env python3
"""
Test script to compare negative prompt effectiveness
"""

import sys
import os
sys.path.append('src')

from src.core.image_generator import ImageGenerator

def test_negative_prompt_comparison():
    """Test image generation with and without negative prompts."""
    
    # Initialize image generator with 16:9 dimensions
    image_gen = ImageGenerator(
        width=1920,
        height=1080,
        enable_prompt_enhancement=False
    )
    
    base_prompt = "(wide shot:1.2) little Indian girl with long black hair tied in two braids, wearing pink frock, sitting near a pond under full moonlight, soft ripples in water, lotus flowers, storybook style:1.3"
    
    # Test 1: No negative prompt
    print("Test 1: No negative prompt")
    result1 = image_gen.generate_cartoon_image(
        prompt=base_prompt,
        output_path="test_no_negative.png",
        negative_prompt=None
    )
    print(f"Result 1: {result1}")
    
    # Test 2: Basic negative prompt
    print("\nTest 2: Basic negative prompt")
    result2 = image_gen.generate_cartoon_image(
        prompt=base_prompt,
        output_path="test_basic_negative.png",
        negative_prompt="blurry, low resolution, text, watermark"
    )
    print(f"Result 2: {result2}")
    
    # Test 3: Strong negative prompt
    print("\nTest 3: Strong negative prompt")
    result3 = image_gen.generate_cartoon_image(
        prompt=base_prompt,
        output_path="test_strong_negative.png",
        negative_prompt="extra humans, multiple people, crowded, group, blurry, low resolution, distorted, unnatural poses, text, watermark, dark edges, cropped limbs, chaotic background"
    )
    print(f"Result 3: {result3}")
    
    # Test 4: Very specific negative prompt
    print("\nTest 4: Very specific negative prompt")
    result4 = image_gen.generate_cartoon_image(
        prompt=base_prompt,
        output_path="test_specific_negative.png",
        negative_prompt="photorealistic, realistic, photo, 3d render, cgi, anime, manga, blurry, low quality, dark, scary, violent, adult content, nsfw, hyperrealistic, detailed textures, photographic, film grain, realistic lighting, realistic shadows, realistic proportions, detailed skin, detailed hair, detailed clothing textures"
    )
    print(f"Result 4: {result4}")

if __name__ == "__main__":
    test_negative_prompt_comparison()
