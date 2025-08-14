#!/usr/bin/env python3
"""
Test script focusing on positive prompt for single character
"""

import sys
import os
sys.path.append('src')

from src.core.image_generator import ImageGenerator

def test_positive_focus():
    """Test image generation focusing on positive prompt for single character."""
    
    # Initialize image generator with 16:9 dimensions
    image_gen = ImageGenerator(
        width=1920,
        height=1080,
        enable_prompt_enhancement=False
    )
    
    # Test 1: Very clear single character prompt with minimal negative
    print("Test 1: Clear single character prompt")
    result1 = image_gen.generate_cartoon_image(
        prompt="(wide shot:1.2) one single Indian girl with black braids, pink dress, sitting alone by pond under moonlight, lotus flowers, storybook style:1.3",
        output_path="test_positive_1.png",
        negative_prompt="blurry, low resolution, text, watermark"
    )
    print(f"Result 1: {result1}")
    
    # Test 2: Emphasize solitude in prompt
    print("\nTest 2: Emphasize solitude")
    result2 = image_gen.generate_cartoon_image(
        prompt="(wide shot:1.2) solitary Indian girl with black braids, pink dress, sitting by herself near pond under moonlight, lotus flowers, storybook style:1.3",
        output_path="test_positive_2.png",
        negative_prompt="blurry, low resolution, text, watermark"
    )
    print(f"Result 2: {result2}")
    
    # Test 3: Focus on individual character
    print("\nTest 3: Focus on individual")
    result3 = image_gen.generate_cartoon_image(
        prompt="(wide shot:1.2) individual Indian girl with black braids, pink dress, sitting by pond under moonlight, lotus flowers, storybook style:1.3",
        output_path="test_positive_3.png",
        negative_prompt="blurry, low resolution, text, watermark"
    )
    print(f"Result 3: {result3}")
    
    # Test 4: Simple but clear about single character
    print("\nTest 4: Simple but clear")
    result4 = image_gen.generate_cartoon_image(
        prompt="one Indian girl with braids, pink dress, pond, moonlight, lotus, storybook style",
        output_path="test_positive_4.png",
        negative_prompt="blurry, low resolution, text, watermark"
    )
    print(f"Result 4: {result4}")

if __name__ == "__main__":
    test_positive_focus()
