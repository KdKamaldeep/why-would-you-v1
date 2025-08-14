#!/usr/bin/env python3
"""
Test script with fixed prompt that fits within token limits
"""

import sys
import os
sys.path.append('src')

from src.core.image_generator import ImageGenerator

def test_fixed_prompt():
    """Test image generation with a fixed prompt that fits within token limits."""
    
    # Initialize image generator with 16:9 dimensions
    image_gen = ImageGenerator(
        width=1920,
        height=1080,
        enable_prompt_enhancement=False
    )
    
    # Test 1: Shorter prompt that fits within 77 tokens
    print("Test 1: Short prompt within token limits")
    result1 = image_gen.generate_cartoon_image(
        prompt="(wide shot:1.2) single Indian girl with black braids, pink dress, sitting by pond under moonlight, lotus flowers, storybook style:1.3",
        output_path="test_fixed_1.png",
        negative_prompt="multiple people, multiple characters, two people, three people, group of people, crowd, extra humans, additional characters, second person, third person, other people, background characters, side characters, multiple figures, duo, trio, ensemble, multiple subjects, people in background, bystanders, onlookers, spectators, multiple girls, multiple children, siblings, friends, family members, blurry, low resolution, text, watermark"
    )
    print(f"Result 1: {result1}")
    
    # Test 2: Even shorter prompt
    print("\nTest 2: Very short prompt")
    result2 = image_gen.generate_cartoon_image(
        prompt="(wide shot:1.2) Indian girl with braids, pink dress, pond, moonlight, lotus, storybook style:1.3",
        output_path="test_fixed_2.png",
        negative_prompt="multiple people, multiple characters, two people, three people, group of people, crowd, extra humans, additional characters, second person, third person, other people, background characters, side characters, multiple figures, duo, trio, ensemble, multiple subjects, people in background, bystanders, onlookers, spectators, multiple girls, multiple children, siblings, friends, family members, blurry, low resolution, text, watermark"
    )
    print(f"Result 2: {result2}")
    
    # Test 3: Simple prompt without weights
    print("\nTest 3: Simple prompt without weights")
    result3 = image_gen.generate_cartoon_image(
        prompt="Indian girl with black braids wearing pink dress sitting by pond under moonlight with lotus flowers, storybook style",
        output_path="test_fixed_3.png",
        negative_prompt="multiple people, multiple characters, two people, three people, group of people, crowd, extra humans, additional characters, second person, third person, other people, background characters, side characters, multiple figures, duo, trio, ensemble, multiple subjects, people in background, bystanders, onlookers, spectators, multiple girls, multiple children, siblings, friends, family members, blurry, low resolution, text, watermark"
    )
    print(f"Result 3: {result3}")

if __name__ == "__main__":
    test_fixed_prompt()
