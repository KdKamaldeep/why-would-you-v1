#!/usr/bin/env python3
"""
Test script to ensure single character generation
"""

import sys
import os
sys.path.append('src')

from src.core.image_generator import ImageGenerator

def test_single_character():
    """Test image generation to ensure only one character is generated."""
    
    # Initialize image generator with 16:9 dimensions
    image_gen = ImageGenerator(
        width=1920,
        height=1080,
        enable_prompt_enhancement=False
    )
    
    base_prompt = "(wide shot:1.2) little Indian girl with long black hair tied in two braids, wearing pink frock, sitting near a pond under full moonlight, soft ripples in water, lotus flowers, storybook style:1.3"
    
    # Test 1: Very strong single character negative prompt
    print("Test 1: Very strong single character negative prompt")
    result1 = image_gen.generate_cartoon_image(
        prompt=base_prompt,
        output_path="test_single_character_1.png",
        negative_prompt="multiple people, multiple characters, two people, three people, group of people, crowd, extra humans, additional characters, second person, third person, other people, background characters, side characters, multiple figures, duo, trio, ensemble, multiple subjects, people in background, bystanders, onlookers, spectators, multiple girls, multiple children, siblings, friends, family members, blurry, low resolution, text, watermark"
    )
    print(f"Result 1: {result1}")
    
    # Test 2: Even more specific negative prompt
    print("\nTest 2: Ultra-specific single character negative prompt")
    result2 = image_gen.generate_cartoon_image(
        prompt=base_prompt,
        output_path="test_single_character_2.png",
        negative_prompt="2 people, 3 people, 4 people, multiple girls, multiple children, two girls, three girls, group of girls, crowd of children, multiple figures, duo, trio, ensemble, multiple subjects, people in background, bystanders, onlookers, spectators, multiple girls, multiple children, siblings, friends, family members, extra humans, additional characters, second person, third person, other people, background characters, side characters, multiple figures, blurry, low resolution, text, watermark"
    )
    print(f"Result 2: {result2}")
    
    # Test 3: Modified prompt to emphasize single character
    print("\nTest 3: Modified prompt emphasizing single character")
    result3 = image_gen.generate_cartoon_image(
        prompt="(wide shot:1.2) single little Indian girl with long black hair tied in two braids, wearing pink frock, sitting alone near a pond under full moonlight, soft ripples in water, lotus flowers, storybook style:1.3",
        output_path="test_single_character_3.png",
        negative_prompt="multiple people, multiple characters, two people, three people, group of people, crowd, extra humans, additional characters, second person, third person, other people, background characters, side characters, multiple figures, duo, trio, ensemble, multiple subjects, people in background, bystanders, onlookers, spectators, multiple girls, multiple children, siblings, friends, family members, blurry, low resolution, text, watermark"
    )
    print(f"Result 3: {result3}")

if __name__ == "__main__":
    test_single_character()
