#!/usr/bin/env python3
"""
Test script to find optimal prompt for single character generation
"""

import sys
import os
sys.path.append('src')

from src.core.image_generator import ImageGenerator

def test_optimal_prompt():
    """Test different prompt variations to find the optimal one."""
    
    # Initialize image generator with 16:9 dimensions
    image_gen = ImageGenerator(
        width=1920,
        height=1080,
        enable_prompt_enhancement=False
    )
    
    # Test 1: Very simple prompt with strong negative
    print("Test 1: Very simple prompt")
    result1 = image_gen.generate_cartoon_image(
        prompt="Indian girl with braids, pink dress, pond, moonlight, lotus, storybook style",
        output_path="test_optimal_1.png",
        negative_prompt="multiple people, multiple characters, two people, three people, group of people, crowd, extra humans, additional characters, second person, third person, other people, background characters, side characters, multiple figures, duo, trio, ensemble, multiple subjects, people in background, bystanders, onlookers, spectators, multiple girls, multiple children, siblings, friends, family members, blurry, low resolution, text, watermark"
    )
    print(f"Result 1: {result1}")
    
    # Test 2: Medium complexity prompt
    print("\nTest 2: Medium complexity prompt")
    result2 = image_gen.generate_cartoon_image(
        prompt="(wide shot:1.2) Indian girl with black braids, pink dress, sitting by pond under moonlight, lotus flowers, storybook style:1.3",
        output_path="test_optimal_2.png",
        negative_prompt="multiple people, multiple characters, two people, three people, group of people, crowd, extra humans, additional characters, second person, third person, other people, background characters, side characters, multiple figures, duo, trio, ensemble, multiple subjects, people in background, bystanders, onlookers, spectators, multiple girls, multiple children, siblings, friends, family members, blurry, low resolution, text, watermark"
    )
    print(f"Result 2: {result2}")
    
    # Test 3: Focus on single character with different approach
    print("\nTest 3: Focus on single character")
    result3 = image_gen.generate_cartoon_image(
        prompt="one Indian girl with braids, pink dress, pond, moonlight, lotus, storybook style",
        output_path="test_optimal_3.png",
        negative_prompt="multiple people, multiple characters, two people, three people, group of people, crowd, extra humans, additional characters, second person, third person, other people, background characters, side characters, multiple figures, duo, trio, ensemble, multiple subjects, people in background, bystanders, onlookers, spectators, multiple girls, multiple children, siblings, friends, family members, blurry, low resolution, text, watermark"
    )
    print(f"Result 3: {result3}")
    
    # Test 4: Most basic prompt possible
    print("\nTest 4: Most basic prompt")
    result4 = image_gen.generate_cartoon_image(
        prompt="Indian girl, pink dress, pond, moonlight, storybook style",
        output_path="test_optimal_4.png",
        negative_prompt="multiple people, multiple characters, two people, three people, group of people, crowd, extra humans, additional characters, second person, third person, other people, background characters, side characters, multiple figures, duo, trio, ensemble, multiple subjects, people in background, bystanders, onlookers, spectators, multiple girls, multiple children, siblings, friends, family members, blurry, low resolution, text, watermark"
    )
    print(f"Result 4: {result4}")

if __name__ == "__main__":
    test_optimal_prompt()
