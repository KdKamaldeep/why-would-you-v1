#!/usr/bin/env python3
"""
Test script with optimized prompts that fit within CLIP token limits
"""

import sys
import os
sys.path.append('src')

from src.core.image_generator import ImageGenerator

def test_optimized_prompts():
    """Test image generation with optimized prompts that fit within token limits."""
    
    # Initialize image generator with 16:9 dimensions
    image_gen = ImageGenerator(
        width=1920,
        height=1080,
        enable_prompt_enhancement=False
    )
    
    # Test 1: Optimized prompt that fits within 77 tokens
    print("Test 1: Optimized prompt within token limits")
    result1 = image_gen.generate_cartoon_image(
        prompt="(Lavanya:1.2), (11-year-old girl:1.1), (yellow dress:1.1), (flying:1.2), (golden glowbirds:1.1), (magical sky:0.9), (storybook style:0.8)",
        output_path="test_optimized_1.png",
        negative_prompt="blurry, low resolution, text, watermark"
    )
    print(f"Result 1: {result1}")
    
    # Test 2: Even more concise prompt
    print("\nTest 2: Concise prompt")
    result2 = image_gen.generate_cartoon_image(
        prompt="(Lavanya:1.2), (girl in yellow dress:1.1), (flying with golden birds:1.2), (magical sky:0.9), (cartoon style:0.8)",
        output_path="test_optimized_2.png",
        negative_prompt="blurry, low resolution, text, watermark"
    )
    print(f"Result 2: {result2}")
    
    # Test 3: Simple prompt without weights
    print("\nTest 3: Simple prompt without weights")
    result3 = image_gen.generate_cartoon_image(
        prompt="Lavanya, 11-year-old girl in yellow dress, flying with golden glowbirds, magical sky, cartoon style",
        output_path="test_optimized_3.png",
        negative_prompt="blurry, low resolution, text, watermark"
    )
    print(f"Result 3: {result3}")

if __name__ == "__main__":
    test_optimized_prompts()
