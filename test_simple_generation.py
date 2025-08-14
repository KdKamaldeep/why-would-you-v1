#!/usr/bin/env python3
"""
Simple test script to debug blank image generation
"""

import sys
import os
sys.path.append('src')

from src.core.image_generator import ImageGenerator

def test_simple_generation():
    """Test basic image generation without prompt enhancement."""
    
    # Initialize image generator with 16:9 dimensions
    image_gen = ImageGenerator(
        width=1920,
        height=1080,
        enable_prompt_enhancement=False  # Disable prompt enhancement
    )
    
    # Test 1: Simple prompt without weights
    print("Test 1: Simple prompt without weights")
    result1 = image_gen.generate_cartoon_image(
        prompt="11-year-old girl in yellow dress flying with golden birds",
        output_path="test_simple_1.png",
        negative_prompt="blurry, low resolution, text, watermark"
    )
    print(f"Result 1: {result1}")
    
    # Test 2: Your original prompt with weights
    print("\nTest 2: Original prompt with weights")
    result2 = image_gen.generate_cartoon_image(
        prompt="(11-year-old girl, Lavanya, yellow dress, brown braid:1.2), (flying joyfully:1.1), (golden glowbird leading:1.1), (soft glowing glowbirds:1.0), (magical sky:0.8), (storybook style:0.7), (cinematic wide shot:0.9)",
        output_path="test_simple_2.png",
        negative_prompt="blurry, low resolution, text, watermark"
    )
    print(f"Result 2: {result2}")
    
    # Test 3: Even simpler negative prompt
    print("\nTest 3: Minimal negative prompt")
    result3 = image_gen.generate_cartoon_image(
        prompt="(11-year-old girl, Lavanya, yellow dress, brown braid:1.2), (flying joyfully:1.1), (golden glowbird leading:1.1), (soft glowing glowbirds:1.0), (magical sky:0.8), (storybook style:0.7), (cinematic wide shot:0.9)",
        output_path="test_simple_3.png",
        negative_prompt="blurry"
    )
    print(f"Result 3: {result3}")

if __name__ == "__main__":
    test_simple_generation()
