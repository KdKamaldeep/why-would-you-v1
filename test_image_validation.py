#!/usr/bin/env python3
"""
Test script to demonstrate image validation and automatic prompt adjustment functionality.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.image_generator import ImageGenerator

def test_image_validation():
    """Test the image validation and prompt adjustment functionality."""
    
    print("🎨 Testing Image Validation and Prompt Adjustment")
    print("=" * 60)
    
    # Test prompts that might produce blank or poor quality images
    test_prompts = [
        "empty room",  # Likely to be blank
        "white background",  # Very minimal
        "dark scene",  # Might be too dark
        "simple shape",  # Too vague
        "Indian village evening, children gathered around elderly man, warm lantern light, storybook cozy",  # Good prompt from user's script
        "close-up old map with faded markings, hands holding it carefully",  # Another good prompt
    ]
    
    # Initialize image generator with validation enabled
    print("🚀 Initializing Image Generator with validation...")
    image_generator = ImageGenerator(
        enable_prompt_enhancement=True
    )
    
    if not image_generator.is_sd_available():
        print("⚠️ Stable Diffusion not available. This test requires SD to be properly configured.")
        print("💡 The validation logic will still be demonstrated with placeholder images.")
    
    # Test each prompt
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📝 Test {i}: '{prompt}'")
        print("-" * 40)
        
        # Test with validation enabled
        output_path = f"test_validation_scene_{i}.png"
        
        try:
            print(f"🎨 Generating image with validation...")
            result_path = image_generator.generate_cartoon_image_with_validation(
                prompt, 
                output_path, 
                max_attempts=3
            )
            
            print(f"✅ Final result: {result_path}")
            
            # Check if the final image passes validation
            if image_generator._is_image_blank_or_poor_quality(result_path):
                print(f"⚠️ Final image still has quality issues")
            else:
                print(f"✅ Final image passes quality validation")
                
        except Exception as e:
            print(f"❌ Error during generation: {e}")
    
    print(f"\n🎉 Image validation test completed!")
    print(f"📁 Check the generated images to see the results")

def test_blank_image_detection():
    """Test the blank image detection logic specifically."""
    
    print("\n🔍 Testing Blank Image Detection Logic")
    print("=" * 50)
    
    image_generator = ImageGenerator()
    
    # Create some test images to validate the detection logic
    test_cases = [
        ("blank_white.png", "Create a completely white image"),
        ("blank_black.png", "Create a completely black image"),
        ("simple_gradient.png", "Create a simple gradient"),
        ("complex_scene.png", "Create a complex cartoon scene with multiple elements")
    ]
    
    for filename, description in test_cases:
        print(f"\n📊 Testing: {description}")
        
        # This would normally generate an actual image
        # For this test, we'll simulate the validation logic
        print(f"   Would analyze: {filename}")
        print(f"   Detection logic: Variance, unique colors, brightness analysis")
        print(f"   Expected result: {'Blank detected' if 'blank' in filename else 'Quality OK'}")

if __name__ == "__main__":
    test_image_validation()
    test_blank_image_detection()
