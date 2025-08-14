#!/usr/bin/env python3
"""
Test script to identify image generation issues between shorts and normal video formats
"""

import os
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_image_generation_formats():
    """Test image generation specifically for both video formats."""
    
    print("🎨 Testing Image Generation for Different Video Formats")
    print("=" * 60)
    
    # Test prompt from your storyboard
    test_prompt = "(wide shot:1.2) little Indian girl with long black hair tied in two braids, wearing pink frock, sitting near a pond under full moonlight, soft ripples in water, lotus flowers, storybook style:1.3"
    test_negative_prompt = "blurry, dark, low resolution, distorted, unnatural colors, scary, violent"
    
    try:
        from core.image_generator import ImageGenerator
        
        # Test 1: Shorts format (9:16) - 768x1024
        print("\n📱 Test 1: YouTube Shorts Format (9:16)")
        print("-" * 40)
        print(f"📐 Dimensions: 768x1024")
        print(f"🎯 Aspect ratio: 0.75 (9:16)")
        
        image_gen_shorts = ImageGenerator(
            width=768,
            height=1024,
            enable_prompt_enhancement=False
        )
        
        output_shorts = "test_image_shorts.png"
        print(f"🎨 Generating image for shorts format...")
        
        try:
            result_shorts = image_gen_shorts.generate_cartoon_image(
                test_prompt, 
                output_shorts, 
                negative_prompt=test_negative_prompt
            )
            print(f"✅ Shorts image generated: {result_shorts}")
            
            # Check if file exists and has content
            if Path(result_shorts).exists():
                size = Path(result_shorts).stat().st_size
                print(f"📊 File size: {size} bytes")
                if size > 0:
                    print("✅ Image file is valid")
                else:
                    print("❌ Image file is empty")
            else:
                print("❌ Image file not created")
                
        except Exception as e:
            print(f"❌ Error generating shorts image: {e}")
        
        # Test 2: Normal format (16:9) - 1920x1080
        print("\n📺 Test 2: Normal Video Format (16:9)")
        print("-" * 40)
        print(f"📐 Dimensions: 1920x1080")
        print(f"🎯 Aspect ratio: 1.78 (16:9)")
        
        image_gen_normal = ImageGenerator(
            width=1920,
            height=1080,
            enable_prompt_enhancement=False
        )
        
        output_normal = "test_image_normal.png"
        print(f"🎨 Generating image for normal format...")
        
        try:
            result_normal = image_gen_normal.generate_cartoon_image(
                test_prompt, 
                output_normal, 
                negative_prompt=test_negative_prompt
            )
            print(f"✅ Normal image generated: {result_normal}")
            
            # Check if file exists and has content
            if Path(result_normal).exists():
                size = Path(result_normal).stat().st_size
                print(f"📊 File size: {size} bytes")
                if size > 0:
                    print("✅ Image file is valid")
                else:
                    print("❌ Image file is empty")
            else:
                print("❌ Image file not created")
                
        except Exception as e:
            print(f"❌ Error generating normal image: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 3: Check if SD is available for both formats
        print("\n🔍 Test 3: Stable Diffusion Availability Check")
        print("-" * 40)
        
        print(f"📱 Shorts format SD available: {image_gen_shorts.is_sd_available()}")
        print(f"📺 Normal format SD available: {image_gen_normal.is_sd_available()}")
        
        # Test 4: Check memory usage and device
        print("\n💻 Test 4: System Information")
        print("-" * 40)
        
        print(f"📱 Shorts device: {image_gen_shorts.device}")
        print(f"📺 Normal device: {image_gen_normal.device}")
        
        try:
            import torch
            if torch.cuda.is_available():
                print(f"🎮 CUDA available: Yes")
                print(f"🎮 CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                print(f"🎮 CUDA memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
            else:
                print(f"🎮 CUDA available: No")
        except Exception as e:
            print(f"🎮 CUDA check failed: {e}")
        
        # Summary
        print("\n📊 Test Summary")
        print("=" * 40)
        print("This test helps identify if the issue is:")
        print("1. Memory-related (larger 16:9 images)")
        print("2. Model compatibility (different aspect ratios)")
        print("3. Pipeline initialization (format-specific)")
        print("4. File system or permissions")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_image_generation_formats()
