#!/usr/bin/env python3
"""
Test script to check memory-related issues with different image sizes
"""

import os
import sys
import gc
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_memory_with_different_sizes():
    """Test image generation with different sizes to identify memory issues."""
    
    print("🧠 Testing Memory Usage with Different Image Sizes")
    print("=" * 60)
    
    test_prompt = "(wide shot:1.2) little Indian girl with long black hair tied in two braids, wearing pink frock, sitting near a pond under full moonlight, soft ripples in water, lotus flowers, storybook style:1.3"
    test_negative_prompt = "blurry, dark, low resolution, distorted, unnatural colors, scary, violent"
    
    try:
        import torch
        from core.image_generator import ImageGenerator
        
        # Check initial memory state
        print("\n💻 Initial System State")
        print("-" * 30)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"🎮 CUDA available: Yes")
            print(f"🎮 Initial CUDA memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        else:
            print(f"🎮 CUDA available: No")
        
        # Test different image sizes
        test_sizes = [
            (512, 512, "Small square"),
            (768, 1024, "Shorts format"),
            (1024, 768, "Landscape shorts"),
            (1280, 720, "HD landscape"),
            (1920, 1080, "Full HD"),
            (2560, 1440, "2K"),
        ]
        
        for width, height, description in test_sizes:
            print(f"\n📐 Testing {description}: {width}x{height}")
            print("-" * 40)
            
            # Clear memory before each test
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                print(f"🧹 Memory cleared. CUDA memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
            try:
                # Initialize image generator with specific size
                image_gen = ImageGenerator(
                    width=width,
                    height=height,
                    enable_prompt_enhancement=False
                )
                
                print(f"✅ ImageGenerator initialized for {width}x{height}")
                print(f"🎮 Device: {image_gen.device}")
                print(f"🎨 SD available: {image_gen.is_sd_available()}")
                
                if torch.cuda.is_available():
                    print(f"🎮 Memory after init: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                
                # Try to generate image
                output_file = f"test_{width}x{height}.png"
                print(f"🎨 Attempting to generate image...")
                
                result = image_gen.generate_cartoon_image(
                    test_prompt,
                    output_file,
                    negative_prompt=test_negative_prompt
                )
                
                if Path(result).exists():
                    size = Path(result).stat().st_size
                    print(f"✅ Image generated successfully: {size} bytes")
                    
                    if torch.cuda.is_available():
                        print(f"🎮 Memory after generation: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                else:
                    print(f"❌ Image generation failed - file not created")
                
                # Clean up
                del image_gen
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                
            except Exception as e:
                print(f"❌ Error with {width}x{height}: {e}")
                if torch.cuda.is_available():
                    print(f"🎮 Memory at error: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # Final memory check
        print(f"\n💻 Final System State")
        print("-" * 30)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            print(f"🎮 Final CUDA memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        print(f"\n📊 Test Summary")
        print("=" * 30)
        print("This test helps identify:")
        print("1. Memory usage patterns with different image sizes")
        print("2. Whether larger images cause memory issues")
        print("3. If the model has size limitations")
        print("4. Memory leaks during generation")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_memory_with_different_sizes()
