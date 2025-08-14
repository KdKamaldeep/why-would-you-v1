#!/usr/bin/env python3
"""
Test script to verify the image generation fix for different video formats
"""

import os
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_fix():
    """Test the image generation fix for both formats."""
    
    print("🔧 Testing Image Generation Fix")
    print("=" * 50)
    
    # Test prompt from your storyboard
    test_prompt = "(wide shot:1.2) little Indian girl with long black hair tied in two braids, wearing pink frock, sitting near a pond under full moonlight, soft ripples in water, lotus flowers, storybook style:1.3"
    test_negative_prompt = "blurry, dark, low resolution, distorted, unnatural colors, scary, violent"
    
    try:
        from core.image_generator import ImageGenerator
        
        # Test 1: Shorts format (should work as before)
        print("\n📱 Test 1: YouTube Shorts Format (768x1024)")
        print("-" * 40)
        
        image_gen_shorts = ImageGenerator(
            width=768,
            height=1024,
            enable_prompt_enhancement=False
        )
        
        output_shorts = "test_fix_shorts.png"
        print(f"🎨 Generating shorts format image...")
        
        try:
            result_shorts = image_gen_shorts.generate_cartoon_image(
                test_prompt, 
                output_shorts, 
                negative_prompt=test_negative_prompt
            )
            print(f"✅ Shorts image generated: {result_shorts}")
            
            if Path(result_shorts).exists():
                size = Path(result_shorts).stat().st_size
                print(f"📊 File size: {size} bytes")
                print("✅ Shorts format working correctly")
            else:
                print("❌ Shorts image not created")
                
        except Exception as e:
            print(f"❌ Error with shorts format: {e}")
        
        # Test 2: Normal format (should now work with the fix)
        print("\n📺 Test 2: Normal Video Format (1920x1080)")
        print("-" * 40)
        
        image_gen_normal = ImageGenerator(
            width=1920,
            height=1080,
            enable_prompt_enhancement=False
        )
        
        output_normal = "test_fix_normal.png"
        print(f"🎨 Generating normal format image...")
        
        try:
            result_normal = image_gen_normal.generate_cartoon_image(
                test_prompt, 
                output_normal, 
                negative_prompt=test_negative_prompt
            )
            print(f"✅ Normal image generated: {result_normal}")
            
            if Path(result_normal).exists():
                size = Path(result_normal).stat().st_size
                print(f"📊 File size: {size} bytes")
                print("✅ Normal format now working with fix")
            else:
                print("❌ Normal image not created")
                
        except Exception as e:
            print(f"❌ Error with normal format: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 3: Even larger format to test scaling
        print("\n🖥️ Test 3: Large Format (2560x1440)")
        print("-" * 40)
        
        image_gen_large = ImageGenerator(
            width=2560,
            height=1440,
            enable_prompt_enhancement=False
        )
        
        output_large = "test_fix_large.png"
        print(f"🎨 Generating large format image...")
        
        try:
            result_large = image_gen_large.generate_cartoon_image(
                test_prompt, 
                output_large, 
                negative_prompt=test_negative_prompt
            )
            print(f"✅ Large image generated: {result_large}")
            
            if Path(result_large).exists():
                size = Path(result_large).stat().st_size
                print(f"📊 File size: {size} bytes")
                print("✅ Large format working with scaling fix")
            else:
                print("❌ Large image not created")
                
        except Exception as e:
            print(f"❌ Error with large format: {e}")
        
        # Summary
        print("\n📊 Fix Verification Summary")
        print("=" * 40)
        print("✅ The fix should now handle:")
        print("1. Shorts format (768x1024) - unchanged")
        print("2. Normal format (1920x1080) - now working")
        print("3. Large formats (2560x1440) - with scaling")
        print("4. Memory optimization for larger images")
        print("5. Automatic dimension validation")
        
        # Cleanup test files
        test_files = ["test_fix_shorts.png", "test_fix_normal.png", "test_fix_large.png"]
        for file in test_files:
            if Path(file).exists():
                Path(file).unlink()
                print(f"🧹 Cleaned up: {file}")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fix()
