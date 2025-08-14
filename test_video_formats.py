#!/usr/bin/env python3
"""
Test script to demonstrate video format functionality.
This script shows how to create both YouTube Shorts (9:16) and normal videos (16:9).
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.generate_cartoon_short import CartoonShortsGenerator, VideoConfig

def test_youtube_shorts():
    """Test creating a YouTube Shorts video (9:16 aspect ratio)."""
    print("🎬 Testing YouTube Shorts format (9:16 aspect ratio)...")
    
    config = VideoConfig(
        prompt="A baby lion opens a smoothie shop in the jungle",
        duration=20,
        video_format="shorts",
        output_path="output/shorts_test",
        style="cartoon",
        language="en",
        enable_prompt_enhancement=True,
        scene_pause_duration=0.0,
        enable_image_validation=True
    )
    
    print(f"📐 Video dimensions: {config.width}x{config.height}")
    print(f"📐 Aspect ratio: {config.width/config.height:.2f}:1")
    
    try:
        generator = CartoonShortsGenerator(config)
        output_path = generator.generate()
        print(f"✅ YouTube Shorts created: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Error creating YouTube Shorts: {e}")
        return False

def test_normal_video():
    """Test creating a normal video (16:9 aspect ratio)."""
    print("\n🎬 Testing Normal Video format (16:9 aspect ratio)...")
    
    config = VideoConfig(
        prompt="A baby lion opens a smoothie shop in the jungle",
        duration=20,
        video_format="normal",
        output_path="output/normal_test",
        style="cartoon",
        language="en",
        enable_prompt_enhancement=True,
        scene_pause_duration=0.0,
        enable_image_validation=True
    )
    
    print(f"📐 Video dimensions: {config.width}x{config.height}")
    print(f"📐 Aspect ratio: {config.width/config.height:.2f}:1")
    
    try:
        generator = CartoonShortsGenerator(config)
        output_path = generator.generate()
        print(f"✅ Normal video created: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Error creating normal video: {e}")
        return False

def main():
    """Run both video format tests."""
    print("🚀 Testing Video Format Functionality")
    print("=" * 50)
    
    # Check if required environment variables are set
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key before running this test.")
        return
    
    # Test YouTube Shorts
    shorts_success = test_youtube_shorts()
    
    # Test Normal Video
    normal_success = test_normal_video()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"YouTube Shorts (9:16): {'✅ PASS' if shorts_success else '❌ FAIL'}")
    print(f"Normal Video (16:9): {'✅ PASS' if normal_success else '❌ FAIL'}")
    
    if shorts_success and normal_success:
        print("\n🎉 All tests passed! Video format functionality is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()
