#!/usr/bin/env python3
"""
Test script to verify storyboard works with both video formats
"""

import json
import os
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_storyboard_formats():
    """Test the same storyboard with both shorts and normal formats."""
    
    # Your storyboard data
    storyboard_data = {
        "title": "Meera and the Moonlight Boat",
        "description": "Meera, a kind-hearted little girl, discovers a magical paper boat glowing under the moonlight and sails on it to help guide lost fireflies home.",
        "total_duration": 50,
        "cast": [
            {"name": "Meera", "role": "kind-hearted little girl"},
            {"name": "Moonlight Boat", "role": "glowing magical paper boat"},
            {"name": "Firefly Queen", "role": "gentle leader of glowing fireflies"}
        ],
        "scenes": [
            {
                "duration": 8,
                "description": "wide shot of Meera sitting near a small pond under a bright full moon",
                "visual_prompt": "(wide shot:1.2) little Indian girl with long black hair tied in two braids, wearing pink frock, sitting near a pond under full moonlight, soft ripples in water, lotus flowers, storybook style:1.3",
                "negative_prompt": "blurry, dark, low resolution, distorted, unnatural colors, scary, violent",
                "narration": "मीरा चाँदनी रात में अपने घर के पास तालाब के किनारे बैठी थी।",
                "subtitle": "The Moonlit Night",
                "characters": ["Meera"],
                "motion_prompt": "girl sitting by pond, moonlight reflecting on water"
            },
            {
                "duration": 8,
                "description": "close-up of a glowing paper boat floating on the pond",
                "visual_prompt": "(close-up:1.2) glowing magical paper boat with soft golden light floating gently on pond, surrounded by moonlit ripples, storybook style:1.3",
                "negative_prompt": "blurry, dark, low resolution, distorted, unnatural colors, scary, violent",
                "narration": "अचानक उसने तालाब में एक चमकती कागज़ की नाव तैरती देखी।",
                "subtitle": "The Magical Boat",
                "characters": ["Moonlight Boat"],
                "motion_prompt": "glowing paper boat gently floating on pond"
            }
        ],
        "tags": ["kids", "magical", "boat", "friendship", "storybook", "adventure", "night", "cute"]
    }
    
    print("🎬 Testing Storyboard with Both Video Formats")
    print("=" * 60)
    
    try:
        from core.generate_cartoon_short import CartoonShortsGenerator, VideoConfig
        
        # Test 1: Shorts format
        print("\n📱 Test 1: YouTube Shorts Format")
        print("-" * 40)
        
        config_shorts = VideoConfig(
            prompt="Meera and the Moonlight Boat",
            duration=16,  # Reduced for testing
            video_format="shorts",
            output_path="test_storyboard_shorts",
            title=storyboard_data["title"],
            description=storyboard_data["description"],
            custom_scenes=storyboard_data["scenes"],
            scene_duration=8,
            reuse_existing=False,
            add_subtitles=True,
            language="hi",
            enable_prompt_enhancement=False
        )
        
        print(f"📐 Dimensions: {config_shorts.width}x{config_shorts.height}")
        print(f"🎯 Aspect ratio: {config_shorts.width/config_shorts.height:.2f}")
        
        generator_shorts = CartoonShortsGenerator(config_shorts)
        output_shorts = generator_shorts.generate()
        
        print(f"✅ Shorts video generated: {output_shorts}")
        
        # Test 2: Normal format
        print("\n📺 Test 2: Normal Video Format")
        print("-" * 40)
        
        config_normal = VideoConfig(
            prompt="Meera and the Moonlight Boat",
            duration=16,  # Reduced for testing
            video_format="normal",
            output_path="test_storyboard_normal",
            title=storyboard_data["title"],
            description=storyboard_data["description"],
            custom_scenes=storyboard_data["scenes"],
            scene_duration=8,
            reuse_existing=False,
            add_subtitles=True,
            language="hi",
            enable_prompt_enhancement=False
        )
        
        print(f"📐 Dimensions: {config_normal.width}x{config_normal.height}")
        print(f"🎯 Aspect ratio: {config_normal.width/config_normal.height:.2f}")
        
        generator_normal = CartoonShortsGenerator(config_normal)
        output_normal = generator_normal.generate()
        
        print(f"✅ Normal video generated: {output_normal}")
        
        # Summary
        print("\n🎉 Test Results Summary")
        print("=" * 40)
        print(f"📱 Shorts format: {output_shorts}")
        print(f"📺 Normal format: {output_normal}")
        print(f"📊 Both formats should now work with the same storyboard JSON")
        print(f"🔧 The image generation fix should resolve the normal format issue")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_storyboard_formats()
