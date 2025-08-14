#!/usr/bin/env python3
"""
Test script to verify video format handling with storyboard JSON
"""

import json
import os
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_video_formats():
    """Test both shorts and normal video formats with the same storyboard."""
    
    # The storyboard JSON you provided
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
            },
            {
                "duration": 8,
                "description": "medium shot of Meera touching the boat and it growing bigger",
                "visual_prompt": "(medium shot:1.2) little Indian girl with long black hair in braids, wearing pink frock, touching glowing magical paper boat which is expanding in size, moonlight, storybook style:1.3",
                "negative_prompt": "blurry, dark, low resolution, distorted, unnatural colors, scary, violent",
                "narration": "मीरा ने नाव को छुआ और वह अचानक बड़ी होकर बैठने लायक हो गई।",
                "subtitle": "Growing Magic",
                "characters": ["Meera", "Moonlight Boat"],
                "motion_prompt": "girl touching boat, boat glowing and growing bigger"
            },
            {
                "duration": 8,
                "description": "medium shot of the Firefly Queen flying near Meera",
                "visual_prompt": "(medium shot:1.2) beautiful golden firefly queen with delicate wings, glowing softly, hovering in front of girl in moonlit night, magical forest backdrop, storybook style:1.3",
                "negative_prompt": "blurry, dark, low resolution, distorted, unnatural colors, scary, violent",
                "narration": "तभी एक सुनहरी जुगनू रानी आई और बोली, 'क्या तुम हमारे खोए हुए दोस्तों को घर पहुँचाने में मदद करोगी?'",
                "subtitle": "The Firefly Queen",
                "characters": ["Firefly Queen", "Meera"],
                "motion_prompt": "firefly queen fluttering in front of girl"
            },
            {
                "duration": 9,
                "description": "wide shot of Meera sailing the glowing boat with fireflies around her",
                "visual_prompt": "(wide shot:1.2) little Indian girl with long black hair in braids, wearing pink frock, sailing glowing magical paper boat in moonlit pond, dozens of golden fireflies flying around, storybook style:1.3",
                "negative_prompt": "blurry, dark, low resolution, distorted, unnatural colors, scary, violent",
                "narration": "मीरा नाव पर बैठकर जुगनुओं के साथ उनकी रोशनी का पीछा करती हुई आगे बढ़ी।",
                "subtitle": "The Moonlight Journey",
                "characters": ["Meera", "Firefly Queen"],
                "motion_prompt": "girl sailing boat with fireflies around"
            },
            {
                "duration": 9,
                "description": "medium shot of Meera releasing the fireflies into the starry sky",
                "visual_prompt": "(medium shot:1.2) little Indian girl with long black hair in braids, wearing pink frock, standing on glowing paper boat, releasing golden fireflies into star-filled sky, magical atmosphere, storybook style:1.3",
                "negative_prompt": "blurry, dark, low resolution, distorted, unnatural colors, scary, violent",
                "narration": "अंत में, मीरा ने जुगनुओं को तारों भरे आसमान में छोड़ दिया। 'अगर आपको यह कहानी पसंद आई हो, तो कृपया सब्सक्राइब करें और बेल आइकन दबाएं।'",
                "subtitle": "Home at Last",
                "characters": ["Meera", "Firefly Queen"],
                "motion_prompt": "girl releasing fireflies flying into sky"
            }
        ],
        "tags": [
            "kids",
            "magical",
            "boat",
            "friendship",
            "storybook",
            "adventure",
            "night",
            "cute"
        ]
    }
    
    # Save the storyboard to a temporary file
    storyboard_path = "test_storyboard.json"
    with open(storyboard_path, 'w', encoding='utf-8') as f:
        json.dump(storyboard_data, f, indent=2)
    
    print("🎬 Testing Video Format Handling")
    print("=" * 50)
    
    try:
        from core.generate_cartoon_short import CartoonShortsGenerator, VideoConfig
        
        # Test 1: Shorts format (9:16)
        print("\n📱 Test 1: YouTube Shorts Format (9:16)")
        print("-" * 30)
        
        config_shorts = VideoConfig(
            prompt="Meera and the Moonlight Boat",
            duration=50,
            video_format="shorts",
            output_path="test_output_shorts",
            title=storyboard_data["title"],
            description=storyboard_data["description"],
            custom_scenes=storyboard_data["scenes"],
            scene_duration=8,
            reuse_existing=False,  # Force regeneration
            add_subtitles=True,
            language="hi",
            enable_prompt_enhancement=False  # Use original prompts
        )
        
        print(f"📐 Dimensions: {config_shorts.width}x{config_shorts.height}")
        print(f"🎯 Aspect ratio: {config_shorts.width/config_shorts.height:.2f}")
        
        generator_shorts = CartoonShortsGenerator(config_shorts)
        output_shorts = generator_shorts.generate()
        
        print(f"✅ Shorts video generated: {output_shorts}")
        
        # Test 2: Normal format (16:9)
        print("\n📺 Test 2: Normal Video Format (16:9)")
        print("-" * 30)
        
        config_normal = VideoConfig(
            prompt="Meera and the Moonlight Boat",
            duration=50,
            video_format="normal",
            output_path="test_output_normal",
            title=storyboard_data["title"],
            description=storyboard_data["description"],
            custom_scenes=storyboard_data["scenes"],
            scene_duration=8,
            reuse_existing=False,  # Force regeneration
            add_subtitles=True,
            language="hi",
            enable_prompt_enhancement=False  # Use original prompts
        )
        
        print(f"📐 Dimensions: {config_normal.width}x{config_normal.height}")
        print(f"🎯 Aspect ratio: {config_normal.width/config_normal.height:.2f}")
        
        generator_normal = CartoonShortsGenerator(config_normal)
        output_normal = generator_normal.generate()
        
        print(f"✅ Normal video generated: {output_normal}")
        
        # Summary
        print("\n🎉 Test Results Summary")
        print("=" * 30)
        print(f"📱 Shorts format: {output_shorts}")
        print(f"📺 Normal format: {output_normal}")
        print(f"📊 Both formats should work with the same storyboard JSON")
        
        # Cleanup
        if os.path.exists(storyboard_path):
            os.remove(storyboard_path)
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_video_formats()
