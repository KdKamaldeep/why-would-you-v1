#!/usr/bin/env python3
"""
Test script to demonstrate scene pause functionality with the provided script.
"""

import sys
import os
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.generate_cartoon_short import VideoConfig, CartoonShortsGenerator

def test_scene_pauses():
    """Test scene pause functionality with the provided script."""
    
    # The script provided by the user
    script_data = {
        "title": "The Secret Treasure of Sundarvan",
        "description": "Aarav and Meera find an old map and embark on a thrilling quest to discover a hidden treasure buried deep in their village forest.",
        "total_duration": 50,
        "cast": [
            {"name": "Aarav", "role": "curious and brave boy"},
            {"name": "Meera", "role": "clever and resourceful girl"},
            {"name": "Nanu", "role": "wise village elder and storyteller"}
        ],
        "scenes": [
            {
                "duration": 5,
                "description": "warm village evening, children gathered around Nanu listening to stories",
                "visual_prompt": "(wide shot:1.3) Indian village evening, children gathered around elderly man, warm lantern light, (storybook cozy:1.4)",
                "narration": "नानू ने सुनदरवन के जंगल में छुपे खजाने की कहानी सुनाई।",
                "subtitle": "The Secret",
                "characters": ["Nanu", "Children"]
            },
            {
                "duration": 5,
                "description": "close-up of old weathered map in Aarav's hands",
                "visual_prompt": "close-up old map with faded markings, hands holding it carefully",
                "narration": "आरव को एक पुराना नक्शा मिला, जो खजाने तक ले जाता था।",
                "subtitle": "The Map",
                "characters": ["Aarav"]
            },
            {
                "duration": 5,
                "description": "Meera and Aarav studying map excitedly under lantern light",
                "visual_prompt": "medium shot of two children looking at an old map by warm lantern light in a cozy Indian village, storybook illustration style, soft warm glow",
                "narration": "मीरा ने उत्साह से कहा, 'चलो, हम इसे खोजते हैं!'",
                "subtitle": "The Plan",
                "characters": ["Aarav", "Meera"]
            },
            {
                "duration": 5,
                "description": "children running through village towards forest at sunset",
                "visual_prompt": "(wide shot:1.3) children running through village streets, sunset sky, (storybook bright:1.4)",
                "narration": "सूरज ढल रहा था, और वे जंगल की ओर दौड़े।",
                "subtitle": "Setting Out",
                "characters": ["Aarav", "Meera"]
            },
            {
                "duration": 5,
                "description": "entrance to dense forest with soft glowing lights and shadows",
                "visual_prompt": "(wide shot:1.3) forest entrance, soft glowing lights and shadows, gentle evening, (storybook magical:1.4)",
                "narration": "जंगल के रास्ते में हर कदम उनके लिए नया रहस्य लेकर आया।",
                "subtitle": "The Forest Path",
                "characters": ["Aarav", "Meera"]
            },
            {
                "duration": 5,
                "description": "children walking carefully on narrow forest trail holding lanterns",
                "visual_prompt": "(medium shot) children with lanterns walking on narrow forest trail, soft ambient light, (storybook magical:1.3)",
                "narration": "लालटेन की रौशनी में वे आगे बढ़ते रहे।",
                "subtitle": "Guided by Light",
                "characters": ["Aarav", "Meera"]
            },
            {
                "duration": 5,
                "description": "hidden clearing with ancient stone chest covered in moss",
                "visual_prompt": "(wide shot:1.3) hidden forest clearing, moss-covered stone chest, dappled moonlight, (storybook mysterious:1.4)",
                "narration": "अंत में, उन्होंने छुपा हुआ खजाना देखा।",
                "subtitle": "Treasure Found",
                "characters": ["Aarav", "Meera"]
            },
            {
                "duration": 5,
                "description": "close-up of children opening chest, glowing light spilling out",
                "visual_prompt": "(close-up) children opening glowing treasure chest, golden light spilling out, (storybook magical:1.4)",
                "narration": "खजाने की चमक ने उनके दिलों को रोशन कर दिया।",
                "subtitle": "The Glow",
                "characters": ["Aarav", "Meera"]
            },
            {
                "duration": 5,
                "description": "children carrying treasure chest back towards village with joyful faces",
                "visual_prompt": "(medium shot) children carrying treasure chest through forest, smiling faces, soft dawn light, (storybook bright:1.3)",
                "narration": "खजाना लेकर वे खुशी-खुशी गाँव लौटे।",
                "subtitle": "Return Home",
                "characters": ["Aarav", "Meera"]
            },
            {
                "duration": 5,
                "description": "village morning celebration with villagers smiling, treasure chest in center",
                "visual_prompt": "(wide shot:1.3) bright village morning, villagers celebrating, treasure chest displayed, (storybook joyful:1.4)",
                "narration": "पूरा गाँव उनकी सफलता पर खुश था।",
                "subtitle": "Celebration",
                "characters": ["Aarav", "Meera", "Nanu"]
            }
        ],
        "tags": [
            "kids", "adventure", "treasure", "friendship", "storybook", "Indian village", "magic"
        ]
    }
    
    print("🎬 Testing Scene Pause Functionality")
    print("=" * 50)
    
    # Test different pause durations
    pause_durations = [0.0, 0.5, 1.0, 1.5]
    
    for pause_duration in pause_durations:
        print(f"\n⏸️ Testing with {pause_duration:.1f}s pause between scenes")
        print("-" * 40)
        
        # Create configuration with custom scenes and pause duration
        config = VideoConfig(
            prompt="The Secret Treasure of Sundarvan",
            duration=50,
            output_path=f"output_pause_{pause_duration}",
            custom_scenes=script_data["scenes"],
            scene_duration=5,  # Default duration for scenes that don't specify it
            scene_pause_duration=pause_duration,
            enable_prompt_enhancement=True,
            reuse_existing=False  # Force regeneration for testing
        )
        
        # Calculate expected timing
        total_scenes = len(script_data["scenes"])
        total_scene_duration = sum(scene.get("duration", 5) for scene in script_data["scenes"])
        total_pause_duration = pause_duration * (total_scenes - 1)  # Pauses between scenes
        total_video_duration = total_scene_duration + total_pause_duration
        
        print(f"📊 Scene count: {total_scenes}")
        print(f"📊 Total scene duration: {total_scene_duration}s")
        print(f"📊 Total pause duration: {total_pause_duration:.1f}s")
        print(f"📊 Expected video duration: {total_video_duration:.1f}s")
        
        if pause_duration > 0:
            print(f"⏸️ Pauses will be added between each scene")
        else:
            print(f"⏸️ No pauses - scenes will transition directly")
        
        # Note: We're not actually running the generation here to avoid API costs
        # In a real test, you would uncomment the following lines:
        
        # try:
        #     generator = CartoonShortsGenerator(config)
        #     output_path = generator.generate()
        #     print(f"✅ Generated video: {output_path}")
        # except Exception as e:
        #     print(f"❌ Error: {e}")
    
    print(f"\n🎉 Scene pause functionality test completed!")
    print(f"📝 To run actual generation, uncomment the generation code in this script")

if __name__ == "__main__":
    test_scene_pauses()
