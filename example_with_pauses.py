#!/usr/bin/env python3
"""
Example script showing how to use scene pauses with the provided story.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.generate_cartoon_short import VideoConfig, CartoonShortsGenerator

def main():
    """Generate the treasure hunt story with configurable pauses."""
    
    # The story scenes from the user's script
    custom_scenes = [
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
    ]
    
    print("🎬 The Secret Treasure of Sundarvan")
    print("=" * 50)
    print("📖 A story about Aarav and Meera's treasure hunt adventure")
    print(f"📊 {len(custom_scenes)} scenes, each {custom_scenes[0]['duration']} seconds")
    
    # Configuration with pauses and image validation
    config = VideoConfig(
        prompt="The Secret Treasure of Sundarvan",
        duration=50,
        output_path="treasure_hunt_with_pauses",
        custom_scenes=custom_scenes,
        scene_duration=5,
        scene_pause_duration=1.0,  # 1 second pause between scenes
        enable_prompt_enhancement=True,
        enable_image_validation=True,  # Enable automatic image validation and prompt adjustment
        language="hi",  # Hindi language
        reuse_existing=True  # Reuse existing assets if available
    )
    
    # Calculate timing
    total_scenes = len(custom_scenes)
    total_scene_duration = sum(scene.get("duration", 5) for scene in custom_scenes)
    total_pause_duration = config.scene_pause_duration * (total_scenes - 1)
    total_video_duration = total_scene_duration + total_pause_duration
    
    print(f"\n⏱️ Timing Breakdown:")
    print(f"   📊 Scene duration: {total_scene_duration}s")
    print(f"   ⏸️ Pause duration: {total_pause_duration:.1f}s")
    print(f"   🎬 Total video: {total_video_duration:.1f}s")
    print(f"   ⏸️ Pause between scenes: {config.scene_pause_duration:.1f}s")
    
    print(f"\n🎯 Configuration:")
    print(f"   📁 Output: {config.output_path}")
    print(f"   🎨 Style: {config.style}")
    print(f"   🌍 Language: {config.language}")
    print(f"   🎯 Prompt enhancement: {'Enabled' if config.enable_prompt_enhancement else 'Disabled'}")
    print(f"   🔍 Image validation: {'Enabled' if config.enable_image_validation else 'Disabled'}")
    
    # Generate the video
    try:
        print(f"\n🚀 Starting video generation...")
        generator = CartoonShortsGenerator(config)
        output_path = generator.generate()
        
        print(f"\n🎉 Video generated successfully!")
        print(f"📁 Output: {output_path}")
        print(f"⏱️ Duration: {total_video_duration:.1f}s")
        print(f"⏸️ Includes {total_scenes-1} pauses of {config.scene_pause_duration:.1f}s each")
        
    except Exception as e:
        print(f"\n❌ Error generating video: {e}")
        print(f"💡 Make sure you have set up your environment variables:")
        print(f"   - OPENAI_API_KEY")
        print(f"   - REPLICATE_API_KEY (for image generation)")
        print(f"   - ELEVENLABS_API_KEY (for voice synthesis)")

if __name__ == "__main__":
    main()
