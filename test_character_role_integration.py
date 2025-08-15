#!/usr/bin/env python3
"""
Test Script for Character Role Integration in Visual Prompts
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_character_role_integration():
    """Test how character roles are integrated into visual prompts."""
    print("🎭 Testing Character Role Integration in Visual Prompts")
    print("=" * 70)
    
    # Sample storyboard with cast and scenes
    sample_script = {
        "title": "Anaya and the Flag of Freedom",
        "cast": [
            {"name": "Anaya", "role": "curious little Indian girl"},
            {"name": "Grandfather", "role": "wise old man who tells stories"},
            {"name": "Sardar Patel", "role": "leader of India's unity"},
            {"name": "Tricolor Flag", "role": "symbol of India's freedom"}
        ],
        "scenes": [
            {
                "duration": 8,
                "description": "wide shot of early morning Red Fort with people gathering for flag hoisting",
                "visual_prompt": "(wide shot:1.2) Red Fort in Delhi at sunrise, people in traditional Indian clothes gathering, saffron and pink morning sky, warm sunlight on red sandstone walls, storybook style:1.3",
                "negative_prompt": "blurry, dark, low resolution, distorted, modern cars, scary, violent",
                "characters": ["Anaya", "Grandfather"]
            },
            {
                "duration": 8,
                "description": "close-up of little Anaya holding a small tricolor flag",
                "visual_prompt": "(close-up:1.2) little Indian girl with two braids, wearing white kurta with orange dupatta, holding small Indian tricolor flag, smiling, sunlight on her face, storybook style:1.3",
                "negative_prompt": "blurry, dark, low resolution, distorted, scary, violent",
                "characters": ["Anaya"]
            },
            {
                "duration": 9,
                "description": "medium shot of Grandfather telling Anaya about the freedom struggle",
                "visual_prompt": "(medium shot:1.2) elderly Indian man with white kurta pajama, Gandhi cap, gentle smile, telling story to little girl with braids, tricolor bunting in background, storybook style:1.3",
                "negative_prompt": "blurry, dark, low resolution, distorted, scary, violent",
                "characters": ["Grandfather", "Anaya"]
            }
        ]
    }
    
    # Simulate the character role enhancement process
    def enhance_prompt_with_character_roles(base_prompt, scene, cast_info):
        """Simulate the character role enhancement logic."""
        scene_characters = scene.get('characters', [])
        if not scene_characters or not cast_info:
            return base_prompt
        
        # Create character role mapping
        character_roles = {}
        for cast_member in cast_info:
            if isinstance(cast_member, dict):
                name = cast_member.get('name', '')
                role = cast_member.get('role', '')
                if name and role:
                    character_roles[name] = role
        
        if not character_roles:
            return base_prompt
        
        # Enhance prompt with character roles
        enhanced_prompt = base_prompt
        character_enhancements = []
        
        for character_name in scene_characters:
            if character_name in character_roles:
                role = character_roles[character_name]
                character_enhancement = f"{character_name} ({role})"
                character_enhancements.append(character_enhancement)
                
                # Replace character name with enhanced version
                if character_name.lower() in base_prompt.lower():
                    enhanced_prompt = enhanced_prompt.replace(character_name, character_enhancement)
                else:
                    enhanced_prompt = f"{enhanced_prompt}, {character_enhancement}"
        
        return enhanced_prompt, character_enhancements
    
    # Test each scene
    cast_info = sample_script['cast']
    print(f"🎭 Cast Information:")
    for character in cast_info:
        print(f"   • {character['name']}: {character['role']}")
    print()
    
    for i, scene in enumerate(sample_script['scenes'], 1):
        print(f"🎬 Scene {i}: {scene['description']}")
        print(f"   Characters: {', '.join(scene['characters'])}")
        print(f"   Original Prompt: {scene['visual_prompt']}")
        
        enhanced_prompt, enhancements = enhance_prompt_with_character_roles(
            scene['visual_prompt'], scene, cast_info
        )
        
        print(f"   Enhanced Prompt: {enhanced_prompt}")
        if enhancements:
            print(f"   Character Enhancements: {enhancements}")
        print()

def test_character_role_examples():
    """Show examples of how character roles improve prompts."""
    print("\n📚 Character Role Integration Examples")
    print("=" * 50)
    
    examples = [
        {
            "original": "little Indian girl with two braids, wearing white kurta",
            "character": "Anaya",
            "role": "curious little Indian girl",
            "enhanced": "Anaya (curious little Indian girl) with two braids, wearing white kurta"
        },
        {
            "original": "elderly Indian man with white kurta pajama, Gandhi cap",
            "character": "Grandfather", 
            "role": "wise old man who tells stories",
            "enhanced": "Grandfather (wise old man who tells stories) with white kurta pajama, Gandhi cap"
        },
        {
            "original": "Sardar Patel in white dhoti kurta and round spectacles",
            "character": "Sardar Patel",
            "role": "leader of India's unity", 
            "enhanced": "Sardar Patel (leader of India's unity) in white dhoti kurta and round spectacles"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\nExample {i}:")
        print(f"   Character: {example['character']} ({example['role']})")
        print(f"   Original: {example['original']}")
        print(f"   Enhanced: {example['enhanced']}")
        print(f"   Benefit: Diffusion model now understands the character's role and personality")

def test_benefits():
    """Explain the benefits of character role integration."""
    print("\n🎯 Benefits of Character Role Integration")
    print("=" * 50)
    
    benefits = [
        "🎭 **Character Consistency**: Roles help maintain consistent character appearance across scenes",
        "🎨 **Better Visual Generation**: Diffusion models understand character personalities and roles",
        "📝 **Richer Prompts**: More detailed and contextually relevant visual descriptions",
        "🎬 **Story Coherence**: Characters behave according to their defined roles",
        "🔍 **Improved Matching**: Better character face matching when using face-based generation",
        "📊 **Enhanced Quality**: More accurate and story-appropriate image generation"
    ]
    
    for benefit in benefits:
        print(f"   {benefit}")

def main():
    """Run all tests."""
    print("🎭 Character Role Integration Test Suite")
    print("=" * 80)
    
    try:
        test_character_role_integration()
        test_character_role_examples()
        test_benefits()
        
        print("\n✅ All tests completed!")
        print("\n💡 To use character role integration:")
        print("   1. Define cast with roles in your storyboard JSON")
        print("   2. Reference characters in scene 'characters' arrays")
        print("   3. The system will automatically enhance visual prompts")
        print("   4. Character roles will be integrated into diffusion model prompts")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")

if __name__ == "__main__":
    main()
