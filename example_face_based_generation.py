#!/usr/bin/env python3
"""
Example: Face-Based Character Generation in Cartoon Videos

This script demonstrates how to use the face-based character generation
feature in your cartoon video generation system.
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))



def demonstrate_face_based_generation():
    """Demonstrate face-based character generation."""
    print("🎭 Face-Based Character Generation Demo")
    print("=" * 50)
    
    # Example 1: Basic face-based generation
    print("\n📝 Example 1: Basic face-based generation")
    print("Command:")
    print('python -m src.interfaces.simple_cartoon_generator --prompt "A brave lion opens a smoothie shop" --character-faces example_characters.json')
    
    # Example 2: With custom style
    print("\n📝 Example 2: With custom style")
    print("Command:")
    print('python -m src.interfaces.simple_cartoon_generator --prompt "A robot learns to dance" --style anime --character-faces example_characters.json')
    
    # Example 3: With storyboard
    print("\n📝 Example 3: With storyboard")
    print("Command:")
    print('python -m src.interfaces.simple_cartoon_generator --prompt "Magic forest adventure" --storyboard storyboards/example.json')
    
    # Example 4: Longer duration
    print("\n📝 Example 4: Longer duration")
    print("Command:")
    print('python -m src.interfaces.simple_cartoon_generator --prompt "A princess and wizard go on an adventure" --duration 60')

def create_example_storyboard():
    """Create an example storyboard with character references."""
    storyboard = {
        "title": "The Lion's Smoothie Shop",
        "description": "A brave lion opens a smoothie shop in the jungle",
        "scene_duration": 8,
        "cast": [
            {"name": "Lion", "role": "main character", "face": "source-face-images/sardar.png"},
            {"name": "Robot", "role": "helper", "face": "source-face-images/robot_face.jpg"},
            {"name": "Princess", "role": "customer"}  # No face specified - will auto-generate
        ],
        "scenes": [
            {
                "title": "Lion's Dream",
                "visual_prompt": "A lion standing in front of an empty shop space, looking determined",
                "subtitle": "Once upon a time, a brave lion had a dream...",
                "characters": ["Lion"]
            },
            {
                "title": "Building the Shop",
                "visual_prompt": "A lion and robot working together to build a smoothie shop",
                "subtitle": "With the help of his robot friend, the lion built his shop...",
                "characters": ["Lion", "Robot"]
            },
            {
                "title": "First Customer",
                "visual_prompt": "A princess visiting the lion's smoothie shop, looking excited",
                "subtitle": "The first customer was a beautiful princess...",
                "characters": ["Lion", "Princess"]
            }
        ]
    }
    
    # Create storyboards directory if it doesn't exist
    Path("storyboards").mkdir(exist_ok=True)
    
    with open("storyboards/example.json", "w") as f:
        json.dump(storyboard, f, indent=2)
    
    print("✅ Created storyboards/example.json")
    return storyboard

def show_character_matching_examples():
    """Show examples of how character matching works."""
    print("\n🎯 Character Matching Examples")
    print("=" * 40)
    
    examples = [
        ("A brave lion opens a smoothie shop", "Lion"),
        ("A robot learns to dance", "Robot"),
        ("A princess visits the castle", "Princess"),
        ("A wizard casts a spell", "Wizard"),
        ("A dragon guards the treasure", "Dragon"),
        ("The king and queen rule the kingdom", "King"),  # Will match first available
        ("A cat and dog become friends", "Cat"),  # Will match first available
    ]
    
    for prompt, expected_character in examples:
        print(f"📝 Prompt: '{prompt}'")
        print(f"🎭 Expected character: {expected_character}")
        print()

def main():
    """Main demonstration function."""
    print("🎭 Face-Based Character Generation System")
    print("=" * 60)
    
    # Create example files
    print("\n📁 Creating example files...")
    storyboard = create_example_storyboard()
    
    # Show character matching examples
    show_character_matching_examples()
    
    # Demonstrate usage
    demonstrate_face_based_generation()
    
    print("\n🎯 How it works:")
    print("1. The system reads the cast array from the storyboard JSON")
    print("2. Characters with a 'face' field use face-based generation")
    print("3. Characters without a 'face' field use auto-generated faces")
    print("4. The system automatically detects faces in the reference images")
    print("5. Face-based generation falls back to standard generation if needed")
    
    print("\n💡 Tips:")
    print("- Add a 'face' field to cast members to use custom faces")
    print("- Ensure face images are well-lit and clearly show the face")
    print("- Characters without 'face' field will use auto-generated faces")
    print("- The system supports mixing custom and auto-generated faces")
    
    print("\n🚀 Ready to generate face-based cartoons!")
    print("Try running one of the example commands above.")

if __name__ == "__main__":
    main()
