#!/usr/bin/env python3
"""
Demonstration of the Narration Converter

This script shows how to use the narration converter to transform
storyboard narration text with inline cues to both SSML and plain text formats,
and demonstrates the new voice property functionality.
"""

import sys
import os
import json

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.narration_converter import NarrationConverter

def demo_individual_conversions():
    """Demonstrate individual narration text conversions"""
    converter = NarrationConverter()
    
    print("Narration Converter Demonstration")
    print("=" * 60)
    
    # Show supported cues
    print("\nSupported Cues:")
    print("-" * 20)
    for cue in converter.get_supported_cues():
        print(f"  [{cue}]")
    
    # Test cases
    test_cases = [
        "Hut [nervous]: Uh-oh! The lightning is coming.",
        "The hero [excited] shouted: Victory is ours!",
        "The wise old man [slow, wise] spoke: Patience is a virtue.",
        "There was a [pause] moment of silence.",
        "This is [emphasis strong] very important!",
        "The [baby] giggled with joy.",
        "The [giant] roared loudly.",
        "The [robot] said: Beep boop.",
        "The [magical] fairy whispered secrets."
    ]
    
    print("\n\nConversion Examples:")
    print("-" * 30)
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\n{i}. Original: {test_text}")
        
        # Convert to SSML
        ssml_result = converter.convert_narration(test_text, use_ssml=True)
        print(f"   SSML:     {ssml_result}")
        
        # Convert to plain text
        plain_result = converter.convert_narration(test_text, use_ssml=False)
        print(f"   Plain:    {plain_result}")

def demo_voice_resolution():
    """Demonstrate voice file resolution"""
    converter = NarrationConverter()
    
    print("\n\nVoice File Resolution:")
    print("-" * 30)
    
    # Show available voices
    print("Available voices:")
    for voice in converter.get_available_voices():
        voice_file = converter.resolve_voice_file(voice)
        print(f"  {voice} -> {voice_file}")
    
    # Test voice resolution
    test_voices = [
        "hi-IN-SwaraNeural-female",
        "hi-IN-AnanyaNeural-kid", 
        "hi-IN-ArjunNeural-male",
        "female_voice",
        "male_hindi_speaker"
    ]
    
    print("\nVoice resolution tests:")
    for voice_name in test_voices:
        resolved_file = converter.resolve_voice_file(voice_name)
        if resolved_file:
            print(f"  ✓ '{voice_name}' -> {resolved_file}")
        else:
            print(f"  ✗ '{voice_name}' -> NOT FOUND")

def demo_storyboard_conversion():
    """Demonstrate storyboard conversion with voice properties"""
    converter = NarrationConverter()
    
    # Create a sample storyboard with narration cues and voice properties
    sample_storyboard = {
        "title": "Demo Story with Voices",
        "description": "A demonstration story with narration cues and voice properties",
        "scenes": [
            {
                "title": "Scene 1",
                "narration": "The hero [excited] ran into battle.",
                "voice": "hi-IN-ArjunNeural-male",
                "characters": ["Hero"]
            },
            {
                "title": "Scene 2", 
                "narration": "The wise wizard [slow, wise] cast a spell.",
                "voice": "hi-IN-SwaraNeural-female",
                "characters": ["Wizard"]
            },
            {
                "title": "Scene 3",
                "narration": "There was a [pause] moment of tension.",
                "characters": ["All"]
            },
            {
                "title": "Scene 4",
                "narration": "The [baby] giggled and the [giant] smiled.",
                "voice": "hi-IN-AnanyaNeural-kid",
                "characters": ["Baby", "Giant"]
            }
        ]
    }
    
    print("\n\nStoryboard Conversion with Voices:")
    print("-" * 40)
    
    # Convert to SSML
    ssml_storyboard = converter.convert_storyboard(sample_storyboard, use_ssml=True)
    
    # Convert to plain text
    plain_storyboard = converter.convert_storyboard(sample_storyboard, use_ssml=False)
    
    print("\nOriginal Storyboard:")
    print(json.dumps(sample_storyboard, indent=2))
    
    print("\nSSML Storyboard (with voice files):")
    print(json.dumps(ssml_storyboard, indent=2))
    
    print("\nPlain Text Storyboard (with voice files):")
    print(json.dumps(plain_storyboard, indent=2))

def demo_file_conversion():
    """Demonstrate file conversion using the example storyboard with voices"""
    converter = NarrationConverter()
    
    input_file = "storyboards/example_with_voices.json"
    
    if os.path.exists(input_file):
        print(f"\n\nFile Conversion Demo (with voices):")
        print("-" * 40)
        print(f"Input file: {input_file}")
        
        try:
            # Convert to SSML
            ssml_output = converter.convert_storyboard_file(input_file, use_ssml=True)
            print(f"SSML output: {ssml_output}")
            
            # Convert to plain text
            plain_output = converter.convert_storyboard_file(input_file, use_ssml=False)
            print(f"Plain output: {plain_output}")
            
            # Show a sample of the conversion
            with open(input_file, 'r', encoding='utf-8') as f:
                original_data = json.load(f)
            
            with open(ssml_output, 'r', encoding='utf-8') as f:
                ssml_data = json.load(f)
            
            with open(plain_output, 'r', encoding='utf-8') as f:
                plain_data = json.load(f)
            
            print("\nSample conversions (first scene):")
            print(f"Original:  {original_data['scenes'][0]['narration']}")
            print(f"Voice:     {original_data['scenes'][0].get('voice', 'None')}")
            print(f"SSML:      {ssml_data['scenes'][0]['narration']}")
            print(f"Voice File: {ssml_data['scenes'][0].get('voice_file', 'None')}")
            print(f"Plain:     {plain_data['scenes'][0]['narration']}")
            print(f"Voice File: {plain_data['scenes'][0].get('voice_file', 'None')}")
            
        except Exception as e:
            print(f"Error during file conversion: {e}")
    else:
        print(f"\nFile {input_file} not found, skipping file conversion demo")

def main():
    """Run the demonstration"""
    demo_individual_conversions()
    demo_voice_resolution()
    demo_storyboard_conversion()
    demo_file_conversion()
    
    print("\n" + "=" * 60)
    print("Demonstration Complete!")
    print("\nNew Features:")
    print("- Voice property support in scenes")
    print("- Automatic voice file resolution")
    print("- Voice file path mapping")
    print("\nUsage Tips:")
    print("- Add 'voice' property to scenes for different voices")
    print("- Voice files are automatically resolved from tts-speaker directory")
    print("- Use SSML format if your TTS engine supports it")
    print("- Use plain text format if SSML causes issues")
    print("- Test with a small sample first")
    print("\nCommand line usage:")
    print("  python src/utils/narration_converter.py input.json --plain")
    print("  python src/utils/narration_converter.py input.json")
    print("  python src/utils/narration_converter.py --list-cues")
    print("  python src/utils/narration_converter.py --list-voices")

if __name__ == "__main__":
    main()
