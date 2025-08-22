#!/usr/bin/env python3
"""
Voice Integration Demonstration

This script shows how to use the narration converter with the Coqui voice synthesizer
to generate audio using different voices for different scenes.
"""

import sys
import os
import json
import tempfile

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.narration_converter import NarrationConverter

def demo_voice_integration():
    """Demonstrate integration between narration converter and voice synthesizer"""
    
    print("Voice Integration Demonstration")
    print("=" * 50)
    
    # Initialize the converter
    converter = NarrationConverter()
    
    # Load and convert a storyboard
    input_file = "storyboards/example_with_voices.json"
    
    if not os.path.exists(input_file):
        print(f"File {input_file} not found. Please create it first.")
        return
    
    print(f"Loading storyboard: {input_file}")
    
    # Convert to plain text (safer for TTS)
    plain_output = converter.convert_storyboard_file(input_file, use_ssml=False)
    
    # Load the converted storyboard
    with open(plain_output, 'r', encoding='utf-8') as f:
        storyboard = json.load(f)
    
    print(f"Converted storyboard saved to: {plain_output}")
    
    # Show the voice mapping for each scene
    print("\nVoice Mapping for Scenes:")
    print("-" * 30)
    
    for i, scene in enumerate(storyboard['scenes'], 1):
        voice_name = scene.get('voice', 'None')
        voice_file = scene.get('voice_file', 'None')
        narration = scene.get('narration', 'No narration')
        
        print(f"\nScene {i}: {scene.get('title', 'Untitled')}")
        print(f"  Voice: {voice_name}")
        print(f"  Voice File: {voice_file}")
        print(f"  Narration: {narration[:50]}...")
    
    # Demonstrate how to use with Coqui TTS
    print("\n\nIntegration with Coqui TTS:")
    print("-" * 30)
    print("To use with Coqui TTS, you would:")
    print("1. Load the converted storyboard")
    print("2. For each scene, use the 'voice_file' property as voice_clone_audio")
    print("3. Use the 'narration' property as the text to synthesize")
    print("\nExample code:")
    
    example_code = '''
# Example integration code:
from src.core.coqui_voice_synthesizer import CoquiVoiceSynthesizer, CoquiVoiceConfig

# Initialize the synthesizer
config = CoquiVoiceConfig(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
    language="hi"  # or "en" for English
)
synthesizer = CoquiVoiceSynthesizer(config)

# Process each scene
for i, scene in enumerate(storyboard['scenes']):
    narration = scene['narration']
    voice_file = scene.get('voice_file')
    
    # Generate audio with the scene's voice
    output_path = f"output/scene_{i+1}.wav"
    synthesizer.synthesize_voice(
        narration_lines=[narration],
        output_path=output_path,
        voice_clone_audio=voice_file  # Use the scene's voice file
    )
    print(f"Generated: {output_path}")
'''
    
    print(example_code)

def demo_voice_file_resolution():
    """Demonstrate voice file resolution in detail"""
    
    print("\n\nVoice File Resolution Details:")
    print("-" * 30)
    
    converter = NarrationConverter()
    
    # Show all available voice files
    print("Available voice files in tts-speaker directory:")
    voice_files = converter._get_voice_files()
    for voice_name, file_path in voice_files.items():
        print(f"  {voice_name} -> {file_path}")
    
    # Test specific voice resolutions
    test_cases = [
        "hi-IN-SwaraNeural-female",
        "hi-IN-AnanyaNeural-kid",
        "hi-IN-ArjunNeural-male",
        "female_voice",
        "male_hindi_speaker"
    ]
    
    print("\nVoice resolution test cases:")
    for voice_name in test_cases:
        resolved = converter.resolve_voice_file(voice_name)
        status = "✓" if resolved else "✗"
        print(f"  {status} {voice_name} -> {resolved or 'NOT FOUND'}")

def demo_storyboard_processing():
    """Demonstrate processing a storyboard with voice properties"""
    
    print("\n\nStoryboard Processing Demo:")
    print("-" * 30)
    
    converter = NarrationConverter()
    
    # Create a simple test storyboard
    test_storyboard = {
        "title": "Test Story",
        "scenes": [
            {
                "title": "Scene 1",
                "narration": "Hello [excited] world!",
                "voice": "hi-IN-SwaraNeural-female"
            },
            {
                "title": "Scene 2", 
                "narration": "This is a [pause] test.",
                "voice": "hi-IN-AnanyaNeural-kid"
            }
        ]
    }
    
    print("Original storyboard:")
    print(json.dumps(test_storyboard, indent=2))
    
    # Convert to plain text
    converted = converter.convert_storyboard(test_storyboard, use_ssml=False)
    
    print("\nConverted storyboard (plain text):")
    print(json.dumps(converted, indent=2))
    
    # Convert to SSML
    converted_ssml = converter.convert_storyboard(test_storyboard, use_ssml=True)
    
    print("\nConverted storyboard (SSML):")
    print(json.dumps(converted_ssml, indent=2))

def main():
    """Run the voice integration demonstration"""
    
    demo_voice_file_resolution()
    demo_storyboard_processing()
    demo_voice_integration()
    
    print("\n" + "=" * 50)
    print("Voice Integration Demo Complete!")
    print("\nKey Points:")
    print("- Voice files are automatically resolved from tts-speaker directory")
    print("- Each scene can have its own voice for character differentiation")
    print("- The converter adds 'voice_file' property with the actual file path")
    print("- Use 'voice_file' as 'voice_clone_audio' parameter in Coqui TTS")
    print("- Plain text conversion is recommended for better TTS compatibility")
    print("\nNext Steps:")
    print("1. Test with a small storyboard first")
    print("2. Ensure all voice files exist in tts-speaker directory")
    print("3. Use the converted storyboard with your TTS pipeline")

if __name__ == "__main__":
    main()
