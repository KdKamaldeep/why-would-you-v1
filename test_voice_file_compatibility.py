#!/usr/bin/env python3
"""
Test script to check voice file compatibility with Coqui TTS
"""

import os
import sys
import tempfile

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_voice_file_compatibility():
    """Test if Coqui TTS can use different voice file formats"""
    
    print("Testing Voice File Compatibility with Coqui TTS")
    print("=" * 60)
    
    # Test voice files
    voice_files = [
        "tts-speaker/hi-IN-SwaraNeural-cheerful-female.mp3",
        "tts-speaker/female_voice.wav",
        "tts-speaker/male_hindi_speaker.wav"
    ]
    
    try:
        from core.coqui_voice_synthesizer import CoquiVoiceSynthesizer, CoquiVoiceConfig
        
        # Initialize with XTTS model for voice cloning
        config = CoquiVoiceConfig(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            gpu=False,  # Use CPU to avoid GPU issues
            language="en"
        )
        
        synthesizer = CoquiVoiceSynthesizer(config)
        
        print(f"Using model: {config.model_name}")
        print(f"GPU enabled: {config.gpu}")
        print()
        
        test_text = "Hello, this is a test of voice cloning."
        
        for voice_file in voice_files:
            if not os.path.exists(voice_file):
                print(f"❌ Voice file not found: {voice_file}")
                continue
                
            print(f"Testing voice file: {voice_file}")
            print(f"File exists: {os.path.exists(voice_file)}")
            print(f"File size: {os.path.getsize(voice_file)} bytes")
            
            try:
                # Create temporary output file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    output_path = tmp_file.name
                
                # Try to synthesize with this voice file
                synthesizer.synthesize_voice(
                    narration_lines=[test_text],
                    output_path=output_path,
                    voice_clone_audio=voice_file
                )
                
                # Check if file was created and has content
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    print("✅ SUCCESS - Audio file generated")
                    print(f"  Output file size: {os.path.getsize(output_path)} bytes")
                else:
                    print("❌ FAIL - No audio file generated")
                
                # Clean up
                if os.path.exists(output_path):
                    os.unlink(output_path)
                    
            except Exception as e:
                print(f"❌ ERROR: {e}")
            
            print("-" * 40)
            
    except ImportError as e:
        print(f"❌ ERROR: Could not import Coqui TTS: {e}")
        print("Make sure Coqui TTS is installed and the model is available.")
    except Exception as e:
        print(f"❌ ERROR: {e}")

def test_voice_resolution():
    """Test voice file resolution with different formats"""
    
    print("\nTesting Voice File Resolution")
    print("=" * 40)
    
    try:
        from utils.narration_converter import NarrationConverter
        
        converter = NarrationConverter()
        
        # Test voice names from the storyboard
        voice_names = [
            "hi-IN-SwaraNeural-cheerful-female",
            "hi-IN-AnanyaNeural-kid", 
            "hi-IN-ArjunNeural-male"
        ]
        
        for voice_name in voice_names:
            print(f"Resolving voice: {voice_name}")
            voice_file = converter.resolve_voice_file(voice_name)
            if voice_file:
                print(f"  ✅ Resolved to: {voice_file}")
                print(f"  File exists: {os.path.exists(voice_file)}")
                print(f"  File size: {os.path.getsize(voice_file)} bytes")
            else:
                print(f"  ❌ Could not resolve voice file")
            print()
            
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    test_voice_file_compatibility()
    test_voice_resolution()
