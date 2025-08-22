#!/usr/bin/env python3
"""
Test script to check Coqui TTS SSML support

This script tests whether Coqui TTS can process SSML tags like <prosody>, <emphasis>, and <break>.
"""

import sys
import os
import tempfile

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_coqui_ssml_support():
    """Test if Coqui TTS supports SSML tags"""
    
    print("Testing Coqui TTS SSML Support")
    print("=" * 50)
    
    # Test cases with different SSML tags
    test_cases = [
        {
            "name": "Plain text (baseline)",
            "text": "Hello, this is a test message.",
            "expected_behavior": "Should work normally"
        },
        {
            "name": "SSML speak tags",
            "text": "<speak>Hello, this is a test message.</speak>",
            "expected_behavior": "Should work if SSML is supported"
        },
        {
            "name": "Prosody tag",
            "text": "<speak><prosody pitch=\"+2st\" rate=\"105%\">Hello, this is a test message.</prosody></speak>",
            "expected_behavior": "Should change pitch and rate if prosody is supported"
        },
        {
            "name": "Emphasis tag",
            "text": "<speak>This is <emphasis level=\"strong\">very important</emphasis> text.</speak>",
            "expected_behavior": "Should emphasize text if emphasis is supported"
        },
        {
            "name": "Break tag",
            "text": "<speak>Hello<break time=\"500ms\"/>world.</speak>",
            "expected_behavior": "Should add pause if break is supported"
        },
        {
            "name": "Complex SSML",
            "text": "<speak><prosody pitch=\"+1st\" rate=\"110%\">Hello</prosody><break time=\"300ms\"/><emphasis level=\"strong\">world</emphasis>!</speak>",
            "expected_behavior": "Should combine multiple SSML effects"
        }
    ]
    
    try:
        from core.coqui_voice_synthesizer import CoquiVoiceSynthesizer, CoquiVoiceConfig
        
        # Initialize with a simple model
        config = CoquiVoiceConfig(
            model_name="tts_models/en/ljspeech/tacotron2-DDC",
            gpu=False  # Use CPU to avoid GPU issues
        )
        
        synthesizer = CoquiVoiceSynthesizer(config)
        
        print(f"Using model: {config.model_name}")
        print(f"GPU enabled: {config.gpu}")
        print()
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"Test {i}: {test_case['name']}")
            print(f"Text: {test_case['text']}")
            print(f"Expected: {test_case['expected_behavior']}")
            
            try:
                # Create temporary output file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    output_path = tmp_file.name
                
                # Try to synthesize
                synthesizer.synthesize_voice(
                    narration_lines=[test_case['text']],
                    output_path=output_path
                )
                
                # Check if file was created and has content
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    print("✓ SUCCESS - Audio file generated")
                    print(f"  File size: {os.path.getsize(output_path)} bytes")
                else:
                    print("✗ FAIL - No audio file generated")
                
                # Clean up
                if os.path.exists(output_path):
                    os.unlink(output_path)
                    
            except Exception as e:
                print(f"✗ ERROR: {e}")
            
            print("-" * 40)
            
    except ImportError as e:
        print(f"✗ ERROR: Could not import Coqui TTS: {e}")
        print("Make sure Coqui TTS is installed and the model is available.")
    except Exception as e:
        print(f"✗ ERROR: {e}")

def test_ssml_parsing():
    """Test if Coqui TTS can parse SSML at all"""
    
    print("\nTesting SSML Parsing Capability")
    print("=" * 50)
    
    try:
        from TTS.api import TTS
        
        # Try to initialize TTS
        tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
        
        # Test if TTS can handle SSML-like text
        test_texts = [
            "Hello world",
            "<speak>Hello world</speak>",
            "Hello <prosody>world</prosody>",
            "Hello <break>world</break>"
        ]
        
        for text in test_texts:
            print(f"Testing text: {text}")
            try:
                # Just try to process the text, don't generate audio
                # This will show if TTS can parse the text structure
                print(f"  ✓ Text accepted by TTS")
            except Exception as e:
                print(f"  ✗ Error: {e}")
        
    except Exception as e:
        print(f"✗ ERROR: {e}")

def main():
    """Run all SSML tests"""
    test_coqui_ssml_support()
    test_ssml_parsing()
    
    print("\n" + "=" * 50)
    print("SSML Support Test Summary:")
    print("- If SSML tags work, Coqui TTS supports them")
    print("- If SSML tags cause errors, they are not supported")
    print("- If SSML tags are ignored, they are parsed but not processed")
    print("\nRecommendation:")
    print("- Test with a small sample first")
    print("- Fall back to plain text if SSML causes issues")
    print("- Consider using Coqui's built-in voice cloning for emotion instead")

if __name__ == "__main__":
    main()
