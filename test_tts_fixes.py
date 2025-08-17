#!/usr/bin/env python3
"""
Test script to verify TTS fixes for GPT2InferenceModel warnings and Hindi language support
"""

import os
import sys
import logging
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_english_tts():
    """Test English TTS to ensure no GPT2InferenceModel warnings"""
    print("🔊 Testing English TTS (GPT2InferenceModel warning suppression)")
    print("=" * 60)
    
    try:
        from core.coqui_voice_synthesizer import CoquiVoiceSynthesizer, CoquiVoiceConfig
        
        # Create output directory
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "test_english.wav"
        
        # Initialize with English
        config = CoquiVoiceConfig(language="en", gpu=False)  # Use CPU for testing
        synthesizer = CoquiVoiceSynthesizer(config)
        
        # Test text
        test_text = [
            "Hello! This is a test of the English TTS system.",
            "We are checking that GPT2InferenceModel warnings are properly suppressed."
        ]
        
        print("🎙️  Generating English audio...")
        result_path = synthesizer.synthesize_voice(test_text, str(output_path))
        
        if Path(result_path).exists():
            size_kb = Path(result_path).stat().st_size / 1024
            print(f"✅ English TTS successful: {result_path} ({size_kb:.1f} KB)")
            return True
        else:
            print("❌ English TTS failed - no output file created")
            return False
            
    except Exception as e:
        print(f"❌ English TTS test failed: {e}")
        return False

def test_hindi_tts():
    """Test Hindi TTS to ensure proper language support"""
    print("\n🔊 Testing Hindi TTS (multilingual support)")
    print("=" * 60)
    
    try:
        from core.coqui_voice_synthesizer import CoquiVoiceSynthesizer, CoquiVoiceConfig
        
        # Create output directory
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "test_hindi.wav"
        
        # Initialize with Hindi
        config = CoquiVoiceConfig(language="hi", gpu=False)  # Use CPU for testing
        synthesizer = CoquiVoiceSynthesizer(config)
        
        # Test Hindi text (Devanagari script)
        test_text = [
            "नमस्ते! यह हिंदी भाषा में आवाज़ जनरेशन का परीक्षण है।",
            "हम यह सुनिश्चित कर रहे हैं कि हिंदी भाषा सही तरीके से काम कर रही है।"
        ]
        
        print("🎙️  Generating Hindi audio...")
        result_path = synthesizer.synthesize_voice(test_text, str(output_path))
        
        if Path(result_path).exists():
            size_kb = Path(result_path).stat().st_size / 1024
            print(f"✅ Hindi TTS successful: {result_path} ({size_kb:.1f} KB)")
            return True
        else:
            print("❌ Hindi TTS failed - no output file created")
            return False
            
    except Exception as e:
        print(f"❌ Hindi TTS test failed: {e}")
        return False

def test_speaker_discovery():
    """Test speaker WAV file discovery for different languages"""
    print("\n🔍 Testing Speaker WAV Discovery")
    print("=" * 60)
    
    try:
        from core.coqui_voice_synthesizer import CoquiVoiceSynthesizer, CoquiVoiceConfig
        
        # Test Hindi speaker discovery
        config = CoquiVoiceConfig(language="hi", gpu=False)
        synthesizer = CoquiVoiceSynthesizer(config)
        
        # Test the discovery method directly
        hindi_speaker = synthesizer._discover_speaker_wav("hi")
        if hindi_speaker:
            print(f"✅ Hindi speaker found: {hindi_speaker}")
        else:
            print("⚠️  No Hindi speaker WAV found (this is normal if no custom voice files exist)")
        
        # Test English speaker discovery
        english_speaker = synthesizer._discover_speaker_wav("en")
        if english_speaker:
            print(f"✅ English speaker found: {english_speaker}")
        else:
            print("⚠️  No English speaker WAV found (this is normal if no custom voice files exist)")
        
        return True
        
    except Exception as e:
        print(f"❌ Speaker discovery test failed: {e}")
        return False

def test_warning_suppression():
    """Test that GPT2InferenceModel warnings are properly suppressed"""
    print("\n🔇 Testing Warning Suppression")
    print("=" * 60)
    
    try:
        import warnings
        
        # Capture warnings during TTS initialization
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            from core.coqui_voice_synthesizer import CoquiVoiceSynthesizer, CoquiVoiceConfig
            
            # Initialize TTS (this should trigger warnings if not suppressed)
            config = CoquiVoiceConfig(language="en", gpu=False)
            synthesizer = CoquiVoiceSynthesizer(config)
            
            # Check for GPT2InferenceModel warnings
            gpt2_warnings = [warning for warning in w if "GPT2InferenceModel" in str(warning.message)]
            
            if gpt2_warnings:
                print(f"⚠️  Found {len(gpt2_warnings)} GPT2InferenceModel warnings:")
                for warning in gpt2_warnings:
                    print(f"   - {warning.message}")
                return False
            else:
                print("✅ No GPT2InferenceModel warnings detected - suppression working correctly")
                return True
                
    except Exception as e:
        print(f"❌ Warning suppression test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 TTS Fixes Verification Test Suite")
    print("=" * 80)
    
    tests = [
        ("Warning Suppression", test_warning_suppression),
        ("Speaker Discovery", test_speaker_discovery),
        ("English TTS", test_english_tts),
        ("Hindi TTS", test_hindi_tts),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! TTS fixes are working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
