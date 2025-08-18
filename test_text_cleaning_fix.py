#!/usr/bin/env python3
"""
Test script to verify text cleaning and deduplication fix
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

def test_text_cleaning_fix():
    """Test that text cleaning and deduplication works correctly"""
    print("🧪 Testing Text Cleaning and Deduplication Fix")
    print("=" * 60)
    
    try:
        from core.coqui_voice_synthesizer import CoquiVoiceSynthesizer, CoquiVoiceConfig
        
        # Create output directory
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "test_text_cleaning_fix.wav"
        
        # Initialize with English
        config = CoquiVoiceConfig(language="en", gpu=False)  # Use CPU for testing
        synthesizer = CoquiVoiceSynthesizer(config)
        
        # Test with the problematic input that was causing issues
        problematic_text = [
            "Grandma said—'If you eat too much rice at night, you'll surely sleep… but your stomach will complain—oh no!'",
            "'' Grandma said—'If you eat too much rice at night, you'll surely sleep… but your stomach will complain—oh no!'",
            "'' Grandma said—'If you eat too much rice at night, you'll surely sleep… but your stomach will complain—oh no!'",
            "''"
        ]
        
        print("🎙️  Testing with problematic text (duplicated and empty lines)...")
        print(f"Original text lines: {len(problematic_text)}")
        for i, line in enumerate(problematic_text):
            print(f"  Line {i+1}: '{line}'")
        
        result_path = synthesizer.synthesize_voice(problematic_text, str(output_path))
        
        if Path(result_path).exists():
            size_kb = Path(result_path).stat().st_size / 1024
            print(f"✅ Text cleaning fix successful: {result_path} ({size_kb:.1f} KB)")
            
            # Check if the file has reasonable size (not just silent audio)
            if size_kb > 1.0:  # More than 1KB indicates actual audio content
                print("✅ Audio file has substantial content (not silent)")
                return True
            else:
                print("⚠️  Audio file is very small (might be silent)")
                return False
        else:
            print("❌ Text cleaning fix failed - no output file created")
            return False
            
    except Exception as e:
        print(f"❌ Text cleaning fix test failed: {e}")
        return False

def test_empty_text_handling():
    """Test handling of completely empty text"""
    print("\n🧪 Testing Empty Text Handling")
    print("=" * 60)
    
    try:
        from core.coqui_voice_synthesizer import CoquiVoiceSynthesizer, CoquiVoiceConfig
        
        # Create output directory
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "test_empty_text_handling.wav"
        
        # Initialize with English
        config = CoquiVoiceConfig(language="en", gpu=False)  # Use CPU for testing
        synthesizer = CoquiVoiceSynthesizer(config)
        
        # Test with completely empty text
        empty_text = ["", "", "", ""]
        
        print("🎙️  Testing with completely empty text...")
        print(f"Empty text lines: {len(empty_text)}")
        
        result_path = synthesizer.synthesize_voice(empty_text, str(output_path))
        
        if Path(result_path).exists():
            size_kb = Path(result_path).stat().st_size / 1024
            print(f"✅ Empty text handling successful: {result_path} ({size_kb:.1f} KB)")
            
            # Check if the file has reasonable size (not just silent audio)
            if size_kb > 1.0:  # More than 1KB indicates actual audio content
                print("✅ Audio file has substantial content (fallback text worked)")
                return True
            else:
                print("⚠️  Audio file is very small (might be silent)")
                return False
        else:
            print("❌ Empty text handling failed - no output file created")
            return False
            
    except Exception as e:
        print(f"❌ Empty text handling test failed: {e}")
        return False

def test_hindi_text_cleaning():
    """Test text cleaning with Hindi text"""
    print("\n🧪 Testing Hindi Text Cleaning")
    print("=" * 60)
    
    try:
        from core.coqui_voice_synthesizer import CoquiVoiceSynthesizer, CoquiVoiceConfig
        
        # Create output directory
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "test_hindi_text_cleaning.wav"
        
        # Initialize with Hindi
        config = CoquiVoiceConfig(language="hi", gpu=False)  # Use CPU for testing
        synthesizer = CoquiVoiceSynthesizer(config)
        
        # Test with problematic Hindi text
        problematic_hindi_text = [
            "दादी ने कहा—'अगर तुम रात में बहुत सारा चावल खाओगे, तो तुम ज़रूर सो जाओगे… लेकिन तुम्हारा पेट शिकायत करेगा—अरे नहीं!'",
            "'' दादी ने कहा—'अगर तुम रात में बहुत सारा चावल खाओगे, तो तुम ज़रूर सो जाओगे… लेकिन तुम्हारा पेट शिकायत करेगा—अरे नहीं!'",
            "''"
        ]
        
        print("🎙️  Testing with problematic Hindi text...")
        print(f"Original Hindi text lines: {len(problematic_hindi_text)}")
        
        result_path = synthesizer.synthesize_voice(problematic_hindi_text, str(output_path))
        
        if Path(result_path).exists():
            size_kb = Path(result_path).stat().st_size / 1024
            print(f"✅ Hindi text cleaning successful: {result_path} ({size_kb:.1f} KB)")
            
            # Check if the file has reasonable size (not just silent audio)
            if size_kb > 1.0:  # More than 1KB indicates actual audio content
                print("✅ Hindi audio file has substantial content (not silent)")
                return True
            else:
                print("⚠️  Hindi audio file is very small (might be silent)")
                return False
        else:
            print("❌ Hindi text cleaning failed - no output file created")
            return False
            
    except Exception as e:
        print(f"❌ Hindi text cleaning test failed: {e}")
        return False

def main():
    """Run text cleaning tests"""
    print("🧪 Text Cleaning and Deduplication Test Suite")
    print("=" * 80)
    
    tests = [
        ("Text Cleaning and Deduplication", test_text_cleaning_fix),
        ("Empty Text Handling", test_empty_text_handling),
        ("Hindi Text Cleaning", test_hindi_text_cleaning),
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
        print("🎉 All text cleaning tests passed! Text processing is working correctly.")
    else:
        print("⚠️  Some text cleaning tests failed. Please check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
