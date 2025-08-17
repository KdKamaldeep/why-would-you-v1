#!/usr/bin/env python3
"""
Test script to verify improved kernel size error handling
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

def test_improved_kernel_fix():
    """Test the improved kernel size error handling"""
    print("🧪 Testing Improved Kernel Size Error Handling")
    print("=" * 60)
    
    try:
        from core.coqui_voice_synthesizer import CoquiVoiceSynthesizer, CoquiVoiceConfig
        
        # Create output directory
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "test_improved_kernel_fix.wav"
        
        # Initialize with English
        config = CoquiVoiceConfig(language="en", gpu=False)  # Use CPU for testing
        synthesizer = CoquiVoiceSynthesizer(config)
        
        # Test with extremely short text that should trigger the new padding
        very_short_text = ["Hi"]  # Just 2 characters - should trigger padding
        
        print("🎙️  Testing with extremely short text (2 characters)...")
        print(f"Original text: {very_short_text}")
        
        result_path = synthesizer.synthesize_voice(very_short_text, str(output_path))
        
        if Path(result_path).exists():
            size_kb = Path(result_path).stat().st_size / 1024
            print(f"✅ Improved kernel size fix successful: {result_path} ({size_kb:.1f} KB)")
            
            # Check if the file has reasonable size (not just silent audio)
            if size_kb > 1.0:  # More than 1KB indicates actual audio content
                print("✅ Audio file has substantial content (not silent)")
                return True
            else:
                print("⚠️  Audio file is very small (might be silent)")
                return False
        else:
            print("❌ Improved kernel size fix failed - no output file created")
            return False
            
    except Exception as e:
        print(f"❌ Improved kernel size fix test failed: {e}")
        return False

def test_hindi_improved_kernel_fix():
    """Test improved kernel size handling with Hindi text"""
    print("\n🧪 Testing Hindi Improved Kernel Size Error Handling")
    print("=" * 60)
    
    try:
        from core.coqui_voice_synthesizer import CoquiVoiceSynthesizer, CoquiVoiceConfig
        
        # Create output directory
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "test_hindi_improved_kernel_fix.wav"
        
        # Initialize with Hindi
        config = CoquiVoiceConfig(language="hi", gpu=False)  # Use CPU for testing
        synthesizer = CoquiVoiceSynthesizer(config)
        
        # Test with extremely short Hindi text
        very_short_hindi_text = ["नमस्ते"]  # Just 6 characters - should trigger padding
        
        print("🎙️  Testing with extremely short Hindi text (6 characters)...")
        print(f"Original text: {very_short_hindi_text}")
        
        result_path = synthesizer.synthesize_voice(very_short_hindi_text, str(output_path))
        
        if Path(result_path).exists():
            size_kb = Path(result_path).stat().st_size / 1024
            print(f"✅ Hindi improved kernel size fix successful: {result_path} ({size_kb:.1f} KB)")
            
            # Check if the file has reasonable size (not just silent audio)
            if size_kb > 1.0:  # More than 1KB indicates actual audio content
                print("✅ Hindi audio file has substantial content (not silent)")
                return True
            else:
                print("⚠️  Hindi audio file is very small (might be silent)")
                return False
        else:
            print("❌ Hindi improved kernel size fix failed - no output file created")
            return False
            
    except Exception as e:
        print(f"❌ Hindi improved kernel size fix test failed: {e}")
        return False

def main():
    """Run improved kernel size tests"""
    print("🧪 Improved Kernel Size Error Handling Test Suite")
    print("=" * 80)
    
    tests = [
        ("English Improved Kernel Size Fix", test_improved_kernel_fix),
        ("Hindi Improved Kernel Size Fix", test_hindi_improved_kernel_fix),
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
        print("🎉 All improved kernel size tests passed! Enhanced error handling is working correctly.")
    else:
        print("⚠️  Some improved kernel size tests failed. Please check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
