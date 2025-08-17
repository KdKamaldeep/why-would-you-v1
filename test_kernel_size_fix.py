#!/usr/bin/env python3
"""
Test script to verify kernel size error handling
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

def test_kernel_size_fix():
    """Test that kernel size errors are properly handled"""
    print("🧪 Testing Kernel Size Error Handling")
    print("=" * 60)
    
    try:
        from core.coqui_voice_synthesizer import CoquiVoiceSynthesizer, CoquiVoiceConfig
        
        # Create output directory
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "test_kernel_fix.wav"
        
        # Initialize with English
        config = CoquiVoiceConfig(language="en", gpu=False)  # Use CPU for testing
        synthesizer = CoquiVoiceSynthesizer(config)
        
        # Test with very short text that might cause kernel size issues
        short_text = ["Hi", "Hello"]  # Very short text that should trigger padding
        
        print("🎙️  Testing with very short text...")
        result_path = synthesizer.synthesize_voice(short_text, str(output_path))
        
        if Path(result_path).exists():
            size_kb = Path(result_path).stat().st_size / 1024
            print(f"✅ Kernel size fix successful: {result_path} ({size_kb:.1f} KB)")
            return True
        else:
            print("❌ Kernel size fix failed - no output file created")
            return False
            
    except Exception as e:
        print(f"❌ Kernel size fix test failed: {e}")
        return False

def test_hindi_kernel_size_fix():
    """Test kernel size handling with Hindi text"""
    print("\n🧪 Testing Hindi Kernel Size Error Handling")
    print("=" * 60)
    
    try:
        from core.coqui_voice_synthesizer import CoquiVoiceSynthesizer, CoquiVoiceConfig
        
        # Create output directory
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "test_hindi_kernel_fix.wav"
        
        # Initialize with Hindi
        config = CoquiVoiceConfig(language="hi", gpu=False)  # Use CPU for testing
        synthesizer = CoquiVoiceSynthesizer(config)
        
        # Test with very short Hindi text
        short_hindi_text = ["नमस्ते", "हैलो"]  # Very short Hindi text that should trigger padding
        
        print("🎙️  Testing with very short Hindi text...")
        result_path = synthesizer.synthesize_voice(short_hindi_text, str(output_path))
        
        if Path(result_path).exists():
            size_kb = Path(result_path).stat().st_size / 1024
            print(f"✅ Hindi kernel size fix successful: {result_path} ({size_kb:.1f} KB)")
            return True
        else:
            print("❌ Hindi kernel size fix failed - no output file created")
            return False
            
    except Exception as e:
        print(f"❌ Hindi kernel size fix test failed: {e}")
        return False

def main():
    """Run kernel size tests"""
    print("🧪 Kernel Size Error Handling Test Suite")
    print("=" * 80)
    
    tests = [
        ("English Kernel Size Fix", test_kernel_size_fix),
        ("Hindi Kernel Size Fix", test_hindi_kernel_size_fix),
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
        print("🎉 All kernel size tests passed! Error handling is working correctly.")
    else:
        print("⚠️  Some kernel size tests failed. Please check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
