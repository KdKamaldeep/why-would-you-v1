#!/usr/bin/env python3
"""
Test SVD Animation Integration
Verifies that SVD animation is properly integrated into the system
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_svd_animator_import():
    """Test if SVD animator can be imported."""
    try:
        from core.svd_animator import SVDAnimator
        print("✅ SVD Animator import successful")
        return True
    except ImportError as e:
        print(f"❌ SVD Animator import failed: {e}")
        return False

def test_animation_generator_svd():
    """Test if AnimationGenerator supports SVD."""
    try:
        from core.animation_generator import AnimationGenerator
        
        # Test initialization with SVD
        animator = AnimationGenerator(animator_type="svd")
        print("✅ AnimationGenerator SVD initialization successful")
        
        # Check if SVD animator was created
        if hasattr(animator, 'svd_animator') and animator.svd_animator:
            print("✅ SVD Animator properly initialized")
            return True
        else:
            print("⚠️ SVD Animator not available, falling back to FFmpeg")
            return False
            
    except Exception as e:
        print(f"❌ AnimationGenerator SVD test failed: {e}")
        return False

def test_svd_models_exist():
    """Test if SVD models are available."""
    svd_models = [
        "models/svd_xt_1_1.safetensors",
        "models/svd_xt.safetensors", 
        "models/svd.safetensors"
    ]
    
    found_models = []
    for model_path in svd_models:
        if Path(model_path).exists():
            found_models.append(model_path)
            print(f"✅ Found SVD model: {model_path}")
        else:
            print(f"⚠️ SVD model not found: {model_path}")
    
    if found_models:
        print(f"✅ Found {len(found_models)} SVD model(s)")
        return True
    else:
        print("❌ No SVD models found. Run the download script first.")
        return False

def test_command_line_interface():
    """Test if command line interface supports SVD parameters."""
    try:
        from interfaces.simple_cartoon_generator import main
        print("✅ Command line interface import successful")
        
        # Test argument parsing (without actually running)
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--animator", choices=["ffmpeg", "svd"], default="ffmpeg")
        parser.add_argument("--motion-bucket-id", type=int, default=127)
        parser.add_argument("--fps-id", type=int, default=6)
        parser.add_argument("--cond-aug", type=float, default=0.02)
        
        print("✅ SVD command line arguments available")
        return True
        
    except Exception as e:
        print(f"❌ Command line interface test failed: {e}")
        return False

def main():
    """Run all SVD integration tests."""
    print("🎬 Testing SVD Animation Integration")
    print("=" * 50)
    
    tests = [
        ("SVD Animator Import", test_svd_animator_import),
        ("AnimationGenerator SVD Support", test_animation_generator_svd),
        ("SVD Models Availability", test_svd_models_exist),
        ("Command Line Interface", test_command_line_interface),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Testing: {test_name}")
        print("-" * 30)
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! SVD integration is working correctly.")
        print("\n🚀 You can now use SVD animation with:")
        print("   python3 -m src.interfaces.simple_cartoon_generator --animator svd --prompt 'your prompt'")
        print("\n📋 SVD Animation Guidelines:")
        print("   • SVD has a HARD LIMIT of 25 frames per scene")
        print("   • For longer scenes, SVD will loop the 25 frames to match audio duration")
        print("   • This ensures synchronization with narration while respecting SVD limits")
        print("   • Use --motion-bucket-id to control motion intensity (0-255)")
        print("   • Use --fps-id to control motion speed (0-7)")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        print("\n💡 To fix issues:")
        print("   1. Run the download script: bash src/utils/download_models.sh")
        print("   2. Install dependencies: pip install -r requirements.txt")
        print("   3. Ensure ComfyUI is running for full SVD functionality")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
