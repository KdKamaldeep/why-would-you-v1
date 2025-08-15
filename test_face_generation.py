#!/usr/bin/env python3
"""
Test script for face-based image generation
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_face_detection():
    """Test face detection functionality."""
    print("🧪 Testing Face Detection...")
    
    try:
        from core.face_image_generator import FaceImageGenerator
        
        # Initialize generator
        generator = FaceImageGenerator()
        
        # Test face detection with a sample image
        # You can replace this with an actual image path
        test_image_path = "source-face-images/sardar.png"
        
        if Path(test_image_path).exists():
            face_data = generator.detect_face(test_image_path)
            if face_data:
                face_region, landmarks = face_data
                print(f"✅ Face detected! Region shape: {face_region.shape}")
                print(f"✅ Landmarks found: {len(landmarks)} points")
                return True
            else:
                print("❌ No face detected in test image")
                return False
        else:
            print(f"⚠️ Test image not found: {test_image_path}")
            print("💡 Please add a test image to test face detection")
            return False
            
    except Exception as e:
        print(f"❌ Face detection test failed: {e}")
        return False

def test_diffusion_pipeline():
    """Test diffusion pipeline initialization."""
    print("\n🧪 Testing Diffusion Pipeline...")
    
    try:
        from core.face_image_generator import FaceImageGenerator
        
        # Initialize generator
        generator = FaceImageGenerator()
        
        if generator.pipe is not None:
            print("✅ Diffusion pipeline initialized successfully!")
            return True
        else:
            print("❌ Diffusion pipeline initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ Diffusion pipeline test failed: {e}")
        return False

def test_face_generation():
    """Test face-based image generation."""
    print("\n🧪 Testing Face-Based Image Generation...")
    
    try:
        from core.face_image_generator import FaceImageGenerator
        
        # Initialize generator
        generator = FaceImageGenerator()
        
        if generator.pipe is None:
            print("❌ Diffusion pipeline not available")
            return False
        
        # Test with a sample face image
        test_face_path = "test_face.jpg"
        
        if not Path(test_face_path).exists():
            print(f"⚠️ Test face image not found: {test_face_path}")
            print("💡 Please add a test face image to test generation")
            return False
        
        # Generate test image
        result = generator.generate_with_face(
            prompt="A cartoon character in a magical forest",
            face_image_path=test_face_path,
            output_path="test_generated_face.png"
        )
        
        if result:
            print("✅ Face-based image generation successful!")
            return True
        else:
            print("❌ Face-based image generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Face generation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🎭 Face-Based Image Generation Tests")
    print("=" * 40)
    
    # Create output directory
    output_dir = Path("output/face_generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run tests
    tests = [
        ("Face Detection", test_face_detection),
        ("Diffusion Pipeline", test_diffusion_pipeline),
        ("Face Generation", test_face_generation)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 40)
    print("📊 Test Results Summary:")
    print("=" * 40)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Face-based image generation is ready to use.")
        print("\n💡 Usage examples:")
        print("1. Interactive mode: python -m src.interfaces.face_generator_interface")
        print("2. Single image: python -m src.interfaces.face_generator_interface --mode single --prompt 'your prompt' --face-image 'path/to/face.jpg'")
        print("3. Video generation: python -m src.interfaces.face_generator_interface --mode video --prompt 'your prompt' --face-image 'path/to/face.jpg'")
    else:
        print("⚠️ Some tests failed. Please check the setup and dependencies.")
        print("\n💡 Troubleshooting:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Make sure you have a diffusion model available")
        print("3. Add a test face image to test_face.jpg")

if __name__ == "__main__":
    main()
