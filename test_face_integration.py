#!/usr/bin/env python3
"""
Test script for face-based character generation integration
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_cast_extraction():
    """Test extracting character faces from cast array."""
    print("🧪 Testing Cast Extraction...")
    
    try:
        from src.interfaces.simple_cartoon_generator import extract_character_faces_from_cast
        
        # Test cast with mixed face specifications
        test_cast = [
            {"name": "Lion", "role": "main character", "face": "source-face-images/sardar.png"},
            {"name": "Robot", "role": "helper", "face": "source-face-images/robot_face.jpg"},
            {"name": "Princess", "role": "customer"},  # No face
            {"name": "Wizard", "role": "magical advisor", "face": "source-face-images/wizard_face.jpg"}
        ]
        
        character_faces = extract_character_faces_from_cast(test_cast)
        
        print(f"✅ Extracted {len(character_faces)} character faces")
        for char, face in character_faces.items():
            print(f"   - {char}: {face}")
        
        # Verify expected results
        expected_faces = {
            "Lion": "source-face-images/sardar.png",
            "Robot": "source-face-images/robot_face.jpg",
            "Wizard": "source-face-images/wizard_face.jpg"
        }
        
        if character_faces == expected_faces:
            print("✅ Cast extraction test passed!")
            return True
        else:
            print("❌ Cast extraction test failed!")
            print(f"Expected: {expected_faces}")
            print(f"Got: {character_faces}")
            return False
            
    except Exception as e:
        print(f"❌ Cast extraction test failed: {e}")
        return False

def test_storyboard_loading():
    """Test loading storyboard with character faces."""
    print("\n🧪 Testing Storyboard Loading...")
    
    try:
        storyboard_path = "storyboards/example_with_faces.json"
        
        if not Path(storyboard_path).exists():
            print(f"⚠️ Storyboard not found: {storyboard_path}")
            print("💡 Run: python example_face_based_generation.py")
            return False
        
        with open(storyboard_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        cast_list = data.get('cast', [])
        scenes = data.get('scenes', [])
        
        print(f"✅ Loaded storyboard with {len(cast_list)} cast members and {len(scenes)} scenes")
        
        # Test character face extraction
        from src.interfaces.simple_cartoon_generator import extract_character_faces_from_cast
        character_faces = extract_character_faces_from_cast(cast_list)
        
        print(f"✅ Found {len(character_faces)} characters with custom faces")
        
        # Check if scenes reference characters with faces
        scene_characters = set()
        for scene in scenes:
            chars = scene.get('characters', [])
            scene_characters.update(chars)
        
        print(f"✅ Scenes reference {len(scene_characters)} unique characters")
        
        # Check overlap
        characters_with_faces = set(character_faces.keys())
        overlap = scene_characters.intersection(characters_with_faces)
        
        print(f"✅ {len(overlap)} scene characters have custom faces: {list(overlap)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Storyboard loading test failed: {e}")
        return False

def test_face_generation_availability():
    """Test if face-based generation is available."""
    print("\n🧪 Testing Face Generation Availability...")
    
    try:
        from src.core.image_generator import ImageGenerator
        
        # Create a test image generator
        generator = ImageGenerator()
        
        # Test face generation capability
        can_use_face_generation = generator._can_use_face_generation()
        
        if can_use_face_generation:
            print("✅ Face-based generation is available")
        else:
            print("⚠️ Face-based generation not available (missing dependencies)")
            print("💡 Install: pip install mediapipe opencv-python")
        
        return can_use_face_generation
        
    except Exception as e:
        print(f"❌ Face generation availability test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🎭 Face-Based Character Generation Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Cast Extraction", test_cast_extraction),
        ("Storyboard Loading", test_storyboard_loading),
        ("Face Generation Availability", test_face_generation_availability)
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
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Face-based character generation is ready!")
        print("\n💡 Usage:")
        print("python -m src.interfaces.simple_cartoon_generator --prompt 'Your story' --storyboard storyboards/example_with_faces.json")
    else:
        print("⚠️ Some tests failed. Please check the setup.")
        print("\n💡 To set up example files:")
        print("python example_face_based_generation.py")

if __name__ == "__main__":
    main()
