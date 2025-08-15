#!/usr/bin/env python3
"""
Test script to verify face detection is working with the Sardar Patel face image.
"""

import cv2
import numpy as np
from PIL import Image
import logging
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.face_image_generator import FaceImageGenerator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_face_detection():
    """Test face detection with the Sardar Patel face image."""
    
    face_image_path = "faces/sardar_patel_face.jpg"
    
    # Check if the face image exists
    if not os.path.exists(face_image_path):
        print(f"❌ Face image not found: {face_image_path}")
        return False
    
    print(f"✅ Face image found: {face_image_path}")
    
    # Initialize the face image generator
    print("🔄 Initializing face image generator...")
    generator = FaceImageGenerator()
    
    # Test face detection
    print("🔄 Testing face detection...")
    face_data = generator.detect_face(face_image_path)
    
    if face_data is None:
        print("❌ No face detected in the image")
        return False
    
    face_region, landmarks = face_data
    print(f"✅ Face detected successfully!")
    print(f"   Face region shape: {face_region.shape}")
    print(f"   Number of landmarks: {len(landmarks)}")
    
    # Test creating a control image
    print("🔄 Testing control image creation...")
    control_image = generator.create_face_control_image(face_region)
    print(f"✅ Control image created successfully!")
    print(f"   Control image shape: {control_image.shape}")
    
    # Test image generation (without actually generating)
    print("🔄 Testing image generation setup...")
    if generator.pipe is None:
        print("❌ Diffusion pipeline not initialized")
        return False
    
    print("✅ Diffusion pipeline initialized successfully!")
    
    # Check if ControlNet is available
    if hasattr(generator, 'controlnet_available'):
        if generator.controlnet_available:
            print("✅ ControlNetPipeline is available")
        else:
            print("⚠️ ControlNetPipeline is not available (this is expected)")
    
    # Check if ControlNet model is loaded
    if generator.controlnet is not None:
        print("✅ ControlNet model loaded")
    else:
        print("⚠️ ControlNet model not loaded (this is expected if ControlNetPipeline is not available)")
    
    print("\n" + "=" * 50)
    print("✅ Face detection test completed successfully!")
    print("The system should be able to use the Sardar Patel face for generation.")
    
    return True

if __name__ == "__main__":
    success = test_face_detection()
    sys.exit(0 if success else 1)
