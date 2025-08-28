#!/usr/bin/env python3
"""
Test script for SVD motion animation with ComfyUI
Tests the complete pipeline: SD image generation -> SVD motion animation
"""

import os
import sys
import logging
import tempfile
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.image_generator import ImageGenerator
from core.svd_animator import SVDAnimator
from core.animation_generator import AnimationGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_sd_image_generation():
    """Test Stable Diffusion image generation."""
    logger.info("🎨 Testing Stable Diffusion image generation...")
    
    try:
        # Initialize image generator
        image_generator = ImageGenerator()
        
        # Test prompt
        prompt = "A majestic lion sitting on a rock, realistic style, high quality, detailed"
        
        # Generate image
        logger.info(f"📝 Generating image with prompt: {prompt}")
        image_path = image_generator.generate_cartoon_image(
            prompt=prompt,
            output_path="test_output/sd_test_image.png"
        )
        
        if os.path.exists(image_path):
            logger.info(f"✅ SD image generated successfully: {image_path}")
            return image_path
        else:
            logger.error(f"❌ SD image generation failed: {image_path}")
            return None
            
    except Exception as e:
        logger.error(f"❌ SD image generation error: {e}")
        return None

def test_comfyui_connection():
    """Test ComfyUI connection."""
    logger.info("🔗 Testing ComfyUI connection...")
    
    try:
        import requests
        
        # Test connection to ComfyUI
        response = requests.get("http://127.0.0.1:8188/system_stats", timeout=5)
        if response.status_code == 200:
            logger.info("✅ ComfyUI is running and accessible")
            return True
        else:
            logger.error(f"❌ ComfyUI responded with status: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ ComfyUI connection failed: {e}")
        return False

def test_svd_animation(image_path):
    """Test SVD motion animation."""
    logger.info("🎬 Testing SVD motion animation...")
    
    if not image_path or not os.path.exists(image_path):
        logger.error("❌ No valid image path provided for SVD animation")
        return None
    
    try:
        # Initialize SVD animator
        svd_animator = SVDAnimator()
        
        # Test different motion settings
        test_cases = [
            {
                "name": "Low Motion",
                "motion_bucket_id": 63,
                "fps_id": 3,
                "description": "Gentle, subtle movement"
            },
            {
                "name": "Medium Motion", 
                "motion_bucket_id": 127,
                "fps_id": 4,
                "description": "Normal movement"
            },
            {
                "name": "High Motion",
                "motion_bucket_id": 191,
                "fps_id": 6,
                "description": "Active, dynamic movement"
            }
        ]
        
        results = []
        
        for i, test_case in enumerate(test_cases):
            logger.info(f"🎬 Testing {test_case['name']}: {test_case['description']}")
            
            # Create output directory for this test
            output_dir = f"test_output/svd_test_{i+1}_{test_case['name'].lower().replace(' ', '_')}"
            
            # Run SVD animation
            try:
                frames_dir = svd_animator.animate_image(
                    image_path=image_path,
                    output_dir=output_dir,
                    num_frames=25,  # SVD limit
                    motion_bucket_id=test_case["motion_bucket_id"],
                    fps_id=test_case["fps_id"],
                    cond_aug=0.02,
                    seed=42  # Fixed seed for reproducibility
                )
                
                if os.path.exists(frames_dir):
                    frame_count = len(list(Path(frames_dir).glob("*.png")))
                    logger.info(f"✅ {test_case['name']} completed: {frame_count} frames in {frames_dir}")
                    results.append({
                        "name": test_case["name"],
                        "frames_dir": frames_dir,
                        "frame_count": frame_count,
                        "status": "success"
                    })
                else:
                    logger.error(f"❌ {test_case['name']} failed: No frames generated")
                    results.append({
                        "name": test_case["name"],
                        "status": "failed",
                        "error": "No frames generated"
                    })
                    
            except Exception as e:
                logger.error(f"❌ {test_case['name']} error: {e}")
                results.append({
                    "name": test_case["name"],
                    "status": "failed",
                    "error": str(e)
                })
        
        return results
        
    except Exception as e:
        logger.error(f"❌ SVD animation error: {e}")
        return None

def test_animation_generator_integration(image_path):
    """Test AnimationGenerator integration with SVD."""
    logger.info("🔧 Testing AnimationGenerator SVD integration...")
    
    if not image_path or not os.path.exists(image_path):
        logger.error("❌ No valid image path provided for animation test")
        return None
    
    try:
        # Initialize animation generator with SVD
        animation_generator = AnimationGenerator(animator_type="svd")
        
        # Test with longer frame count (should loop SVD frames)
        output_dir = "test_output/animation_generator_test"
        
        logger.info("🎬 Testing AnimationGenerator with SVD (100 frames, should loop 25 SVD frames)")
        
        frames_dir = animation_generator.animate_image(
            image_path=image_path,
            output_dir=output_dir,
            num_frames=100,  # More than SVD limit
            motion_bucket_id=127,
            fps_id=4,
            cond_aug=0.02,
            seed=42
        )
        
        if os.path.exists(frames_dir):
            frame_count = len(list(Path(frames_dir).glob("*.png")))
            logger.info(f"✅ AnimationGenerator SVD test completed: {frame_count} frames in {frames_dir}")
            return {
                "frames_dir": frames_dir,
                "frame_count": frame_count,
                "status": "success"
            }
        else:
            logger.error("❌ AnimationGenerator SVD test failed: No frames generated")
            return {"status": "failed", "error": "No frames generated"}
            
    except Exception as e:
        logger.error(f"❌ AnimationGenerator SVD test error: {e}")
        return {"status": "failed", "error": str(e)}

def create_test_video(frames_dir, output_video):
    """Create a test video from frames using FFmpeg."""
    logger.info(f"🎥 Creating test video from {frames_dir}...")
    
    try:
        import subprocess
        
        cmd = [
            'ffmpeg', '-y',
            '-framerate', '15',
            '-i', os.path.join(frames_dir, 'frame_%04d.png'),
            '-c:v', 'libx264',
            '-crf', '20',
            '-pix_fmt', 'yuv420p',
            output_video
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(output_video):
            logger.info(f"✅ Test video created: {output_video}")
            return True
        else:
            logger.error(f"❌ Video creation failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Video creation error: {e}")
        return False

def main():
    """Main test function."""
    logger.info("🚀 Starting SVD Animation Test Suite")
    
    # Create test output directory
    os.makedirs("test_output", exist_ok=True)
    
    # Test 1: ComfyUI Connection
    logger.info("\n" + "="*50)
    logger.info("TEST 1: ComfyUI Connection")
    logger.info("="*50)
    
    if not test_comfyui_connection():
        logger.error("❌ ComfyUI is not running. Please start ComfyUI first:")
        logger.info("   git clone https://github.com/comfyanonymous/ComfyUI.git")
        logger.info("   cd ComfyUI")
        logger.info("   pip install -r requirements.txt")
        logger.info("   python main.py --listen 127.0.0.1 --port 8188")
        return False
    
    # Test 2: SD Image Generation
    logger.info("\n" + "="*50)
    logger.info("TEST 2: Stable Diffusion Image Generation")
    logger.info("="*50)
    
    image_path = test_sd_image_generation()
    if not image_path:
        logger.error("❌ SD image generation failed. Cannot proceed with SVD tests.")
        return False
    
    # Test 3: SVD Animation
    logger.info("\n" + "="*50)
    logger.info("TEST 3: SVD Motion Animation")
    logger.info("="*50)
    
    svd_results = test_svd_animation(image_path)
    if not svd_results:
        logger.error("❌ SVD animation tests failed.")
        return False
    
    # Test 4: AnimationGenerator Integration
    logger.info("\n" + "="*50)
    logger.info("TEST 4: AnimationGenerator SVD Integration")
    logger.info("="*50)
    
    integration_result = test_animation_generator_integration(image_path)
    
    # Test 5: Create Test Videos
    logger.info("\n" + "="*50)
    logger.info("TEST 5: Create Test Videos")
    logger.info("="*50)
    
    video_results = []
    for result in svd_results:
        if result.get("status") == "success":
            video_path = f"test_output/{result['name'].lower().replace(' ', '_')}_test.mp4"
            if create_test_video(result["frames_dir"], video_path):
                video_results.append(video_path)
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    
    logger.info(f"✅ ComfyUI Connection: {'SUCCESS' if test_comfyui_connection() else 'FAILED'}")
    logger.info(f"✅ SD Image Generation: {'SUCCESS' if image_path else 'FAILED'}")
    
    successful_svd_tests = sum(1 for r in svd_results if r.get("status") == "success")
    logger.info(f"✅ SVD Animation Tests: {successful_svd_tests}/{len(svd_results)} SUCCESS")
    
    for result in svd_results:
        status = "✅ SUCCESS" if result.get("status") == "success" else "❌ FAILED"
        logger.info(f"   - {result['name']}: {status}")
        if result.get("error"):
            logger.info(f"     Error: {result['error']}")
    
    if integration_result:
        status = "✅ SUCCESS" if integration_result.get("status") == "success" else "❌ FAILED"
        logger.info(f"✅ AnimationGenerator Integration: {status}")
    
    logger.info(f"✅ Test Videos Created: {len(video_results)} videos")
    for video_path in video_results:
        logger.info(f"   - {video_path}")
    
    logger.info("\n🎬 Test completed! Check the test_output/ directory for results.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
