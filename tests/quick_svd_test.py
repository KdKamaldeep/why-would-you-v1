#!/usr/bin/env python3
"""
Quick SVD Animation Test
Simple test to verify SVD animation works with ComfyUI
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.svd_animator import SVDAnimator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def quick_svd_test():
    """Quick test of SVD animation."""
    logger.info("🚀 Quick SVD Animation Test")
    
    # Check if ComfyUI is running
    try:
        import requests
        response = requests.get("http://127.0.0.1:8188/system_stats", timeout=5)
        if response.status_code != 200:
            logger.error("❌ ComfyUI is not running. Please start ComfyUI first.")
            return False
        logger.info("✅ ComfyUI is running")
    except Exception as e:
        logger.error(f"❌ ComfyUI connection failed: {e}")
        return False
    
    # Create test image if it doesn't exist
    test_image = "test_output/test_image.png"
    os.makedirs("test_output", exist_ok=True)
    
    if not os.path.exists(test_image):
        logger.info("📝 Creating a simple test image...")
        try:
            from PIL import Image, ImageDraw
            
            # Create a simple test image
            img = Image.new('RGB', (768, 1024), color='blue')
            draw = ImageDraw.Draw(img)
            draw.ellipse([300, 400, 468, 568], fill='yellow')  # Simple circle
            draw.text((350, 600), "SVD Test", fill='white')
            img.save(test_image)
            logger.info(f"✅ Test image created: {test_image}")
        except Exception as e:
            logger.error(f"❌ Failed to create test image: {e}")
            return False
    
    # Test SVD animation
    try:
        svd_animator = SVDAnimator()
        
        logger.info("🎬 Testing SVD animation with test image...")
        
        frames_dir = svd_animator.animate_image(
            image_path=test_image,
            output_dir="test_output/quick_svd_test",
            num_frames=25,
            motion_bucket_id=127,
            fps_id=4,
            cond_aug=0.02,
            seed=42
        )
        
        if os.path.exists(frames_dir):
            frame_count = len(list(Path(frames_dir).glob("*.png")))
            logger.info(f"✅ SVD animation successful: {frame_count} frames generated")
            
            # Create a quick video
            try:
                import subprocess
                video_path = "test_output/quick_svd_test.mp4"
                cmd = [
                    'ffmpeg', '-y',
                    '-framerate', '15',
                    '-i', os.path.join(frames_dir, 'frame_%04d.png'),
                    '-c:v', 'libx264',
                    '-crf', '20',
                    '-pix_fmt', 'yuv420p',
                    video_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0 and os.path.exists(video_path):
                    logger.info(f"✅ Test video created: {video_path}")
                else:
                    logger.warning("⚠️ Video creation failed, but frames are available")
                    
            except Exception as e:
                logger.warning(f"⚠️ Video creation failed: {e}")
            
            return True
        else:
            logger.error("❌ SVD animation failed: No frames generated")
            return False
            
    except Exception as e:
        logger.error(f"❌ SVD animation error: {e}")
        return False

if __name__ == "__main__":
    success = quick_svd_test()
    if success:
        logger.info("🎉 Quick SVD test completed successfully!")
    else:
        logger.error("❌ Quick SVD test failed!")
    sys.exit(0 if success else 1)
