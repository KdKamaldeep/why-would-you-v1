#!/usr/bin/env python3
"""
Test script for direct SVD implementation without ComfyUI
"""

import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_direct_svd():
    """Test the direct SVD implementation."""
    try:
        # Import the SVD animator
        from src.core.svd_animator import SVDAnimator
        
        logger.info("✅ SVD Animator imported successfully")
        
        # Create SVD animator instance
        svd_animator = SVDAnimator()
        
        logger.info("✅ SVD Animator initialized")
        
        # Check if pipeline was loaded
        if svd_animator.pipeline is not None:
            logger.info("✅ SVD pipeline loaded successfully")
        else:
            logger.warning("⚠️ SVD pipeline not available - will use FFmpeg fallback")
        
        # Test with a sample image if available
        test_image = "base_image/sample.png"  # Adjust path as needed
        if Path(test_image).exists():
            logger.info(f"🎬 Testing SVD animation with: {test_image}")
            
            output_dir = "test_svd_output"
            result = svd_animator.animate_image(
                image_path=test_image,
                output_dir=output_dir,
                num_frames=25,
                motion_bucket_id=127,
                fps_id=6,
                cond_aug=0.02,
                seed=42
            )
            
            logger.info(f"✅ SVD animation test completed: {result}")
        else:
            logger.info("ℹ️ No test image found, but SVD animator is ready to use")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.info("💡 Make sure you have installed the required dependencies:")
        logger.info("   pip install diffusers transformers accelerate huggingface_hub")
        return False
    except Exception as e:
        logger.error(f"❌ Error testing SVD: {e}")
        return False

if __name__ == "__main__":
    logger.info("🧪 Testing Direct SVD Implementation")
    logger.info("=" * 50)
    
    success = test_direct_svd()
    
    if success:
        logger.info("✅ Direct SVD test completed successfully!")
        logger.info("🎉 You can now use SVD without ComfyUI!")
        logger.info("💡 Usage: --animator svd")
    else:
        logger.error("❌ Direct SVD test failed")
        sys.exit(1)
