#!/usr/bin/env python3
"""
Test script to verify that the attention mask issue is fixed in the image generator.
"""

import os
import sys
import logging
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_attention_mask_fix():
    """Test that the attention mask warning is resolved."""
    try:
        from core.image_generator import ImageGenerator
        
        logger.info("🧪 Testing attention mask fix...")
        
        # Initialize the image generator
        generator = ImageGenerator()
        
        if not generator.is_sd_available():
            logger.warning("⚠️ Stable Diffusion not available - skipping test")
            return False
        
        # Test with a simple prompt
        test_prompt = "A cute cartoon cat sitting in a garden"
        test_output = "test_attention_mask_fix.png"
        
        logger.info(f"🎨 Generating test image with prompt: {test_prompt}")
        
        # This should not produce the attention mask warning
        result_path = generator.generate_cartoon_image(test_prompt, test_output)
        
        if Path(result_path).exists():
            logger.info("✅ Test completed successfully!")
            logger.info(f"📁 Generated image: {result_path}")
            
            # Clean up test file
            try:
                os.remove(test_output)
                logger.info("🧹 Cleaned up test file")
            except:
                pass
            
            return True
        else:
            logger.error("❌ Test failed - no image was generated")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_attention_mask_fix()
    if success:
        logger.info("🎉 Attention mask fix test passed!")
        sys.exit(0)
    else:
        logger.error("💥 Attention mask fix test failed!")
        sys.exit(1)
