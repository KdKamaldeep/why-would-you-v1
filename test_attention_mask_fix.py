#!/usr/bin/env python3
"""
Test script to verify that the attention mask issue and CLIP warnings are fixed in the image generator.
"""

import os
import sys
import logging
import warnings
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_warning_fixes():
    """Test that both attention mask and CLIP warnings are resolved."""
    try:
        # Capture warnings to check if they're suppressed
        captured_warnings = []
        
        def warning_capture(message, category, filename, lineno, file=None, line=None):
            captured_warnings.append(str(message))
        
        # Install warning capture
        old_showwarning = warnings.showwarning
        warnings.showwarning = warning_capture
        
        try:
            from core.image_generator import ImageGenerator
            
            logger.info("🧪 Testing warning fixes...")
            
            # Initialize the image generator
            generator = ImageGenerator()
            
            if not generator.is_sd_available():
                logger.warning("⚠️ Stable Diffusion not available - skipping test")
                return False
            
            # Test with a simple prompt
            test_prompt = "A cute cartoon cat sitting in a garden"
            test_output = "test_warning_fixes.png"
            
            logger.info(f"🎨 Generating test image with prompt: {test_prompt}")
            
            # This should not produce any warnings
            result_path = generator.generate_cartoon_image(test_prompt, test_output)
            
            # Check for unwanted warnings
            unwanted_warnings = [
                "attention mask is not set",
                "pad token is same as eos token",
                "CLIPFeatureExtractor is deprecated",
                "Some weights of the model checkpoint were not used"
            ]
            
            found_warnings = []
            for warning in captured_warnings:
                for unwanted in unwanted_warnings:
                    if unwanted.lower() in warning.lower():
                        found_warnings.append(warning)
            
            if found_warnings:
                logger.warning(f"⚠️ Found unwanted warnings: {found_warnings}")
            else:
                logger.info("✅ No unwanted warnings detected!")
            
            if Path(result_path).exists():
                logger.info("✅ Test completed successfully!")
                logger.info(f"📁 Generated image: {result_path}")
                
                # Clean up test file
                try:
                    os.remove(test_output)
                    logger.info("🧹 Cleaned up test file")
                except:
                    pass
                
                return len(found_warnings) == 0
            else:
                logger.error("❌ Test failed - no image was generated")
                return False
                
        finally:
            # Restore warning handler
            warnings.showwarning = old_showwarning
            
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_warning_fixes()
    if success:
        logger.info("🎉 All warning fixes test passed!")
        sys.exit(0)
    else:
        logger.error("💥 Warning fixes test failed!")
        sys.exit(1)
