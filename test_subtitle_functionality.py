#!/usr/bin/env python3
"""
Test script to demonstrate subtitle functionality in the image generator.
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

def test_subtitle_functionality():
    """Test that subtitle functionality works correctly."""
    try:
        from core.image_generator import ImageGenerator
        
        logger.info("🧪 Testing subtitle functionality...")
        
        # Initialize the image generator
        generator = ImageGenerator()
        
        # Test prompts and subtitles
        test_cases = [
            {
                "prompt": "A cute cartoon cat sitting in a garden with flowers",
                "subtitle": "The cat discovers a magical garden"
            },
            {
                "prompt": "A brave cartoon dragon learning to bake cookies",
                "subtitle": "Even dragons can be sweet!"
            },
            {
                "prompt": "Two cartoon robots having a dance party",
                "subtitle": "Robots know how to have fun too!"
            }
        ]
        
        # Test individual image generation with subtitles
        for i, test_case in enumerate(test_cases):
            logger.info(f"🎨 Testing subtitle case {i+1}: {test_case['subtitle']}")
            
            output_path = f"test_subtitle_{i+1}.png"
            result_path = generator.generate_cartoon_image(
                test_case["prompt"], 
                output_path, 
                test_case["subtitle"]
            )
            
            if Path(result_path).exists():
                logger.info(f"✅ Generated image with subtitle: {result_path}")
            else:
                logger.error(f"❌ Failed to generate image: {result_path}")
        
        # Test multiple images with subtitles
        logger.info("🎬 Testing multiple images with subtitles...")
        
        prompts = [case["prompt"] for case in test_cases]
        subtitles = [case["subtitle"] for case in test_cases]
        
        result_paths = generator.generate_multiple_images(
            prompts, 
            "test_subtitles_output", 
            subtitles
        )
        
        logger.info(f"✅ Generated {len(result_paths)} images with subtitles")
        
        # Clean up test files
        logger.info("🧹 Cleaning up test files...")
        for i in range(len(test_cases)):
            try:
                os.remove(f"test_subtitle_{i+1}.png")
            except:
                pass
        
        # Clean up directory
        try:
            import shutil
            shutil.rmtree("test_subtitles_output")
        except:
            pass
        
        logger.info("🎉 Subtitle functionality test completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_subtitle_functionality()
    if success:
        logger.info("🎉 Subtitle functionality test passed!")
        sys.exit(0)
    else:
        logger.error("💥 Subtitle functionality test failed!")
        sys.exit(1)
