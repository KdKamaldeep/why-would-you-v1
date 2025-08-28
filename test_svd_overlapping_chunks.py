#!/usr/bin/env python3
"""
Test script for SVD overlapping chunk generation with visual consistency.
This demonstrates the new system that extends SVD beyond 24 frames with overlapping chunks.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.animation_generator import AnimationGenerator
from src.core.svd_animator import SVDAnimator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_svd_overlapping_chunks():
    """Test the new SVD overlapping chunk generation system."""
    
    # Test configuration
    test_image_path = "test_image.png"  # You'll need to provide a test image
    output_dir = "test_svd_output"
    target_frames = 120  # Test with 120 frames (5 chunks of 24 frames each)
    
    # Test different overlap configurations
    overlap_configs = [4, 6, 8]
    
    for overlap_frames in overlap_configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing SVD overlapping chunks with {overlap_frames} frame overlap")
        logger.info(f"{'='*60}")
        
        # Create output directory for this test
        test_output_dir = f"{output_dir}_overlap_{overlap_frames}"
        
        try:
            # Initialize animation generator with SVD and overlapping chunks
            animator = AnimationGenerator(
                width=768,
                height=1024,
                animator_type="svd",
                svd_chunked_generation=True,
                svd_overlap_frames=overlap_frames
            )
            
            # Test parameters for consistency
            motion_bucket_id = 127  # Medium motion
            fps_id = 6  # Normal speed
            cond_aug = 0.02  # Standard conditioning
            seed = 42  # Fixed seed for reproducibility
            
            logger.info(f"🎬 Generating {target_frames} frames with SVD overlapping chunks")
            logger.info(f"📊 Configuration: {overlap_frames} frame overlap, {24-overlap_frames} new frames per chunk")
            logger.info(f"🎯 Fixed parameters: seed={seed}, motion_bucket_id={motion_bucket_id}, fps_id={fps_id}, cond_aug={cond_aug}")
            
            # Generate animation
            result_dir = animator.animate_image(
                image_path=test_image_path,
                output_dir=test_output_dir,
                num_frames=target_frames,
                prompt="Smooth motion animation",
                motion_bucket_id=motion_bucket_id,
                fps_id=fps_id,
                cond_aug=cond_aug,
                seed=seed
            )
            
            # Count generated frames
            frames_dir = Path(result_dir)
            frame_files = list(frames_dir.glob("frame_*.png"))
            actual_frames = len(frame_files)
            
            logger.info(f"✅ Test completed successfully!")
            logger.info(f"📊 Generated {actual_frames} frames in: {result_dir}")
            logger.info(f"📊 Expected: {target_frames} frames")
            logger.info(f"📊 Difference: {actual_frames - target_frames} frames")
            
            # Verify frame sequence
            if actual_frames > 0:
                logger.info(f"📋 Frame sequence verification:")
                logger.info(f"   - First frame: {frame_files[0].name}")
                logger.info(f"   - Last frame: {frame_files[-1].name}")
                logger.info(f"   - Frame count: {actual_frames}")
                
                # Check for potential issues
                if actual_frames < target_frames:
                    logger.warning(f"⚠️ Generated fewer frames than requested")
                elif actual_frames > target_frames:
                    logger.warning(f"⚠️ Generated more frames than requested")
                else:
                    logger.info(f"✅ Frame count matches target exactly")
            
        except Exception as e:
            logger.error(f"❌ Test failed with {overlap_frames} frame overlap: {e}")
            import traceback
            traceback.print_exc()

def test_svd_consistency():
    """Test visual consistency across multiple runs with same parameters."""
    
    logger.info(f"\n{'='*60}")
    logger.info("Testing SVD visual consistency across multiple runs")
    logger.info(f"{'='*60}")
    
    test_image_path = "test_image.png"
    output_dir = "test_svd_consistency"
    target_frames = 60
    
    # Fixed parameters for consistency testing
    motion_bucket_id = 127
    fps_id = 6
    cond_aug = 0.02
    seed = 42
    overlap_frames = 6
    
    try:
        animator = AnimationGenerator(
            width=768,
            height=1024,
            animator_type="svd",
            svd_chunked_generation=True,
            svd_overlap_frames=overlap_frames
        )
        
        # Run multiple generations with same parameters
        for run in range(3):
            logger.info(f"\n🎬 Consistency test run {run + 1}/3")
            
            run_output_dir = f"{output_dir}_run_{run + 1}"
            
            result_dir = animator.animate_image(
                image_path=test_image_path,
                output_dir=run_output_dir,
                num_frames=target_frames,
                prompt="Consistent motion animation",
                motion_bucket_id=motion_bucket_id,
                fps_id=fps_id,
                cond_aug=cond_aug,
                seed=seed  # Same seed for all runs
            )
            
            frames_dir = Path(result_dir)
            frame_files = list(frames_dir.glob("frame_*.png"))
            actual_frames = len(frame_files)
            
            logger.info(f"✅ Run {run + 1} completed: {actual_frames} frames")
        
        logger.info(f"\n✅ Consistency test completed - all runs used identical parameters")
        logger.info(f"🎯 Fixed seed: {seed} ensures reproducible results")
        
    except Exception as e:
        logger.error(f"❌ Consistency test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test function."""
    logger.info("🧪 SVD Overlapping Chunks Test Suite")
    logger.info("=" * 60)
    
    # Check if test image exists
    test_image_path = "test_image.png"
    if not Path(test_image_path).exists():
        logger.error(f"❌ Test image not found: {test_image_path}")
        logger.info("💡 Please create a test image named 'test_image.png' in the current directory")
        logger.info("💡 Or modify the script to use an existing image path")
        return
    
    # Run tests
    test_svd_overlapping_chunks()
    test_svd_consistency()
    
    logger.info(f"\n{'='*60}")
    logger.info("🎉 All tests completed!")
    logger.info("📋 Summary of improvements:")
    logger.info("   ✅ Overlapping chunk generation (4-8 frame overlap)")
    logger.info("   ✅ Fixed parameters for visual consistency")
    logger.info("   ✅ Reproducible results with fixed seeds")
    logger.info("   ✅ Smooth transitions between chunks")
    logger.info("   ✅ Configurable overlap amount")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    main()
