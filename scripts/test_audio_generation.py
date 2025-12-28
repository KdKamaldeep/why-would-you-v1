#!/usr/bin/env python3
"""
Test script for audio and music generation using MusicGen and AudioGen.
Tests the generators independently before integration into the main pipeline.
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.music_generator import MusicGenerator
from src.core.sfx_generator import SFXGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_music_generation():
    """Test MusicGen music generation."""
    logger.info("=" * 60)
    logger.info("🎵 Testing Music Generation (MusicGen)")
    logger.info("=" * 60)
    
    # Create output directory
    output_dir = Path("test_output/audio_generation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize generator
        logger.info("\n📦 Initializing MusicGenerator...")
        music_gen = MusicGenerator(model_size="medium")  # Use "small" for faster testing
        
        # Test 1: Simple music generation (short duration)
        logger.info("\n🧪 Test 1: Generating short music (10 seconds)")
        output_path = output_dir / "test_music_short.wav"
        prompt = "soft neutral ambient bed"
        
        generated_path = music_gen.generate_music(
            prompt=prompt,
            duration=10.0,
            output_path=str(output_path),
            use_cache=False
        )
        
        logger.info(f"✅ Test 1 passed: {generated_path}")
        
        # Test 2: Longer music generation
        logger.info("\n🧪 Test 2: Generating longer music (30 seconds)")
        output_path2 = output_dir / "test_music_long.wav"
        prompt2 = "calm peaceful background music, instrumental, no vocals"
        
        generated_path2 = music_gen.generate_music(
            prompt=prompt2,
            duration=30.0,
            output_path=str(output_path2),
            use_cache=False
        )
        
        logger.info(f"✅ Test 2 passed: {generated_path2}")
        
        # Test 3: Very long music (chunked generation)
        logger.info("\n🧪 Test 3: Generating very long music (90 seconds - chunked)")
        output_path3 = output_dir / "test_music_very_long.wav"
        
        generated_path3 = music_gen.generate_long_music(
            prompt=prompt2,
            duration=90.0,
            output_path=str(output_path3),
            chunk_duration=30.0
        )
        
        logger.info(f"✅ Test 3 passed: {generated_path3}")
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ All music generation tests passed!")
        logger.info(f"📁 Output files saved in: {output_dir}")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Music generation test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_sfx_generation():
    """Test AudioGen SFX generation."""
    logger.info("\n" + "=" * 60)
    logger.info("🔊 Testing SFX Generation (AudioGen)")
    logger.info("=" * 60)
    
    # Create output directory
    output_dir = Path("test_output/audio_generation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize generator
        logger.info("\n📦 Initializing SFXGenerator...")
        sfx_gen = SFXGenerator(model_size="medium")  # Use "small" for faster testing
        
        # Test cases from the storyboard
        test_cases = [
            ("room_tone", 6.0, "room_tone.wav"),
            ("low_electrical_ambience", 12.0, "low_electrical_ambience.wav"),
            ("subtle_creak", 14.0, "subtle_creak.wav"),
            ("low_frequency_hum", 16.0, "low_frequency_hum.wav"),
            ("light_tactile_noise", 14.0, "light_tactile_noise.wav"),
            ("very_soft_buzz", 16.0, "very_soft_buzz.wav"),
            ("calm_resolution_tone", 12.0, "calm_resolution_tone.wav"),
        ]
        
        logger.info(f"\n🧪 Generating {len(test_cases)} SFX samples...")
        
        for i, (prompt, duration, filename) in enumerate(test_cases, 1):
            logger.info(f"\n   Test {i}/{len(test_cases)}: {prompt} ({duration}s)")
            output_path = output_dir / filename
            
            generated_path = sfx_gen.generate_sfx(
                prompt=prompt,
                duration=duration,
                output_path=str(output_path),
                use_cache=False
            )
            
            logger.info(f"   ✅ Generated: {generated_path}")
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ All SFX generation tests passed!")
        logger.info(f"📁 Output files saved in: {output_dir}")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ SFX generation test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_integration():
    """Test both generators together (simulate real pipeline)."""
    logger.info("\n" + "=" * 60)
    logger.info("🔗 Testing Integration (Music + SFX)")
    logger.info("=" * 60)
    
    output_dir = Path("test_output/audio_generation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Generate music
        logger.info("\n🎵 Generating background music...")
        music_gen = MusicGenerator(model_size="medium")
        music_path = music_gen.generate_music(
            prompt="soft neutral ambient bed",
            duration=30.0,
            output_path=str(output_dir / "integration_music.wav"),
            use_cache=False
        )
        logger.info(f"✅ Music: {music_path}")
        
        # Generate SFX
        logger.info("\n🔊 Generating SFX...")
        sfx_gen = SFXGenerator(model_size="medium")
        sfx_path = sfx_gen.generate_sfx(
            prompt="room_tone",
            duration=30.0,
            output_path=str(output_dir / "integration_sfx.wav"),
            use_cache=False
        )
        logger.info(f"✅ SFX: {sfx_path}")
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ Integration test passed!")
        logger.info(f"📁 Output files:")
        logger.info(f"   Music: {music_path}")
        logger.info(f"   SFX: {sfx_path}")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Run all tests."""
    logger.info("🚀 Starting Audio Generation Tests")
    logger.info("=" * 60)
    
    results = []
    
    # Test music generation
    try:
        results.append(("Music Generation", test_music_generation()))
    except Exception as e:
        logger.error(f"❌ Music generation test crashed: {e}")
        results.append(("Music Generation", False))
    
    # Test SFX generation
    try:
        results.append(("SFX Generation", test_sfx_generation()))
    except Exception as e:
        logger.error(f"❌ SFX generation test crashed: {e}")
        results.append(("SFX Generation", False))
    
    # Test integration
    try:
        results.append(("Integration", test_integration()))
    except Exception as e:
        logger.error(f"❌ Integration test crashed: {e}")
        results.append(("Integration", False))
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 Test Summary")
    logger.info("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"   {test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    logger.info("=" * 60)
    if all_passed:
        logger.info("🎉 All tests passed!")
    else:
        logger.error("❌ Some tests failed. Check logs above for details.")
    logger.info("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())

