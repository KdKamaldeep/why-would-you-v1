#!/usr/bin/env python3
"""
Test script for Coqui TTS voice synthesis.

This script tests:
- Basic voice synthesis
- Voice cloning with speaker audio files
- Voice speed adjustment
- Speaker audio cleaning/normalization
- Different languages

Usage:
    # Basic test
    python scripts/test_voice_synth.py --text "Hello, this is a test"

    # Test with voice file
    python scripts/test_voice_synth.py --text "Hello, this is a test" --voice path/to/speaker.wav

    # Test with voice speed
    python scripts/test_voice_synth.py --text "Hello, this is a test" --voice-speed 0.9

    # Test different language
    python scripts/test_voice_synth.py --text "नमस्ते" --language hi --voice path/to/hindi_speaker.wav
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.coqui_voice_synthesizer import CoquiVoiceSynthesizer, CoquiVoiceConfig, clean_speaker_audio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def test_basic_synthesis(text: str, output_path: str, language: str = "en"):
    """Test basic voice synthesis without voice cloning."""
    logger.info("=" * 60)
    logger.info("Test 1: Basic Voice Synthesis")
    logger.info("=" * 60)
    
    try:
        config = CoquiVoiceConfig(language=language)
        synthesizer = CoquiVoiceSynthesizer(config)
        
        logger.info(f"📝 Text: {text}")
        logger.info(f"🗣️ Language: {language}")
        logger.info(f"📁 Output: {output_path}")
        
        result = synthesizer.synthesize_voice(
            narration_lines=[text],
            output_path=output_path,
            speaker=None,
            voice_clone_audio=None
        )
        
        if os.path.exists(result):
            logger.info(f"✅ Basic synthesis successful: {result}")
            return True
        else:
            logger.error(f"❌ Output file not found: {result}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Basic synthesis failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_voice_cloning(text: str, output_path: str, voice_file: str, language: str = "en"):
    """Test voice cloning with speaker audio file."""
    logger.info("=" * 60)
    logger.info("Test 2: Voice Cloning with Speaker File")
    logger.info("=" * 60)
    
    if not os.path.exists(voice_file):
        logger.error(f"❌ Voice file not found: {voice_file}")
        return False
    
    try:
        config = CoquiVoiceConfig(language=language)
        synthesizer = CoquiVoiceSynthesizer(config)
        
        logger.info(f"📝 Text: {text}")
        logger.info(f"🗣️ Language: {language}")
        logger.info(f"🎵 Voice file: {voice_file}")
        logger.info(f"📁 Output: {output_path}")
        
        # Check if voice file will be cleaned
        logger.info("🧹 Voice file will be automatically cleaned and normalized")
        
        result = synthesizer.synthesize_voice(
            narration_lines=[text],
            output_path=output_path,
            speaker=None,
            voice_clone_audio=voice_file
        )
        
        if os.path.exists(result):
            logger.info(f"✅ Voice cloning successful: {result}")
            return True
        else:
            logger.error(f"❌ Output file not found: {result}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Voice cloning failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_speaker_cleaning(voice_file: str, output_path: str):
    """Test speaker audio cleaning/normalization."""
    logger.info("=" * 60)
    logger.info("Test 3: Speaker Audio Cleaning")
    logger.info("=" * 60)
    
    if not os.path.exists(voice_file):
        logger.error(f"❌ Voice file not found: {voice_file}")
        return False
    
    try:
        logger.info(f"📥 Input: {voice_file}")
        logger.info(f"📁 Output: {output_path}")
        logger.info("🧹 Cleaning and normalizing speaker audio...")
        logger.info("   - Converting to mono (1 channel)")
        logger.info("   - Resampling to 22050 Hz")
        logger.info("   - Setting 16-bit sample format")
        logger.info("   - Applying highpass filter (80Hz)")
        logger.info("   - Applying lowpass filter (12000Hz)")
        logger.info("   - Applying loudness normalization")
        
        result = clean_speaker_audio(voice_file, output_path)
        
        if os.path.exists(result):
            logger.info(f"✅ Speaker audio cleaned: {result}")
            
            # Get file sizes for comparison
            original_size = os.path.getsize(voice_file)
            cleaned_size = os.path.getsize(result)
            logger.info(f"📊 Original size: {original_size:,} bytes")
            logger.info(f"📊 Cleaned size: {cleaned_size:,} bytes")
            
            return True
        else:
            logger.error(f"❌ Cleaned file not found: {result}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Speaker cleaning failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_voice_speed(text: str, output_path: str, voice_file: str, voice_speed: float, language: str = "en"):
    """Test voice synthesis with speed adjustment."""
    logger.info("=" * 60)
    logger.info("Test 4: Voice Synthesis with Speed Adjustment")
    logger.info("=" * 60)
    
    if not os.path.exists(voice_file):
        logger.error(f"❌ Voice file not found: {voice_file}")
        return False
    
    try:
        from scripts.check_sync_bridge import generate_audio_with_coqui
        
        logger.info(f"📝 Text: {text}")
        logger.info(f"🗣️ Language: {language}")
        logger.info(f"🎵 Voice file: {voice_file}")
        logger.info(f"🎚️ Voice speed: {voice_speed}x")
        logger.info(f"📁 Output: {output_path}")
        
        result = generate_audio_with_coqui(
            text=text,
            output_path=output_path,
            language=language,
            voice_clone_audio=voice_file,
            voice_speed=voice_speed
        )
        
        if os.path.exists(result):
            logger.info(f"✅ Voice synthesis with speed successful: {result}")
            return True
        else:
            logger.error(f"❌ Output file not found: {result}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Voice synthesis with speed failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test Coqui TTS voice synthesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic test
  python scripts/test_voice_synth.py --text "Hello, this is a test"

  # Test with voice file
  python scripts/test_voice_synth.py --text "Hello, this is a test" --voice path/to/speaker.wav

  # Test speaker cleaning only
  python scripts/test_voice_synth.py --test cleaning --voice path/to/speaker.wav

  # Test with voice speed
  python scripts/test_voice_synth.py --text "Hello, this is a test" --voice path/to/speaker.wav --voice-speed 0.9

  # Test different language
  python scripts/test_voice_synth.py --text "नमस्ते" --language hi --voice path/to/hindi_speaker.wav
        """
    )
    
    parser.add_argument(
        '--text',
        type=str,
        default="Hello, this is a test of the Coqui TTS voice synthesis system.",
        help='Text to synthesize (default: test message)'
    )
    
    parser.add_argument(
        '--language',
        type=str,
        default='en',
        help='Language code (default: en, use hi for Hindi, es for Spanish, etc.)'
    )
    
    parser.add_argument(
        '--voice',
        type=str,
        default=None,
        help='Path to speaker WAV file for voice cloning (optional)'
    )
    
    parser.add_argument(
        '--voice-speed',
        type=float,
        default=1.0,
        help='Voice speed multiplier (default: 1.0, 0.8 = slower, 1.2 = faster)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/test_voice',
        help='Output directory for test files (default: outputs/test_voice)'
    )
    
    parser.add_argument(
        '--test',
        type=str,
        choices=['all', 'basic', 'cloning', 'cleaning', 'speed'],
        default='all',
        help='Which test to run (default: all)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("🧪 Coqui TTS Voice Synthesis Test")
    logger.info("=" * 60)
    logger.info(f"📝 Text: {args.text}")
    logger.info(f"🗣️ Language: {args.language}")
    logger.info(f"🎵 Voice file: {args.voice or 'None (using default)'}")
    logger.info(f"🎚️ Voice speed: {args.voice_speed}x")
    logger.info(f"📁 Output directory: {output_dir}")
    logger.info("=" * 60)
    
    results = {}
    
    # Test 1: Basic synthesis
    if args.test in ['all', 'basic']:
        output_path = str(output_dir / "test_basic.wav")
        results['basic'] = test_basic_synthesis(
            text=args.text,
            output_path=output_path,
            language=args.language
        )
        logger.info("")
    
    # Test 2: Voice cloning
    if args.test in ['all', 'cloning']:
        if args.voice:
            output_path = str(output_dir / "test_cloning.wav")
            results['cloning'] = test_voice_cloning(
                text=args.text,
                output_path=output_path,
                voice_file=args.voice,
                language=args.language
            )
        else:
            logger.warning("⚠️ Skipping voice cloning test (no --voice file provided)")
            results['cloning'] = None
        logger.info("")
    
    # Test 3: Speaker cleaning
    if args.test in ['all', 'cleaning']:
        if args.voice:
            output_path = str(output_dir / f"{Path(args.voice).stem}_clean.wav")
            results['cleaning'] = test_speaker_cleaning(
                voice_file=args.voice,
                output_path=output_path
            )
        else:
            logger.warning("⚠️ Skipping speaker cleaning test (no --voice file provided)")
            results['cleaning'] = None
        logger.info("")
    
    # Test 4: Voice speed
    if args.test in ['all', 'speed']:
        if args.voice and args.voice_speed != 1.0:
            output_path = str(output_dir / "test_speed.wav")
            results['speed'] = test_voice_speed(
                text=args.text,
                output_path=output_path,
                voice_file=args.voice,
                voice_speed=args.voice_speed,
                language=args.language
            )
        else:
            if not args.voice:
                logger.warning("⚠️ Skipping voice speed test (no --voice file provided)")
            if args.voice_speed == 1.0:
                logger.warning("⚠️ Skipping voice speed test (--voice-speed is 1.0, no adjustment needed)")
            results['speed'] = None
        logger.info("")
    
    # Print summary
    logger.info("=" * 60)
    logger.info("📊 Test Summary")
    logger.info("=" * 60)
    
    for test_name, result in results.items():
        if result is None:
            status = "⏭️ Skipped"
        elif result:
            status = "✅ Passed"
        else:
            status = "❌ Failed"
        logger.info(f"{test_name.capitalize()}: {status}")
    
    logger.info("=" * 60)
    
    # Exit with error if any test failed
    failed_tests = [name for name, result in results.items() if result is False]
    if failed_tests:
        logger.error(f"❌ {len(failed_tests)} test(s) failed: {', '.join(failed_tests)}")
        sys.exit(1)
    else:
        logger.info("✅ All tests passed!")
        sys.exit(0)


if __name__ == '__main__':
    main()
