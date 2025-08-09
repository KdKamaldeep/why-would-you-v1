#!/usr/bin/env python3
"""
Test script for Coqui TTS voice generation only
"""

import os
import sys
import logging
from pathlib import Path

# Ensure 'src' is on the import path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.coqui_voice_synthesizer import CoquiVoiceSynthesizer, CoquiVoiceConfig


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_voice_generation():
    """Test Coqui TTS voice generation end-to-end."""
    print("🔊 Testing Coqui TTS Voice Generation")
    print("=" * 50)

    # Prepare output directory
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "test_voice.mp3"

    # Info about model path used by the generator
    expected_model_dir = Path("models/tts/XTTS-v2")
    if expected_model_dir.exists():
        print(f"✅ Found Coqui XTTS model directory: {expected_model_dir}")
    else:
        print(f"⚠️ Coqui XTTS model directory not found at: {expected_model_dir}")
        print("   Run: bash src/utils/download_models.sh")

    # Initialize voice generator (hardcoded to models/tts/XTTS-v2)
    print("🚀 Initializing Coqui Voice Synthesizer...")
    vg = CoquiVoiceSynthesizer(CoquiVoiceConfig(language="en"))

    # Generate a short narration
    sample_text = (
        "Hello! This is a quick test of the Coqui XTTS voice generator. "
        "We are generating a short sample narration to verify audio output."
    )

    print("🎙️  Generating narration...")
    try:
        path = vg.synthesize_voice([sample_text], output_path=str(output_path), speaker=None, voice_clone_audio=None)
        if Path(path).exists():
            size_kb = Path(path).stat().st_size / 1024
            print(f"✅ Audio generated: {path} ({size_kb:.1f} KB)")
        else:
            print("❌ Audio file was not created")
    except Exception as e:
        print(f"❌ Voice generation failed: {e}")


if __name__ == "__main__":
    test_voice_generation()


