#!/usr/bin/env python3
"""
Generate per-scene audio from a storyboard using scene-specific voice files.
This skips image/animation steps and focuses only on TTS voice verification.

Usage:
  python scripts/generate_audio_from_storyboard.py --storyboard storyboards/multi-speaker.json --language hi --no-reuse
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.coqui_voice_synthesizer import CoquiVoiceSynthesizer, CoquiVoiceConfig
from utils.narration_converter import NarrationConverter


def main():
    parser = argparse.ArgumentParser(description="Generate audio from storyboard scenes with per-scene voices")
    parser.add_argument("--storyboard", default="storyboards/multi-speaker.json", help="Path to storyboard JSON")
    parser.add_argument("--output", default="output", help="Output directory for audio files")
    parser.add_argument("--language", default="hi", help="Narration language (e.g., hi, en)")
    parser.add_argument("--no-reuse", action="store_true", help="Re-generate audio even if it exists")
    args = parser.parse_args()

    storyboard_path = Path(args.storyboard)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not storyboard_path.exists():
        print(f"❌ Storyboard not found: {storyboard_path}")
        sys.exit(1)

    data = json.loads(storyboard_path.read_text(encoding="utf-8"))
    scenes = data.get("scenes", [])
    if not scenes:
        print("❌ No scenes found in storyboard")
        sys.exit(1)

    # Initialize TTS (prefer XTTS since we pass speaker_wav)
    config = CoquiVoiceConfig(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        language=args.language,
        gpu=True,
    )
    tts = CoquiVoiceSynthesizer(config)
    converter = NarrationConverter()

    print(f"🎵 Generating audio for {len(scenes)} scenes from: {storyboard_path}")

    for idx, scene in enumerate(scenes, start=1):
        narration = scene.get("narration", "").strip()
        voice_name = scene.get("voice")
        voice_file = converter.resolve_voice_file(voice_name) if voice_name else None
        out_path = output_dir / f"storyboard_audio_scene_{idx}.wav"

        print(f"\n— Scene {idx} —")
        print(f"Voice: {voice_name or 'None'}")
        print(f"Resolved voice file: {voice_file or 'None'}")
        print(f"Chars: {len(narration)} | Output: {out_path}")

        if args.no_reuse and out_path.exists():
            try:
                out_path.unlink()
            except Exception:
                pass

        # Synthesize
        generated = tts.synthesize_voice(
            narration_lines=[narration],
            output_path=str(out_path),
            speaker=None,
            voice_clone_audio=voice_file,
        )
        print(f"✅ Generated: {generated}")

    print("\n🎉 Done. Check the output WAV files and the log for per-scene voice usage.")


if __name__ == "__main__":
    main()
