#!/usr/bin/env python3
"""
Voice Generator Module - Coqui TTS (XTTS v2)
"""

import os
import glob
import logging
from typing import List

from pydub import AudioSegment

logger = logging.getLogger(__name__)

try:
    # Lazy import pattern: will fail gracefully, and we will fall back to silent audio
    from TTS.api import TTS  # type: ignore
except Exception:
    TTS = None  # type: ignore


class VoiceGenerator:
    """Handles audio generation using Coqui TTS XTTS v2.

    If a local model directory is provided (default: models/tts/XTTS-v2),
    it will attempt to load from disk. Otherwise it falls back to the
    canonical model name (requires internet and Hugging Face access):
    "tts_models/multilingual/multi-dataset/xtts_v2".

    The second parameter to generation (voice_id) is interpreted as an optional
    path to a reference speaker WAV for cloning. If not provided or not a valid
    file path, the default speaker is used.
    """

    def __init__(self, language: str = "en"):
        self.language = language or "en"
        # Hardcoded local model directory for Coqui XTTS v2
        self.model_dir = "models/tts/XTTS-v2"
        self.tts = None
        self._load_model()

    def _load_model(self) -> None:
        if TTS is None:
            logger.warning("Coqui TTS library not installed. Install with: pip install TTS")
            return

        # Try local model load first
        local_dir = self.model_dir
        model_path = None
        config_path = None

        try:
            if local_dir and os.path.isdir(local_dir):
                # Common filenames in HF snapshot for XTTS
                pth_candidates = glob.glob(os.path.join(local_dir, "*.pth"))
                if not pth_candidates:
                    pth_candidates = glob.glob(os.path.join(local_dir, "**", "*.pth"), recursive=True)
                json_candidates = glob.glob(os.path.join(local_dir, "config*.json"))
                if not json_candidates:
                    json_candidates = glob.glob(os.path.join(local_dir, "**", "config*.json"), recursive=True)

                if pth_candidates and json_candidates:
                    model_path = pth_candidates[0]
                    config_path = json_candidates[0]

            if model_path and config_path:
                logger.info(f"Loading Coqui XTTS v2 from local: {model_path}")
                self.tts = TTS(model_path=model_path, config_path=config_path)
                return
        except Exception as e:
            logger.warning(f"Failed to load local Coqui XTTS v2 from '{local_dir}': {e}")

        # Fallback: model hub name (may trigger a download)
        try:
            logger.info("Loading Coqui XTTS v2 by model name (may require internet access)...")
            self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
        except Exception as e:
            logger.error(f"Failed to initialize Coqui TTS: {e}")
            self.tts = None

    def generate_narration(self, text: str, voice_id: str, output_path: str) -> str:
        """Generate narration audio using Coqui TTS.

        voice_id: Optional path to a reference speaker WAV for cloning.
        """
        if not text:
            return self._generate_silent_audio(output_path)

        if self.tts is None:
            logger.error("Coqui TTS is not available. Generating silent audio.")
            return self._generate_silent_audio(output_path)

        speaker_wav = voice_id if (voice_id and os.path.isfile(voice_id)) else None

        try:
            # XTTS v2 supports speaker_wav for cloning; language is required
            self.tts.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=speaker_wav,
                language=self.language,
            )
            logger.info(f"Generated narration audio: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error generating audio with Coqui TTS: {e}")
            return self._generate_silent_audio(output_path)

    def generate_narration_from_script(self, script: dict, voice_id: str, output_path: str) -> str:
        """Generate narration from script scenes (concatenated text)."""
        narration_text = " ".join([scene.get('narration', '') for scene in script.get('scenes', [])])
        return self.generate_narration(narration_text, voice_id, output_path)

    def generate_individual_narrations(self, script: dict, voice_id: str, output_dir: str) -> List[str]:
        """Generate individual narration files for each scene."""
        audio_paths: List[str] = []
        for i, scene in enumerate(script.get('scenes', [])):
            output_path = f"{output_dir}/audio_{i+1}.mp3"
            audio_path = self.generate_narration(scene.get('narration', ''), voice_id, output_path)
            audio_paths.append(audio_path)
        return audio_paths

    def _generate_silent_audio(self, output_path: str) -> str:
        """Generate silent audio as fallback."""
        audio = AudioSegment.silent(duration=3000)  # 3 seconds
        audio.export(output_path, format="mp3")
        logger.info(f"Generated silent audio: {output_path}")
        return output_path
