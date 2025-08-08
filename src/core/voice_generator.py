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
except Exception as e:  # noqa: F841
    TTS = None  # type: ignore

# Optional: mitigate PyTorch 2.6 safe deserialization issues for XTTS
def _allowlist_torch_xtts_config() -> None:
    """Allowlist XTTS config class for torch.load when weights_only=True.

    This avoids failures like:
    WeightsUnpickler error: Unsupported global: TTS.tts.configs.xtts_config.XttsConfig
    """
    try:
        import importlib
        import torch  # type: ignore

        # PyTorch 2.6 introduced safe deserialization; add class to allowed globals
        add_safe_globals = getattr(getattr(torch, "serialization", torch), "add_safe_globals", None)

        # Allowlist config class used by older XTTS checkpoints
        xtts_cfg_module = importlib.import_module("TTS.tts.configs.xtts_config")
        xtts_cfg_cls = getattr(xtts_cfg_module, "XttsConfig", None)
        if xtts_cfg_cls is not None and callable(add_safe_globals):
            add_safe_globals([xtts_cfg_cls])

        # Allowlist audio config used by newer XTTS weights
        xtts_audio_module = importlib.import_module("TTS.tts.models.xtts")
        xtts_audio_cfg = getattr(xtts_audio_module, "XttsAudioConfig", None)
        if xtts_audio_cfg is not None and callable(add_safe_globals):
            add_safe_globals([xtts_audio_cfg])
    except Exception:
        # Best-effort; proceed if not available
        pass


def _patch_torch_load_weights_only_false() -> None:
    """Ensure torch.load defaults to weights_only=False for legacy checkpoints.

    Coqui XTTS checkpoints may require full pickled objects. PyTorch 2.6
    defaults weights_only=True which breaks these loads. This patch makes
    weights_only default to False when not explicitly provided.
    """
    try:
        import torch  # type: ignore
        original_load = torch.load

        def patched_load(*args, **kwargs):  # type: ignore
            kwargs.setdefault("weights_only", False)
            return original_load(*args, **kwargs)

        # Only patch once
        if getattr(torch.load, "__name__", "") != "patched_load":
            torch.load = patched_load  # type: ignore
    except Exception:
        pass


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
        self.default_speaker: str | None = None
        self.default_ref_wav: str | None = None
        self._load_model()

    def _load_model(self) -> None:
        if TTS is None:
            logger.warning("Coqui TTS library not installed. Install with: pip install TTS==0.22.0")
            return

        # Add safe globals (no-op if unavailable)
        _allowlist_torch_xtts_config()
        # Patch torch.load default for legacy checkpoints
        _patch_torch_load_weights_only_false()

        # Try local model load first (provide explicit config path for compatibility)
        local_dir = self.model_dir
        try:
            if local_dir and os.path.isdir(local_dir):
                config_path = os.path.join(local_dir, "config.json")
                if os.path.isfile(config_path):
                    logger.info(f"Loading Coqui XTTS v2 from local directory: {local_dir}")
                    # Many TTS versions accept directory for model_path with config_path provided
                self.tts = TTS(model_path=local_dir, config_path=config_path)
                self._init_default_speaker()
                self._init_default_reference()
                return
        except Exception as e:
            logger.warning(f"Failed to load local Coqui XTTS v2 from '{local_dir}': {e}")

        # Fallback: model hub name (may trigger a download)
        try:
            logger.info("Loading Coqui XTTS v2 by model name (may require internet access)...")
            # Pin model name to avoid resolution issues across versions
            self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
            self._init_default_speaker()
            self._init_default_reference()
        except Exception as e:
            logger.error(f"Failed to initialize Coqui TTS: {e}")
            self.tts = None

    def _init_default_speaker(self) -> None:
        """Pick a default speaker if model is multi-speaker and none is provided.

        Uses the first available speaker exposed by the model; falls back to
        common XTTS speaker ids if list is not available.
        """
        # XTTS expects either a reference wav via speaker_wav or a speaker id from the model
        # API may not expose speakers list; leave None and rely on default model speaker
        self.default_speaker = None

    def _init_default_reference(self) -> None:
        """Pick a default reference WAV bundled with the model snapshot."""
        try:
            candidates = []
            # Common sample locations
            for sub in ("samples", "."):
                glob_path = os.path.join(self.model_dir, sub, "*.wav")
                candidates.extend(glob.glob(glob_path))
            if candidates:
                # Prefer an English sample if present
                en_candidates = [p for p in candidates if any(tag in os.path.basename(p).lower() for tag in ("en", "english"))]
                chosen = en_candidates[0] if en_candidates else candidates[0]
                self.default_ref_wav = chosen
        except Exception:
            self.default_ref_wav = None

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
        if speaker_wav is None and self.default_ref_wav and os.path.isfile(self.default_ref_wav):
            speaker_wav = self.default_ref_wav

        try:
            # XTTS v2 supports speaker_wav for cloning; language is required
            self.tts.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=speaker_wav,
                speaker=None if speaker_wav else self.default_speaker,
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
