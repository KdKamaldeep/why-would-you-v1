#!/usr/bin/env python3
"""
Coqui TTS Voice Synthesizer - Bark Model

This module provides voice synthesis using Coqui TTS with the Bark model.
Bark is a multi-lingual TTS model that can generate conversational speech,
music, and sound effects.

Based on: https://docs.coqui.ai/en/dev/models/bark.html
"""

import os
import logging
import tempfile
import warnings
import numpy as np
from typing import List, Optional, Dict, Any
import torch
from pydantic import BaseModel

# Suppress torchaudio deprecation warnings
warnings.filterwarnings("ignore", message=".*torchaudio.load.*")
warnings.filterwarnings("ignore", message=".*StreamingMediaDecoder.*")
warnings.filterwarnings("ignore", message=".*torchcodec.*")

# Suppress attention mask warnings from transformers (used by TTS)
warnings.filterwarnings("ignore", message=".*attention mask is not set.*")
warnings.filterwarnings("ignore", message=".*pad token is same as eos token.*")
warnings.filterwarnings("ignore", message=".*CLIPFeatureExtractor is deprecated.*")
warnings.filterwarnings("ignore", message=".*Some weights of the model checkpoint were not used.*")

# Suppress GPT2InferenceModel GenerationMixin warnings from transformers v4.50+
warnings.filterwarnings("ignore", message=".*GPT2InferenceModel has generative capabilities.*")
warnings.filterwarnings("ignore", message=".*doesn't directly inherit from GenerationMixin.*")
warnings.filterwarnings("ignore", message=".*PreTrainedModel will NOT inherit from GenerationMixin.*")
warnings.filterwarnings("ignore", message=".*this model will lose the ability to call generate.*")
warnings.filterwarnings("ignore", message=".*trust_remote_code=True.*")
warnings.filterwarnings("ignore", message=".*modify your model class such that it inherits from GenerationMixin.*")

# Suppress other common TTS/transformers warnings
warnings.filterwarnings("ignore", message=".*model_kwargs.*")
warnings.filterwarnings("ignore", message=".*The attention mask and the pad token id were not set.*")
warnings.filterwarnings("ignore", message=".*Using the model-agnostic default.*")

logger = logging.getLogger(__name__)

# Patch for PyTorch 2.6 weights_only issue
def _patch_torch_load():
    """Patch torch.load to handle PyTorch 2.6 weights_only compatibility issue"""
    original_torch_load = torch.load
    
    def patched_torch_load(f, *args, **kwargs):
        # Force weights_only=False for TTS model loading
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return original_torch_load(f, *args, **kwargs)
    
    torch.load = patched_torch_load
    logger.info("Applied PyTorch 2.6 weights_only compatibility patch")

# Apply the patch when module is imported
_patch_torch_load()

class CoquiVoiceConfig(BaseModel):
    """Configuration for Coqui TTS voice synthesis"""
    # Prefer a small, reliable English model by default; will switch to XTTS for multilingual
    model_name: str = "tts_models/en/ljspeech/tacotron2-DDC"
    gpu: bool = True
    voice_dir: str = "tts_voices/"
    speaker: str = "default"
    text_temp: float = 0.7
    waveform_temp: float = 0.7
    progress_bar: bool = True
    language: str = "en"  # Target language (e.g., 'en', 'hi', 'es', ...)

class CoquiVoiceSynthesizer:
    def __init__(self, config: Optional[CoquiVoiceConfig] = None):
        """
        Initialize Coqui TTS voice synthesizer with Bark model
        
        Args:
            config: Configuration for voice synthesis
        """
        self.config = config or CoquiVoiceConfig()
        
        # Create voice directory if it doesn't exist
        os.makedirs(self.config.voice_dir, exist_ok=True)
        
        # Initialize TTS model
        self.tts = None
        self._load_model()
        
        logger.info(f"Coqui TTS initialized with model: {self.config.model_name}")
        logger.info(f"GPU enabled: {self.config.gpu}")
        logger.info(f"Voice directory: {self.config.voice_dir}")
    
    def _load_model(self):
        """Load the Coqui TTS model with fallback options"""
        try:
            from TTS.api import TTS

            device = "cuda" if self.config.gpu and torch.cuda.is_available() else "cpu"
            
            # Build a prioritized list of models based on requested language
            fallback_models = []
            lang = (self.config.language or "en").lower()
            
            # Language-specific model prioritization
            if lang == "hi":  # Hindi
                # For Hindi, prioritize XTTS v2 which has excellent Hindi support
                fallback_models.extend([
                    "coqui/XTTS-v2",
                    "tts_models/multilingual/multi-dataset/xtts_v2",
                    "tts_models/multilingual/multi-dataset/your_tts",  # YourTTS also supports Hindi
                ])
            elif lang != "en":  # Other non-English languages
                # For other languages, try XTTS first
                fallback_models.extend([
                    "coqui/XTTS-v2",
                    "tts_models/multilingual/multi-dataset/xtts_v2",
                    "tts_models/multilingual/multi-dataset/your_tts",
                ])
            
            # Always try the explicitly configured model
            if self.config.model_name not in fallback_models:
                fallback_models.append(self.config.model_name)
            
            # Add robust English fallbacks
            fallback_models.extend([
                "tts_models/en/ljspeech/fast_pitch",
                "tts_models/en/vctk/vits",
                "tts_models/en/ljspeech/tacotron2-DDC",
            ])
            
            # Ensure XTTS is attempted even for English if earlier attempts failed
            if "tts_models/multilingual/multi-dataset/xtts_v2" not in fallback_models:
                fallback_models.append("tts_models/multilingual/multi-dataset/xtts_v2")
            
            logger.info(f"Loading TTS model for language: {lang}")
            logger.info(f"Model priority list: {fallback_models}")
            
            for model_name in fallback_models:
                try:
                    logger.info(f"Attempting to load TTS model: {model_name}")
                    
                    # Use trust_remote_code=True to avoid GenerationMixin warnings
                    self.tts = TTS(model_name, trust_remote_code=True).to(device)
                    
                    # Verify the model supports the target language
                    if hasattr(self.tts, 'languages') and self.tts.languages:
                        available_langs = [str(l).lower() for l in self.tts.languages]
                        if lang not in available_langs and lang != "en":
                            logger.warning(f"Model {model_name} may not support language '{lang}'. Available: {available_langs}")
                    
                    logger.info(f"✅ Successfully loaded TTS model: {model_name}")
                    logger.info(f"Model device: {device}")
                    
                    # Update config to reflect the actually loaded model
                    self.config.model_name = model_name
                    return
                    
                except Exception as e:
                    logger.warning(f"Failed to load model {model_name}: {e}")
                    continue
            
            # If we get here, all models failed
            raise Exception("All TTS models failed to load")

        except ImportError:
            logger.error("❌ Coqui TTS not installed. Install with: pip install TTS")
            raise ImportError("Coqui TTS not available. Install with: pip install TTS")
        except Exception as e:
            logger.error(f"❌ Failed to load any TTS model: {e}")
            raise

    
    def synthesize_voice(self, 
                        narration_lines: List[str], 
                        output_path: str,
                        speaker: Optional[str] = None,
                        voice_clone_audio: Optional[str] = None) -> str:
        """
        Synthesize voice from text using Coqui TTS Bark
        
        Args:
            narration_lines: List of text lines to synthesize
            output_path: Path to save the audio file
            speaker: Speaker ID for voice cloning (optional)
            voice_clone_audio: Path to audio file for voice cloning (optional)
            
        Returns:
            Path to the created audio file
        """
        if self.tts is None:
            logger.error("Coqui TTS model not loaded")
            raise RuntimeError("Coqui TTS model not loaded")
        
        try:
            # Join all lines with spaces
            full_text = " ".join(narration_lines)
            logger.info(f"Synthesizing voice for text: {full_text[:100]}...")
            
            model_name_lower = (getattr(self.config, 'model_name', '') or '').lower()

            # For XTTS, prefer direct reference wav and pass language
            if "xtts" in model_name_lower:
                logger.info("Generating audio with XTTS (multilingual)")
                logger.info(f"Target language: {self.config.language}")
                logger.info(f"Text length: {len(full_text)} characters")
                
                # Validate text encoding for non-English languages
                if self.config.language != "en":
                    try:
                        # Ensure text is properly encoded for the target language
                        if self.config.language == "hi":
                            # For Hindi, ensure Devanagari script is properly handled
                            import unicodedata
                            # Normalize Unicode characters
                            full_text = unicodedata.normalize('NFC', full_text)
                            logger.info(f"Normalized Hindi text: {full_text[:50]}...")
                    except Exception as e:
                        logger.warning(f"Text normalization warning: {e}")
                
                speaker_wav_arg = voice_clone_audio if (voice_clone_audio and os.path.exists(voice_clone_audio)) else None
                # Auto-discover a language-appropriate speaker WAV if none provided
                if speaker_wav_arg is None:
                    auto_wav = self._discover_speaker_wav(self.config.language)
                    if auto_wav:
                        logger.info(f"Using discovered speaker_wav for language '{self.config.language}': {auto_wav}")
                        speaker_wav_arg = auto_wav
                
                # Ensure a valid speaker is passed for XTTS if no reference wav
                requested_speaker = (
                    speaker if (speaker is not None and str(speaker).strip() != "") else self.config.speaker
                )
                xtts_speaker = self._select_xtts_speaker(requested_speaker if speaker_wav_arg is None else None)
                logger.info(f"XTTS selected speaker: {xtts_speaker if speaker_wav_arg is None else 'speaker_wav provided'}")

                def _xtts_call(speaker_value: Optional[str]) -> None:
                    # Avoid passing progress_bar to suppress model_kwargs warnings
                    if speaker_wav_arg is not None:
                        # Reference voice provided: do not pass speaker token
                        logger.info(f"XTTS synthesis with speaker_wav: {speaker_wav_arg}")
                        self.tts.tts_to_file(
                            text=full_text,
                            file_path=output_path,
                            speaker_wav=speaker_wav_arg,
                            language=self.config.language,
                        )
                    else:
                        # No reference: pass an explicit speaker token
                        chosen_speaker = speaker_value or self.config.speaker or "default"
                        logger.info(f"XTTS synthesis with speaker: {chosen_speaker}")
                        self.tts.tts_to_file(
                            text=full_text,
                            file_path=output_path,
                            speaker=chosen_speaker,
                            language=self.config.language,
                        )

                # Try multiple synthesis strategies
                synthesis_success = False
                synthesis_errors = []
                
                # Strategy 1: Try with discovered speaker WAV
                if speaker_wav_arg:
                    try:
                        _xtts_call(None)
                        synthesis_success = True
                        logger.info("✅ XTTS synthesis successful with speaker_wav")
                    except Exception as e:
                        error_msg = str(e)
                        synthesis_errors.append(f"Speaker WAV synthesis failed: {error_msg}")
                        logger.warning(f"XTTS speaker_wav synthesis failed: {error_msg}")
                
                # Strategy 2: Try with selected speaker token
                if not synthesis_success:
                    try:
                        _xtts_call(xtts_speaker)
                        synthesis_success = True
                        logger.info("✅ XTTS synthesis successful with speaker token")
                    except Exception as e:
                        error_msg = str(e)
                        synthesis_errors.append(f"Speaker token synthesis failed: {error_msg}")
                        logger.warning(f"XTTS speaker token synthesis failed: {error_msg}")
                
                # Strategy 3: Try with default speaker
                if not synthesis_success:
                    try:
                        _xtts_call("default")
                        synthesis_success = True
                        logger.info("✅ XTTS synthesis successful with default speaker")
                    except Exception as e:
                        error_msg = str(e)
                        synthesis_errors.append(f"Default speaker synthesis failed: {error_msg}")
                        logger.warning(f"XTTS default speaker synthesis failed: {error_msg}")
                
                if not synthesis_success:
                    logger.error(f"All XTTS synthesis strategies failed: {synthesis_errors}")
                    raise Exception(f"XTTS synthesis failed: {'; '.join(synthesis_errors)}")
            else:
                # Non-XTTS models: use speaker registry in voice_dir
                current_speaker = speaker or self.config.speaker
                # Optional: create a named speaker from provided audio for YourTTS-like models
                if voice_clone_audio and os.path.exists(voice_clone_audio):
                    logger.info(f"Cloning voice (registry) from: {voice_clone_audio}")
                    speaker_name = os.path.splitext(os.path.basename(voice_clone_audio))[0]
                    speaker_dir = os.path.join(self.config.voice_dir, speaker_name)
                    os.makedirs(speaker_dir, exist_ok=True)
                    import shutil
                    speaker_audio_path = os.path.join(speaker_dir, "speaker.wav")
                    shutil.copy2(voice_clone_audio, speaker_audio_path)
                    current_speaker = speaker_name
                logger.info(f"Generating audio with speaker: {current_speaker}")
                # Avoid passing progress_bar to suppress model_kwargs warnings
                self.tts.tts_to_file(
                    text=full_text,
                    file_path=output_path,
                    voice_dir=self.config.voice_dir
                )
            
            if os.path.exists(output_path):
                logger.info(f"✅ Voice synthesized successfully: {output_path}")
                return output_path
            else:
                logger.error("❌ Audio file was not created")
                return self._create_silent_audio(output_path, len(narration_lines) * 3)
                
        except Exception as e:
            logger.error(f"❌ Failed to synthesize voice with Coqui TTS: {e}")
            logger.info("Creating fallback silent audio")
            return self._create_silent_audio(output_path, len(narration_lines) * 3)

    def _discover_speaker_wav(self, language: str) -> Optional[str]:
        """Discover a language-appropriate speaker WAV file on disk.

        Heuristics:
        - Check language-specific env vars (e.g., HINDI_SPEAKER_WAV)
        - Check common folders like 'tts-speaker', 'tts_speaker', 'tts_voices'
        - Prefer filenames containing the language or gender hints when possible
        """
        try:
            lang = (language or "").lower()
            logger.info(f"Discovering speaker WAV for language: {lang}")
            
            # Environment overrides
            env_map = {
                "hi": os.getenv("HINDI_SPEAKER_WAV"),
                "en": os.getenv("ENGLISH_SPEAKER_WAV"),
                "es": os.getenv("SPANISH_SPEAKER_WAV"),
            }
            if lang in env_map and env_map[lang] and os.path.exists(env_map[lang]):
                logger.info(f"Using environment variable speaker WAV: {env_map[lang]}")
                return env_map[lang]

            candidates: List[str] = []
            
            # Language-specific search patterns
            if lang == "hi":  # Hindi
                hindi_patterns = [
                    "male_hindi_speaker.wav",
                    "hindi_speaker.wav", 
                    "hindi_male.wav",
                    "hindi_voice.wav",
                    "speaker_hindi.wav",
                ]
                for pattern in hindi_patterns:
                    for folder in ["tts-speaker", "tts_speaker", "tts_voices", "voices"]:
                        path = os.path.join(folder, pattern)
                        if os.path.exists(path):
                            candidates.append(path)
            
            # Common directories for any language
            common_patterns = [
                f"{lang}_speaker.wav",
                f"speaker_{lang}.wav", 
                f"{lang}_voice.wav",
                "speaker.wav",
                "voice.wav",
            ]
            
            for pattern in common_patterns:
                for folder in ["tts-speaker", "tts_speaker", "tts_voices", "voices"]:
                    path = os.path.join(folder, pattern)
                    if os.path.exists(path):
                        candidates.append(path)

            # Broader search for any wav under tts-speaker-like dirs
            for folder in ["tts-speaker", "tts_speaker", "tts_voices", "voices"]:
                if os.path.isdir(folder):
                    try:
                        for name in os.listdir(folder):
                            if name.lower().endswith(".wav"):
                                full = os.path.join(folder, name)
                                candidates.append(full)
                    except Exception as e:
                        logger.debug(f"Could not list directory {folder}: {e}")

            # Remove duplicates and rank candidates
            candidates = list(set(candidates))
            
            # Rank: prefer names with language code, then 'hindi', then 'male'
            def score(path: str) -> int:
                name = os.path.basename(path).lower()
                s = 0
                if lang in name:
                    s += 3
                if "hindi" in name and lang == "hi":
                    s += 2
                if "male" in name:
                    s += 1
                return s

            candidates = sorted(candidates, key=lambda p: (-score(p), p))
            
            logger.info(f"Found {len(candidates)} candidate speaker WAV files")
            for i, c in enumerate(candidates[:5]):  # Log top 5 candidates
                if os.path.exists(c):
                    logger.info(f"  {i+1}. {c}")
            
            # Return the best candidate
            for c in candidates:
                if os.path.exists(c):
                    logger.info(f"Selected speaker WAV: {c}")
                    return c
                    
        except Exception as e:
            logger.warning(f"Error discovering speaker WAV: {e}")
        
        logger.info("No suitable speaker WAV found")
        return None

    def _get_builtin_speakers(self) -> List[str]:
        """Attempt to retrieve a list of available speakers from the loaded TTS model."""
        try:
            # Many models expose a simple .speakers list
            speakers = getattr(self.tts, "speakers", None)
            if speakers:
                try:
                    spk_list = list(speakers)
                except Exception:
                    # Some implementations expose a dict-like mapping
                    spk_list = list(speakers.keys()) if hasattr(speakers, 'keys') else []
                logger.info(f"XTTS available speakers: {spk_list}")
                return spk_list
            # Some expose a speaker_manager with various fields
            sm = getattr(self.tts, "speaker_manager", None)
            if sm is not None:
                for attr in ("speaker_names", "speaker_ids", "speakers"):
                    val = getattr(sm, attr, None)
                    if val:
                        try:
                            spk_list = list(val)
                        except Exception:
                            spk_list = list(val.keys()) if hasattr(val, 'keys') else []
                        logger.info(f"XTTS available speakers (speaker_manager): {spk_list}")
                        return spk_list
        except Exception:
            pass
        return []

    def _select_xtts_speaker(self, preferred: Optional[str]) -> Optional[str]:
        """Select a valid XTTS speaker string.

        - Uses preferred if provided and not 'random'.
        - Else tries builtin speakers from the model.
        - Else falls back to a known common XTTS speaker token.
        """
        # Honor explicit non-random preference
        if preferred and str(preferred).strip().lower() not in ("", "random"):
            return preferred

        # Try to use a builtin speaker from the model
        builtin = self._get_builtin_speakers()
        if builtin:
            try:
                # Prefer a female English voice if present; else first available
                for name in builtin:
                    name_str = str(name)
                    if "female" in name_str.lower():
                        return name_str
                return str(builtin[0])
            except Exception:
                pass

        # Fallback: many XTTS builds accept 'default' to pick a bundled voice
        return "default"
    
    def clone_voice(self, 
                   audio_file_path: str, 
                   speaker_name: str,
                   test_text: str = "Hello, this is a test of the cloned voice.") -> bool:
        """
        Clone a voice from an audio file
        
        Args:
            audio_file_path: Path to the audio file for voice cloning
            speaker_name: Name for the cloned voice
            test_text: Text to test the cloned voice
            
        Returns:
            True if voice cloning was successful
        """
        try:
            if not os.path.exists(audio_file_path):
                logger.error(f"Audio file not found: {audio_file_path}")
                return False
            
            logger.info(f"Cloning voice from: {audio_file_path}")
            logger.info(f"Speaker name: {speaker_name}")
            
            # Create speaker directory
            speaker_dir = os.path.join(self.config.voice_dir, speaker_name)
            os.makedirs(speaker_dir, exist_ok=True)
            
            # Copy audio file to speaker directory
            import shutil
            speaker_audio_path = os.path.join(speaker_dir, "speaker.wav")
            shutil.copy2(audio_file_path, speaker_audio_path)
            
            # Test the cloned voice
            test_output_path = os.path.join(speaker_dir, "test_output.wav")
            
            self.tts.tts_to_file(
                text=test_text,
                file_path=test_output_path,
                voice_dir=self.config.voice_dir                
            )
            
            if os.path.exists(test_output_path):
                logger.info(f"✅ Voice cloned successfully: {speaker_name}")
                logger.info(f"Test audio saved: {test_output_path}")
                return True
            else:
                logger.error("❌ Voice cloning failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Voice cloning failed: {e}")
            return False
    
    def get_available_speakers(self) -> List[str]:
        """Get list of available speakers"""
        try:
            speakers = []
            if os.path.exists(self.config.voice_dir):
                for item in os.listdir(self.config.voice_dir):
                    item_path = os.path.join(self.config.voice_dir, item)
                    if os.path.isdir(item_path):
                        # Check if speaker files exist
                        speaker_wav = os.path.join(item_path, "speaker.wav")
                        speaker_npz = os.path.join(item_path, "speaker.npz")
                        if os.path.exists(speaker_wav) or os.path.exists(speaker_npz):
                            speakers.append(item)
            
            # Add random speaker option
            speakers.append("random")
            
            return speakers
            
        except Exception as e:
            logger.error(f"Failed to get available speakers: {e}")
            return ["random"]
    
    def _create_silent_audio(self, output_path: str, duration_seconds: int) -> str:
        """Create a silent audio file as fallback"""
        logger.info(f"Creating silent audio file: {output_path} ({duration_seconds}s)")
        import wave
        import struct
        
        # Create a silent WAV file
        sample_rate = 24000  # Bark uses 24kHz
        num_samples = int(sample_rate * duration_seconds)
        
        with wave.open(output_path, 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            
            # Generate silent audio data
            silent_data = struct.pack('<h', 0) * num_samples
            wav_file.writeframes(silent_data)
        
        logger.info("Silent audio file created successfully")
        return output_path
    

    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.tts:
                del self.tts
                self.tts = None
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Coqui TTS cleanup completed")
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")

# Example usage function
def test_coqui_voice():
    """Test the Coqui TTS voice synthesizer"""
    logger.info("Testing Coqui TTS Voice Synthesizer")
    
    try:
        # Initialize synthesizer
        config = CoquiVoiceConfig(
            gpu=torch.cuda.is_available()            
        )
        
        synthesizer = CoquiVoiceSynthesizer(config)
        
        # Test text
        test_lines = [
            "Hello, this is a test of the Coqui TTS Bark model.",
            "It can generate high-quality speech with various voices."
        ]
        
        # Generate audio
        output_path = "test_coqui_output.wav"
        audio_file = synthesizer.synthesize_voice(test_lines, output_path)
        
        if audio_file:
            logger.info(f"✅ Test successful: {audio_file}")
        
        # Get available speakers
        speakers = synthesizer.get_available_speakers()
        logger.info(f"Available speakers: {speakers}")
        
        # Cleanup
        synthesizer.cleanup()
        
    except Exception as e:
        logger.error(f"Test failed: {e}")

if __name__ == "__main__":
    test_coqui_voice() 