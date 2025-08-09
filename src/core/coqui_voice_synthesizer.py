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
import numpy as np
from typing import List, Optional, Dict, Any
import torch
from pydantic import BaseModel

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
    speaker: str = "random"
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
            # For non-English targets, try multilingual XTTS first
            if lang != "en":
                fallback_models.append("tts_models/multilingual/multi-dataset/xtts_v2")
            # Always try the explicitly configured model next
            fallback_models.append(self.config.model_name)
            # Add robust alternates
            fallback_models.extend([
                "tts_models/en/ljspeech/fast_pitch",
                "tts_models/en/vctk/vits",
                # YourTTS is multilingual but quality varies; leave lower priority
                "tts_models/multilingual/multi-dataset/your_tts",
                # Ensure XTTS is attempted even for English if earlier attempts failed
                "tts_models/multilingual/multi-dataset/xtts_v2",
            ])
            
            for model_name in fallback_models:
                try:
                    logger.info(f"Attempting to load TTS model: {model_name}")
                    self.tts = TTS(model_name).to(device)
                    logger.info(f"✅ Successfully loaded TTS model: {model_name}")
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
                speaker_wav_arg = voice_clone_audio if (voice_clone_audio and os.path.exists(voice_clone_audio)) else None
                # Do NOT pass placeholder speakers like 'random' to XTTS
                xtts_speaker = None
                if speaker_wav_arg is None and speaker and str(speaker).lower() not in ("", "random"):
                    xtts_speaker = speaker
                try:
                    self.tts.tts_to_file(
                        text=full_text,
                        file_path=output_path,
                        speaker_wav=speaker_wav_arg,
                        speaker=xtts_speaker,
                        language=self.config.language,
                        progress_bar=self.config.progress_bar,
                    )
                except TypeError:
                    # Older TTS may not accept progress_bar; retry without
                    self.tts.tts_to_file(
                        text=full_text,
                        file_path=output_path,
                        speaker_wav=speaker_wav_arg,
                        speaker=xtts_speaker,
                        language=self.config.language,
                    )
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
                try:
                    self.tts.tts_to_file(
                        text=full_text,
                        file_path=output_path,
                        voice_dir=self.config.voice_dir,
                        speaker=current_speaker,
                        progress_bar=self.config.progress_bar,
                    )
                except TypeError:
                    self.tts.tts_to_file(
                        text=full_text,
                        file_path=output_path,
                        voice_dir=self.config.voice_dir,
                        speaker=current_speaker,
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
                voice_dir=self.config.voice_dir,
                speaker=speaker_name,
                progress_bar=True
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
            gpu=torch.cuda.is_available(),
            speaker="random"
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