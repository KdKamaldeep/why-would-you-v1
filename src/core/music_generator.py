#!/usr/bin/env python3
"""
Music Generator using Meta's MusicGen model
Generates background music from text prompts
"""

import os
import logging
import hashlib
import torch
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Global singleton instance
_music_gen_instance = None
_music_gen_model_cache = {}


def get_music_gen_instance(model_size: str = "medium", device: Optional[str] = None, force_reload: bool = False):
    """
    Get or initialize the global MusicGen instance (singleton pattern).
    
    Args:
        model_size: Model size - "small" (300M), "medium" (1.5B), or "large" (3.3B)
        device: Device to use ("cuda" or "cpu"). If None, auto-detect.
        force_reload: Force reload of the model even if already loaded.
        
    Returns:
        MusicGen model instance
    """
    global _music_gen_instance, _music_gen_model_cache
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    cache_key = f"musicgen_{model_size}_{device}"
    
    if cache_key in _music_gen_model_cache and not force_reload:
        logger.info(f"♻️ Reusing cached MusicGen model: {cache_key}")
        _music_gen_instance = _music_gen_model_cache[cache_key]
        return _music_gen_instance
    
    try:
        from audiocraft.models import MusicGen
        
        logger.info(f"📦 Loading MusicGen {model_size} model...")
        logger.info(f"💻 Device: {device}")
        
        # Map model size to actual model name
        model_map = {
            "small": ("facebook/musicgen-small", "models/audiocraft/musicgen-small"),
            "medium": ("facebook/musicgen-medium", "models/audiocraft/musicgen-medium"),
            "large": ("facebook/musicgen-large", "models/audiocraft/musicgen-large")
        }
        
        if model_size not in model_map:
            logger.warning(f"⚠️ Unknown model size '{model_size}', using 'medium'")
            model_size = "medium"
        
        hf_model_name, local_model_path = model_map[model_size]
        
        # Check if local model exists, otherwise use HuggingFace model name
        local_path = Path(local_model_path)
        if local_path.exists() and any(local_path.iterdir()):
            logger.info(f"📁 Found local MusicGen model directory: {local_path}")
            # AudioCraft's get_pretrained can load from local path
            # We need to check if it's a valid model directory
            has_model_files = (
                any(local_path.glob("*.safetensors")) or
                any(local_path.glob("*.bin")) or
                (local_path / "config.json").exists() or
                (local_path / "model_index.json").exists()
            )
            if has_model_files:
                try:
                    logger.info(f"📥 Attempting to load from local path: {local_path}")
                    model = MusicGen.get_pretrained(str(local_path), device=device)
                    logger.info(f"✅ Loaded MusicGen model from local path: {local_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load from local path {local_path}: {e}")
                    logger.info(f"📥 Falling back to HuggingFace (will use cache if available): {hf_model_name}")
                    model = MusicGen.get_pretrained(hf_model_name, device=device)
            else:
                logger.warning(f"⚠️ Local path {local_path} exists but doesn't contain valid model files")
                logger.info(f"📥 Using HuggingFace model (will use cache if available): {hf_model_name}")
                model = MusicGen.get_pretrained(hf_model_name, device=device)
        else:
            logger.info(f"📥 Local model not found at {local_path}, using HuggingFace (will use cache if available): {hf_model_name}")
            model = MusicGen.get_pretrained(hf_model_name, device=device)
        
        # Cache the instance
        _music_gen_instance = model
        _music_gen_model_cache[cache_key] = model
        
        logger.info(f"✅ MusicGen {model_size} model loaded successfully")
        return model
        
    except ImportError:
        logger.error("❌ audiocraft not installed. Install it with: pip install audiocraft")
        raise
    except Exception as e:
        logger.error(f"❌ Failed to load MusicGen model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


class MusicGenerator:
    """
    Generates background music using Meta's MusicGen model.
    """
    
    def __init__(self, model_size: str = "medium", device: Optional[str] = None):
        """
        Initialize MusicGenerator.
        
        Args:
            model_size: Model size - "small" (300M), "medium" (1.5B), or "large" (3.3B)
            device: Device to use ("cuda" or "cpu"). If None, auto-detect.
        """
        self.model_size = model_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        
        logger.info(f"🎵 Initializing MusicGenerator (model: {model_size}, device: {self.device})")
    
    def _get_model(self):
        """Get or load the MusicGen model (singleton pattern)."""
        if self.model is None:
            self.model = get_music_gen_instance(model_size=self.model_size, device=self.device)
        return self.model
    
    def _get_cache_key(self, prompt: str, duration: float) -> str:
        """Generate cache key for generated music."""
        key_string = f"{prompt}_{duration:.1f}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def generate_music(
        self,
        prompt: str,
        duration: float,
        output_path: str,
        use_cache: bool = True,
        continuation_sec: Optional[float] = None
    ) -> str:
        """
        Generate music from text prompt.
        
        Args:
            prompt: Text description of the music (e.g., "soft neutral ambient bed")
            duration: Duration in seconds (must be <= 30 seconds for most models, will loop if longer)
            output_path: Path to save generated audio file (WAV format)
            use_cache: If True, check for cached version first
            continuation_sec: Optional continuation duration for seamless looping
            
        Returns:
            Path to generated audio file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check cache if enabled
        if use_cache and output_path.exists():
            logger.info(f"♻️ Using cached music: {output_path}")
            return str(output_path)
        
        try:
            model = self._get_model()
            
            logger.info(f"🎵 Generating music...")
            logger.info(f"   Prompt: {prompt}")
            logger.info(f"   Duration: {duration:.1f}s")
            
            # Set generation parameters
            model.set_generation_params(
                duration=duration,
                temperature=1.0,
                top_k=250,
                top_p=0.0,
                cfg_coef=3.0,
                extend_stride=continuation_sec if continuation_sec else 18
            )
            
            # Generate audio
            with torch.no_grad():
                wav = model.generate([prompt], progress=True)
            
            # Convert tensor to numpy and save
            import soundfile as sf
            import numpy as np
            
            # wav is a tensor with shape [batch, channels, samples] or list of tensors
            # MusicGen returns tensor of shape [1, 1, sample_rate * duration] for mono
            if isinstance(wav, list):
                wav_tensor = wav[0]
            else:
                wav_tensor = wav
            
            # Convert to numpy
            if isinstance(wav_tensor, torch.Tensor):
                wav_np = wav_tensor.cpu().numpy()
            else:
                wav_np = np.array(wav_tensor)
            
            # Handle shape: [batch, channels, samples] -> [samples]
            # Or [batch, samples] -> [samples]
            if wav_np.ndim == 3:
                # Shape: [batch, channels, samples] -> [samples]
                wav_np = wav_np[0, 0]  # Take first batch, first channel
            elif wav_np.ndim == 2:
                # Shape: [batch, samples] or [channels, samples]
                if wav_np.shape[0] == 1:
                    wav_np = wav_np[0]  # [batch, samples] -> [samples]
                else:
                    wav_np = wav_np[0]  # [channels, samples] -> take first channel
            
            # Ensure it's 1D
            wav_np = wav_np.squeeze()
            
            # Get sample rate from model (MusicGen outputs at 32kHz)
            sample_rate = getattr(model, 'sample_rate', 32000)
            
            # Save as WAV (soundfile expects 1D array for mono, or [samples, channels] for stereo)
            sf.write(str(output_path), wav_np, sample_rate)
            
            logger.info(f"✅ Music generated: {output_path}")
            logger.info(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"❌ Error generating music: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def generate_long_music(
        self,
        prompt: str,
        duration: float,
        output_path: str,
        chunk_duration: float = 30.0,
        overlap: float = 2.0
    ) -> str:
        """
        Generate music longer than model's max duration by chunking.
        
        Args:
            prompt: Text description of the music
            duration: Total duration in seconds
            output_path: Path to save generated audio file
            chunk_duration: Duration of each chunk (model limit, typically 30s)
            overlap: Overlap between chunks in seconds (for seamless transitions)
            
        Returns:
            Path to generated audio file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        import soundfile as sf
        import numpy as np
        
        model = self._get_model()
        sample_rate = getattr(model, 'sample_rate', 32000)
        chunks = []
        
        num_chunks = int(np.ceil(duration / chunk_duration))
        logger.info(f"🎵 Generating long music ({duration:.1f}s) in {num_chunks} chunks...")
        
        for i in range(num_chunks):
            chunk_start = i * (chunk_duration - overlap)
            chunk_dur = min(chunk_duration, duration - chunk_start)
            
            if chunk_dur <= 0:
                break
            
            logger.info(f"   Generating chunk {i+1}/{num_chunks} ({chunk_dur:.1f}s)...")
            
            chunk_path = output_path.parent / f"{output_path.stem}_chunk_{i}.wav"
            self.generate_music(prompt, chunk_dur, str(chunk_path), use_cache=False)
            
            # Load chunk
            chunk_wav, sr = sf.read(str(chunk_path))
            if sr != sample_rate:
                import librosa
                chunk_wav = librosa.resample(chunk_wav, orig_sr=sr, target_sr=sample_rate)
            
            # Convert to mono if stereo
            if chunk_wav.ndim > 1:
                chunk_wav = np.mean(chunk_wav, axis=1)
            
            chunks.append(chunk_wav)
            
            # Clean up chunk file
            chunk_path.unlink()
        
        # Concatenate chunks
        full_wav = np.concatenate(chunks)
        
        # Trim to exact duration
        max_samples = int(duration * sample_rate)
        if len(full_wav) > max_samples:
            full_wav = full_wav[:max_samples]
        
        # Save final audio
        sf.write(str(output_path), full_wav, sample_rate)
        
        logger.info(f"✅ Long music generated: {output_path} ({duration:.1f}s)")
        return str(output_path)

