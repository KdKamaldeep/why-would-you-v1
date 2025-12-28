#!/usr/bin/env python3
"""
SFX Generator using Meta's AudioGen model
Generates sound effects from text prompts
"""

import os
import logging
import hashlib
import torch
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Global singleton instance
_audiogen_instance = None
_audiogen_model_cache = {}


def get_audiogen_instance(model_size: str = "medium", device: Optional[str] = None, force_reload: bool = False):
    """
    Get or initialize the global AudioGen instance (singleton pattern).
    
    Args:
        model_size: Model size - "small" (300M) or "medium" (1.5B)
        device: Device to use ("cuda" or "cpu"). If None, auto-detect.
        force_reload: Force reload of the model even if already loaded.
        
    Returns:
        AudioGen model instance
    """
    global _audiogen_instance, _audiogen_model_cache
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    cache_key = f"audiogen_{model_size}_{device}"
    
    if cache_key in _audiogen_model_cache and not force_reload:
        logger.info(f"♻️ Reusing cached AudioGen model: {cache_key}")
        _audiogen_instance = _audiogen_model_cache[cache_key]
        return _audiogen_instance
    
    try:
        from audiocraft.models import AudioGen
        
        logger.info(f"📦 Loading AudioGen {model_size} model...")
        logger.info(f"💻 Device: {device}")
        
        # Map model size to actual model name
        model_map = {
            "small": ("facebook/audiogen-small", "models/audiocraft/audiogen-small"),
            "medium": ("facebook/audiogen-medium", "models/audiocraft/audiogen-medium")
        }
        
        if model_size not in model_map:
            logger.warning(f"⚠️ Unknown model size '{model_size}', using 'medium'")
            model_size = "medium"
        
        hf_model_name, local_model_path = model_map[model_size]
        
        # Check if local model exists, otherwise use HuggingFace model name
        # AudioCraft uses HuggingFace Hub cache, but we can check for local models first
        local_path = Path(local_model_path)
        if local_path.exists() and any(local_path.iterdir()):
            logger.info(f"📁 Found local AudioGen model directory: {local_path}")
            # Check if it contains valid model files
            has_model_files = (
                any(local_path.glob("*.safetensors")) or
                any(local_path.glob("*.bin")) or
                (local_path / "config.json").exists() or
                (local_path / "model_index.json").exists()
            )
            if has_model_files:
                # AudioCraft's get_pretrained should work with local paths that are valid HuggingFace repos
                # Try using the local path directly
                try:
                    logger.info(f"📥 Attempting to load from local path: {local_path}")
                    model = AudioGen.get_pretrained(str(local_path), device=device)
                    logger.info(f"✅ Loaded AudioGen model from local path: {local_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load from local path {local_path}: {e}")
                    logger.info(f"📥 Falling back to HuggingFace (will use cache if available): {hf_model_name}")
                    model = AudioGen.get_pretrained(hf_model_name, device=device)
            else:
                logger.warning(f"⚠️ Local path {local_path} exists but doesn't contain valid model files")
                logger.info(f"📥 Using HuggingFace model (will use cache if available): {hf_model_name}")
                model = AudioGen.get_pretrained(hf_model_name, device=device)
        else:
            logger.info(f"📥 Local model not found at {local_path}, using HuggingFace (will use cache if available): {hf_model_name}")
            model = AudioGen.get_pretrained(hf_model_name, device=device)
        
        # Cache the instance
        _audiogen_instance = model
        _audiogen_model_cache[cache_key] = model
        
        logger.info(f"✅ AudioGen {model_size} model loaded successfully")
        return model
        
    except ImportError:
        logger.error("❌ audiocraft not installed. Install it with: pip install audiocraft")
        raise
    except Exception as e:
        logger.error(f"❌ Failed to load AudioGen model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


class SFXGenerator:
    """
    Generates sound effects using Meta's AudioGen model.
    """
    
    def __init__(self, model_size: str = "medium", device: Optional[str] = None):
        """
        Initialize SFXGenerator.
        
        Args:
            model_size: Model size - "small" (300M) or "medium" (1.5B)
            device: Device to use ("cuda" or "cpu"). If None, auto-detect.
        """
        self.model_size = model_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        
        logger.info(f"🔊 Initializing SFXGenerator (model: {model_size}, device: {self.device})")
    
    def _get_model(self):
        """Get or load the AudioGen model (singleton pattern)."""
        if self.model is None:
            self.model = get_audiogen_instance(model_size=self.model_size, device=self.device)
        return self.model
    
    def generate_sfx(
        self,
        prompt: str,
        duration: float,
        output_path: str,
        use_cache: bool = True
    ) -> str:
        """
        Generate SFX from text prompt.
        
        Args:
            prompt: Text description of the SFX (e.g., "room_tone", "low_electrical_ambience")
            duration: Duration in seconds (typically scene duration)
            output_path: Path to save generated audio file (WAV format)
            use_cache: If True, check for cached version first
            
        Returns:
            Path to generated audio file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check cache if enabled
        if use_cache and output_path.exists():
            logger.info(f"♻️ Using cached SFX: {output_path}")
            return str(output_path)
        
        try:
            model = self._get_model()
            
            # Use prompt directly (no enhancement needed - user provides descriptive prompt)
            logger.info(f"🔊 Generating SFX...")
            logger.info(f"   Prompt: {prompt}")
            logger.info(f"   Duration: {duration:.1f}s")
            
            # Set generation parameters
            model.set_generation_params(
                duration=duration,
                temperature=1.0,
                top_k=250,
                top_p=0.0,
                cfg_coef=3.0
            )
            
            # Generate audio using the prompt directly
            with torch.no_grad():
                wav = model.generate([prompt], progress=True)
            
            # Convert tensor to numpy and save
            import soundfile as sf
            import numpy as np
            
            # wav is a tensor with shape [batch, channels, samples] or list of tensors
            # AudioGen returns tensor of shape [1, 1, sample_rate * duration] for mono
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
            
            # Get sample rate from model (AudioGen outputs at 32kHz)
            sample_rate = getattr(model, 'sample_rate', 32000)
            
            # Save as WAV (soundfile expects 1D array for mono, or [samples, channels] for stereo)
            sf.write(str(output_path), wav_np, sample_rate)
            
            logger.info(f"✅ SFX generated: {output_path}")
            logger.info(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"❌ Error generating SFX: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

