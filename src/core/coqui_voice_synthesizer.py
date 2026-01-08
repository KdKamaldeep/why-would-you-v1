#!/usr/bin/env python3
"""
Coqui TTS Voice Synthesizer

This module provides voice synthesis using Coqui TTS models
(XTTS-v2 for multilingual, and Tacotron/VITS variants for English).
"""

import os
import re
import logging
import tempfile
import warnings
import numpy as np
import subprocess
import shutil
from typing import List, Optional, Dict, Any
from pathlib import Path
import torch
from pydantic import BaseModel
import threading
import time

# Configure TTS_HOME BEFORE any TTS imports
# TTS reads TTS_HOME when the module is first imported, so we must set it here
def _configure_tts_cache():
    """Configure TTS_HOME to use workspace folder if available."""
    # Force set TTS_HOME to workspace if available (even if already set)
    workspace_tts_dir = Path("/workspace/.cache/tts")
    workspace_exists = Path("/workspace").exists()
    
    if workspace_exists:
        # Always use workspace if it exists (override any existing TTS_HOME)
        workspace_tts_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TTS_HOME"] = str(workspace_tts_dir)
        print(f"[TTS_CONFIG] TTS_HOME set to workspace: {workspace_tts_dir}")
        logging.getLogger(__name__).info(f"📁 TTS model cache configured to: {workspace_tts_dir}")
        return str(workspace_tts_dir)
    elif "TTS_HOME" not in os.environ:
        # Use default location only if workspace doesn't exist and TTS_HOME not set
        local_tts_dir = Path.home() / ".local" / "share" / "tts"
        local_tts_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TTS_HOME"] = str(local_tts_dir)
        print(f"[TTS_CONFIG] TTS_HOME set to default: {local_tts_dir}")
        logging.getLogger(__name__).info(f"📁 TTS model cache using default: {local_tts_dir}")
        return str(local_tts_dir)
    else:
        # TTS_HOME already set, log it
        existing_home = os.environ["TTS_HOME"]
        print(f"[TTS_CONFIG] TTS_HOME already set to: {existing_home}")
        logging.getLogger(__name__).info(f"📁 TTS model cache using existing TTS_HOME: {existing_home}")
        return existing_home

# Configure TTS cache directory IMMEDIATELY (before any TTS imports)
_configure_tts_cache()

# Verify TTS_HOME is set correctly
_tts_home_verify = os.environ.get("TTS_HOME", "NOT SET")
print(f"[TTS_CONFIG] Verification - TTS_HOME = {_tts_home_verify}")

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
warnings.filterwarnings("ignore", message=".*modify your model class such that it inherits from GenerationMixin.*")

# Suppress other common TTS/transformers warnings
warnings.filterwarnings("ignore", message=".*model_kwargs.*")
warnings.filterwarnings("ignore", message=".*The attention mask and the pad token id were not set.*")
warnings.filterwarnings("ignore", message=".*Using the model-agnostic default.*")

logger = logging.getLogger(__name__)

# Global singleton TTS instance
_tts_instance = None
_tts_model_cache = {}

# Cache for speaker WAV conditioning latents (keyed by file path + language)
_speaker_wav_cache = {}

# Cache for cleaned speaker audio files (keyed by original file path)
_cleaned_speaker_cache = {}


def clean_speaker_audio(input_audio: str, output_audio: Optional[str] = None) -> str:
    """
    Normalize and clean speaker audio file for Coqui TTS.
    
    Applies:
    - Mono conversion (1 channel)
    - Resample to 22050 Hz
    - 16-bit sample format
    - Highpass filter at 80Hz
    - Lowpass filter at 12000Hz
    - Loudness normalization
    
    Args:
        input_audio: Path to input audio file
        output_audio: Path to output cleaned audio (optional, auto-generated if None)
        
    Returns:
        Path to cleaned audio file
    """
    # Check cache first
    if input_audio in _cleaned_speaker_cache:
        cached_path = _cleaned_speaker_cache[input_audio]
        if os.path.exists(cached_path):
            logger.info(f"♻️ Using cached cleaned speaker audio: {cached_path}")
            return cached_path
    
    # Generate output path if not provided
    if output_audio is None:
        input_path = Path(input_audio)
        output_audio = str(input_path.parent / f"{input_path.stem}_clean{input_path.suffix}")
    
    # Create output directory if needed
    Path(output_audio).parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"🧹 Cleaning and normalizing speaker audio: {input_audio} -> {output_audio}")
    
    # Apply cleaning filters
    cmd = [
        'ffmpeg', '-y',
        '-i', input_audio,
        '-ac', '1',  # Mono (1 channel)
        '-ar', '22050',  # Sample rate 22050 Hz
        '-sample_fmt', 's16',  # 16-bit sample format
        '-af', 'highpass=f=80,lowpass=f=12000,loudnorm',  # Filters: highpass, lowpass, loudness normalization
        output_audio
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"✅ Speaker audio cleaned: {output_audio}")
        
        # Cache the cleaned file path
        _cleaned_speaker_cache[input_audio] = output_audio
        
        return output_audio
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to clean speaker audio: {e}")
        if e.stderr:
            logger.error(f"STDERR: {e.stderr[-500:]}")
        # Fallback: return original file if cleaning fails
        logger.warning(f"⚠️ Using original audio file (cleaning failed): {input_audio}")
        return input_audio


def get_tts_instance(config: Optional["CoquiVoiceConfig"] = None, force_reload: bool = False):
    """
    Get or initialize the global TTS instance (singleton pattern).
    
    Args:
        config: Configuration for voice synthesis. Only used on first load.
        force_reload: Force reload of the TTS model even if already loaded.
        
    Returns:
        TTS API instance or None if loading fails
    """
    global _tts_instance, _tts_model_cache
    
    if _tts_instance is not None and not force_reload:
        logger.info("♻️ Reusing existing TTS instance (singleton) - model already loaded")
        return _tts_instance
    
    if config is None:
        default_model = _get_default_tts_model_path()
        config = CoquiVoiceConfig(model_name=default_model)
    elif config.model_name is None or config.model_name == "models/tts/XTTS-v2":
        config.model_name = _get_default_tts_model_path()
    
    # Create cache key based on model and device
    device = "cuda" if config.gpu and torch.cuda.is_available() else "cpu"
    cache_key = f"{config.model_name}_{device}"
    
    # Check if model is already cached
    if cache_key in _tts_model_cache and not force_reload:
        logger.info(f"♻️ Reusing cached TTS model: {cache_key}")
        _tts_instance = _tts_model_cache[cache_key]
        return _tts_instance
    
    try:
        # Ensure TTS_HOME is set before importing TTS
        if "TTS_HOME" not in os.environ:
            _configure_tts_cache()
        
        tts_home = os.environ.get("TTS_HOME", "")
        if tts_home:
            logger.info(f"📁 TTS_HOME is set to: {tts_home}")
            Path(tts_home).mkdir(parents=True, exist_ok=True)
        
        from TTS.api import TTS
        
        device = "cuda" if config.gpu and torch.cuda.is_available() else "cpu"
        lang = (config.language or "en").lower()
        
        # Helper to check if a path is a valid model directory
        def is_valid_model_path(path: Path) -> bool:
            """Check if path contains a valid TTS model."""
            if not path.exists() or not path.is_dir():
                return False
            return any([
                (path / "config.json").exists(),
                (path / "model.pth").exists(),
                (path / "vocab.json").exists(),
                any(path.glob("*.pth")),
                any(path.glob("*.pt")),
                (path / "model_file.pth").exists(),
            ])
        
        # Build prioritized list - ONLY XTTS-v2 models
        model_priority = []
        
        workspace_model_path = Path("/workspace/models/tts/XTTS-v2")
        if is_valid_model_path(workspace_model_path):
            model_priority.append(str(workspace_model_path))
            logger.info(f"Found workspace cache: {workspace_model_path}")
        
        tts_home = os.environ.get("TTS_HOME", "")
        if tts_home:
            tts_home_path = Path(tts_home)
            for possible_path in [
                tts_home_path / "tts_models" / "multilingual" / "multi-dataset" / "xtts_v2",
                tts_home_path / "coqui" / "XTTS-v2",
                tts_home_path / "XTTS-v2",
            ]:
                if is_valid_model_path(possible_path):
                    model_priority.append(str(possible_path))
                    logger.info(f"Found TTS_HOME cache: {possible_path}")
                    break
        
        local_model_path = Path("models/tts/XTTS-v2")
        if is_valid_model_path(local_model_path):
            model_priority.append(str(local_model_path))
            logger.info(f"Found local cache: {local_model_path}")
        
        # 4. Final fallback: Online model identifier
        model_priority.append("tts_models/multilingual/multi-dataset/xtts_v2")
        
        logger.info(f"📦 Loading TTS model (singleton - will be reused)...")
        logger.info(f"🔍 Model priority list: {model_priority}")
        
        # Try loading models in priority order
        tts_model = None
        last_error = None
        
        for model_path in model_priority:
            try:
                logger.info(f"Attempting to load TTS model: {model_path}")
                # Explicitly specify GPU device if available
                if device == "cuda" and torch.cuda.is_available():
                    tts_model = TTS(model_path, progress_bar=config.progress_bar).to(device)
                    logger.info(f"✅ TTS model loaded and moved to GPU: {device}")
                else:
                    tts_model = TTS(model_path, progress_bar=config.progress_bar)
                    logger.info(f"✅ TTS model loaded on CPU")
                logger.info(f"✅ Successfully loaded TTS model: {model_path}")
                break
            except Exception as e:
                last_error = e
                logger.warning(f"Failed to load model {model_path}: {e}")
                continue
        
        if tts_model is None:
            raise RuntimeError(f"Failed to load any TTS model. Last error: {last_error}")
        
        # Cache the instance
        _tts_instance = tts_model
        _tts_model_cache[cache_key] = tts_model
        logger.info(f"✅ TTS model loaded and cached (singleton): {cache_key}")
        logger.info(f"📦 TTS pipeline initialized ONCE - will be reused for all subsequent generations")
        
        return _tts_instance
        
    except Exception as e:
        logger.error(f"❌ Failed to load TTS model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


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


def _get_default_tts_model_path() -> str:
    """Get default TTS model path, preferring /workspace if available."""
    # Check for /workspace first (RunPod attached disk)
    workspace_model = Path("/workspace/models/tts/XTTS-v2")
    local_model = Path("models/tts/XTTS-v2")
    
    # Check if the model directory actually exists and has model files
    def is_valid_model_path(path: Path) -> bool:
        """Check if path is a valid TTS model directory."""
        if not path.exists():
            return False
        # Check for common model files/directories
        return any([
            (path / "config.json").exists(),
            (path / "model.pth").exists(),
            (path / "vocab.json").exists(),
            any(path.glob("*.pth")),
            any(path.glob("*.pt")),
        ])
    
    if is_valid_model_path(workspace_model):
        return str(workspace_model)
    elif is_valid_model_path(local_model):
        return str(local_model)
    else:
        # If workspace exists but model not there, return workspace path (will download)
        if Path("/workspace").exists():
            return str(workspace_model)
        else:
            # Use online model identifier if local doesn't exist
            return "coqui/XTTS-v2"


class CoquiVoiceConfig(BaseModel):
    """Configuration for Coqui TTS voice synthesis"""
    # Prefer local XTTS v2 model by default; will fall back to online models if needed
    model_name: str = "models/tts/XTTS-v2"
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
        if config is None:
            # Create config with workspace-aware default model path
            default_model = _get_default_tts_model_path()
            config = CoquiVoiceConfig(model_name=default_model)
        elif config.model_name is None or config.model_name == "models/tts/XTTS-v2":
            # Update model path if using default
            config.model_name = _get_default_tts_model_path()
        self.config = config
        
        # Create voice directory if it doesn't exist
        os.makedirs(self.config.voice_dir, exist_ok=True)
        
        # Initialize TTS model using singleton pattern
        self.tts = get_tts_instance(self.config)
        if self.tts is None:
            raise RuntimeError("Failed to initialize TTS model")
        
        logger.info(f"Coqui TTS initialized with model: {self.config.model_name}")
        logger.info(f"GPU enabled: {self.config.gpu}")
        
        # Verify and ensure model is on GPU
        if self.config.gpu and torch.cuda.is_available():
            try:
                # Try to move model to GPU explicitly
                if hasattr(self.tts, 'synthesizer') and hasattr(self.tts.synthesizer, 'model'):
                    self.tts.synthesizer.model = self.tts.synthesizer.model.cuda()
                    model_device = next(self.tts.synthesizer.model.parameters()).device
                    logger.info(f"✅ TTS model moved to GPU: {model_device}")
                elif hasattr(self.tts, 'model'):
                    self.tts.model = self.tts.model.cuda()
                    model_device = next(self.tts.model.parameters()).device
                    logger.info(f"✅ TTS model moved to GPU: {model_device}")
                else:
                    # Try to move TTS object itself
                    try:
                        self.tts = self.tts.to("cuda")
                        logger.info(f"✅ TTS object moved to GPU")
                    except:
                        logger.warning("Could not move TTS to GPU - may use CPU for inference")
            except Exception as e:
                logger.warning(f"Could not move TTS model to GPU: {e}")
        
        # Verify model device
        if hasattr(self.tts, 'synthesizer') and hasattr(self.tts.synthesizer, 'model'):
            model_device = next(self.tts.synthesizer.model.parameters()).device
            logger.info(f"TTS model device: {model_device}")
        elif hasattr(self.tts, 'model'):
            model_device = next(self.tts.model.parameters()).device
            logger.info(f"TTS model device: {model_device}")
        
        logger.info(f"Voice directory: {self.config.voice_dir}")
        logger.info(f"Language: {self.config.language}")
        
        # Cache for speaker WAV conditioning latents (per instance, keyed by file path + language)
        self._speaker_latents_cache = {}
    
    def _get_cached_speaker_latents(self, speaker_wav_path: str, language: str):
        """
        Get cached conditioning latents for a speaker WAV file, or compute and cache them.
        For XTTS models, this avoids re-encoding the speaker WAV on every call.
        
        Args:
            speaker_wav_path: Path to speaker WAV file
            language: Language code
            
        Returns:
            Tuple of (gpt_cond_latent, speaker_embedding) or None if caching not supported
        """
        cache_key = f"{speaker_wav_path}_{language}"
        
        # Check cache first
        if cache_key in self._speaker_latents_cache:
            logger.info(f"♻️ Using cached speaker conditioning latents for: {speaker_wav_path}")
            return self._speaker_latents_cache[cache_key]
        
        # Try to get conditioning latents from XTTS model
        try:
            # For XTTS models, access the underlying model to get conditioning latents
            if hasattr(self.tts, 'synthesizer') and hasattr(self.tts.synthesizer, 'model'):
                model = self.tts.synthesizer.model
                if hasattr(model, 'get_conditioning_latents'):
                    logger.info(f"📦 Computing and caching speaker conditioning latents for: {speaker_wav_path}")
                    latents = model.get_conditioning_latents(speaker_wav_path)
                    self._speaker_latents_cache[cache_key] = latents
                    logger.info(f"✅ Cached speaker conditioning latents (will reuse on next call)")
                    return latents
            # Alternative: Check if model has get_conditioning_latents method directly
            elif hasattr(self.tts, 'model') and hasattr(self.tts.model, 'get_conditioning_latents'):
                logger.info(f"📦 Computing and caching speaker conditioning latents for: {speaker_wav_path}")
                latents = self.tts.model.get_conditioning_latents(speaker_wav_path)
                self._speaker_latents_cache[cache_key] = latents
                logger.info(f"✅ Cached speaker conditioning latents (will reuse on next call)")
                return latents
            # Check if TTS has get_conditioning_latents directly
            elif hasattr(self.tts, 'get_conditioning_latents'):
                logger.info(f"📦 Computing and caching speaker conditioning latents for: {speaker_wav_path}")
                latents = self.tts.get_conditioning_latents(speaker_wav_path)
                self._speaker_latents_cache[cache_key] = latents
                logger.info(f"✅ Cached speaker conditioning latents (will reuse on next call)")
                return latents
        except Exception as e:
            logger.debug(f"Could not cache speaker latents (will encode each time): {e}")
        
        return None
    
    def _load_model(self):
        """Load the Coqui TTS model - prioritize XTTS-v2 from workspace cache, download if needed"""
        try:
            # Ensure TTS_HOME is set before importing TTS
            # TTS reads TTS_HOME when the module is first imported
            if "TTS_HOME" not in os.environ:
                _configure_tts_cache()
            
            # Verify TTS_HOME is set correctly
            tts_home = os.environ.get("TTS_HOME", "")
            if tts_home:
                logger.info(f"📁 TTS_HOME is set to: {tts_home}")
                # Ensure directory exists
                Path(tts_home).mkdir(parents=True, exist_ok=True)
            else:
                logger.warning("⚠️ TTS_HOME not set, TTS will use default location")
            
            from TTS.api import TTS
            
            # After import, verify TTS is using the correct cache
            # TTS stores models in TTS_HOME/tts_models/...
            if tts_home:
                expected_cache = Path(tts_home) / "tts_models"
                logger.info(f"📥 TTS models will be cached in: {expected_cache}")

            device = "cuda" if self.config.gpu and torch.cuda.is_available() else "cpu"
            lang = (self.config.language or "en").lower()
            
            # Helper to check if a path is a valid model directory
            def is_valid_model_path(path: Path) -> bool:
                """Check if path contains a valid TTS model."""
                if not path.exists() or not path.is_dir():
                    return False
                # Check for common model indicator files
                return any([
                    (path / "config.json").exists(),
                    (path / "model.pth").exists(),
                    (path / "vocab.json").exists(),
                    any(path.glob("*.pth")),
                    any(path.glob("*.pt")),
                    (path / "model_file.pth").exists(),
                ])
            
            # Build prioritized list - ONLY XTTS-v2 models, no fallback to other models
            model_priority = []
            
            # 1. First priority: Workspace explicit cache if it exists
            workspace_model_path = Path("/workspace/models/tts/XTTS-v2")
            if is_valid_model_path(workspace_model_path):
                model_priority.append(str(workspace_model_path))
                logger.info(f"Found workspace cache: {workspace_model_path}")
            
            # 2. Second priority: TTS_HOME cache (configured to workspace if available)
            tts_home = os.environ.get("TTS_HOME", "")
            if tts_home:
                tts_home_path = Path(tts_home)
                # Check common TTS model paths in TTS_HOME
                for possible_path in [
                    tts_home_path / "tts_models" / "multilingual" / "multi-dataset" / "xtts_v2",
                    tts_home_path / "coqui" / "XTTS-v2",
                    tts_home_path / "XTTS-v2",
                ]:
                    if is_valid_model_path(possible_path):
                        model_priority.append(str(possible_path))
                        logger.info(f"Found TTS_HOME cache: {possible_path}")
                        break
            
            # 3. Third priority: Local cache
            local_model_path = Path("models/tts/XTTS-v2")
            if is_valid_model_path(local_model_path):
                model_priority.append(str(local_model_path))
                logger.info(f"Found local cache: {local_model_path}")
            
            # 3. Third priority: Download XTTS-v2 (will download on first use)
            # Models will be downloaded to TTS_HOME (configured to workspace if available)
            # Try different XTTS-v2 model identifiers
            model_priority.extend([
                "coqui/XTTS-v2",  # Coqui's XTTS-v2 (downloads to TTS_HOME)
                "tts_models/multilingual/multi-dataset/xtts_v2",  # HuggingFace XTTS-v2 (downloads to TTS_HOME)
            ])
            
            # Log where models will be downloaded
            tts_home = os.environ.get("TTS_HOME", "")
            if tts_home:
                logger.info(f"📥 TTS models will download to: {tts_home}")
            
            logger.info(f"Loading TTS model for language: {lang}")
            logger.info(f"Model priority list: {model_priority}")
            
            last_error = None
            for model_name in model_priority:
                try:
                    logger.info(f"Attempting to load TTS model: {model_name}")
                    
                    # For local paths, verify they exist before trying to load
                    model_path = Path(model_name)
                    if model_path.exists() and model_path.is_dir():
                        # It's a local directory path - verify it's a valid model
                        if not any([
                            (model_path / "config.json").exists(),
                            (model_path / "model.pth").exists(),
                            any(model_path.glob("*.pth")),
                        ]):
                            logger.warning(f"Path exists but doesn't appear to be a valid model, skipping: {model_name}")
                            continue
                    elif "/" in model_name and not model_name.startswith("tts_models") and not model_name.startswith("coqui"):
                        # It's a local path that doesn't exist - skip it
                        logger.warning(f"Local path does not exist, skipping: {model_name}")
                        continue
                    
                    # Load TTS model - this will download if not cached
                    # For local paths, TTS might need the path in a specific format
                    try:
                        self.tts = TTS(model_name).to(device)
                    except (ValueError, TypeError) as e:
                        # If local path fails with unpacking error, try as model identifier
                        if "unpack" in str(e).lower() or "expected" in str(e).lower():
                            logger.warning(f"Local path format issue, trying as model identifier: {e}")
                            # Try using the model identifier instead
                            if "workspace" in model_name.lower() or "models/tts" in model_name.lower():
                                logger.info("Retrying with online model identifier for download")
                                self.tts = TTS("coqui/XTTS-v2").to(device)
                            else:
                                raise
                        else:
                            raise
                    
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
                    last_error = e
                    logger.warning(f"Failed to load model {model_name}: {e}")
                    continue
            
            # If we get here, all XTTS-v2 models failed
            error_msg = f"All XTTS-v2 models failed to load. Last error: {last_error}"
            logger.error(error_msg)
            raise Exception(error_msg)

        except ImportError:
            logger.error("❌ Coqui TTS not installed. Install with: pip install TTS")
            raise ImportError("Coqui TTS not available. Install with: pip install TTS")
        except Exception as e:
            logger.error(f"❌ Failed to load any TTS model: {e}")
            raise

    
    def _tts_call_with_timeout(self, timeout_seconds: int = 300, **kwargs):
        """
        Call TTS synthesis with timeout protection.
        
        Args:
            timeout_seconds: Maximum time to wait for synthesis (default: 5 minutes)
            **kwargs: Arguments to pass to tts_to_file
            
        Returns:
            True if successful, False if timeout
        """
        result = [None]
        exception = [None]
        
        def _call_tts():
            try:
                self.tts.tts_to_file(**kwargs)
                result[0] = True
            except Exception as e:
                exception[0] = e
                result[0] = False
        
        thread = threading.Thread(target=_call_tts, daemon=True)
        thread.start()
        thread.join(timeout=timeout_seconds)
        
        if thread.is_alive():
            logger.error(f"❌ TTS synthesis timed out after {timeout_seconds} seconds")
            logger.error("This may indicate:")
            logger.error("  1. Model is stuck in inference")
            logger.error("  2. Speaker WAV file is corrupted or incompatible")
            logger.error("  3. GPU/CPU resource issues")
            raise TimeoutError(f"TTS synthesis timed out after {timeout_seconds} seconds")
        
        if exception[0] is not None:
            raise exception[0]
        
        return result[0]
    
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
            # Log the voice_clone_audio parameter for debugging
            logger.info(f"🎵 Voice synthesis called with voice_clone_audio: {voice_clone_audio}")
            logger.info(f"🎵 Voice synthesis called with speaker: {speaker}")
            
            # Ensure multilingual model for non-English languages
            if (self.config.language or "en").lower() != "en":
                if "xtts" not in (getattr(self.config, 'model_name', '') or '').lower():
                    logger.info(
                        "Language is non-English (%s) but current model is not XTTS; attempting to switch to XTTS",
                        self.config.language,
                    )
                    # Prefer local XTTS-v2 (workspace path if available)
                    self.config.model_name = _get_default_tts_model_path()
                    try:
                        self._load_model()
                        logger.info("Switched TTS model to XTTS for multilingual synthesis")
                    except Exception as e:
                        logger.warning(f"Failed to switch to XTTS automatically: {e}")

            # Clean and deduplicate narration lines
            cleaned_lines = []
            seen_texts = set()
            
            for line in narration_lines:
                if line and line.strip():  # Skip empty lines
                    cleaned_line = line.strip()
                    # Only add if we haven't seen this exact text before
                    if cleaned_line not in seen_texts:
                        cleaned_lines.append(cleaned_line)
                        seen_texts.add(cleaned_line)
            
            if not cleaned_lines:
                logger.warning("No valid text lines found, using fallback text")
                if self.config.language == "hi":
                    cleaned_lines = ["नमस्ते, यह एक परीक्षण संदेश है।"]
                else:
                    cleaned_lines = ["Hello, this is a test message."]
            
            # Join cleaned lines with proper spacing and sanitize
            full_text = " ".join(cleaned_lines)
            full_text = self._sanitize_text(full_text)
            logger.info(f"Original lines: {len(narration_lines)}, Cleaned lines: {len(cleaned_lines)}")
            logger.info(f"Synthesizing voice for text: {full_text[:100]}...")
            
            # Use text as-is without any padding or modifications
            full_text = full_text.strip()
            
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
                        import unicodedata
                        # NFC is best for XTTS tokenization; also strip Latin noise for Hindi
                        full_text = unicodedata.normalize('NFC', full_text)
                        if self.config.language == "hi":
                            # Remove stray Latin letters or unsupported chars that cause gibberish
                            # Keep Devanagari, punctuation and spaces
                            full_text = re.sub(r"[^\u0900-\u097F\s\.,!?\-–—'\"]+", " ", full_text)
                            full_text = re.sub(r"\s+", " ", full_text).strip()
                        logger.info(f"Normalized {self.config.language} text: {full_text[:50]}...")
                    except Exception as e:
                        logger.warning(f"Text normalization warning: {e}")
                
                # Honor voice_clone_audio if provided (from --voice argument)
                # Only auto-discover if voice_clone_audio was NOT provided (None or empty string)
                # Normalize: treat empty string as None, strip whitespace from non-empty strings
                voice_clone_audio_normalized = None
                if voice_clone_audio:
                    if isinstance(voice_clone_audio, str):
                        stripped = voice_clone_audio.strip()
                        if stripped:  # Only use if non-empty after stripping
                            voice_clone_audio_normalized = stripped
                    else:
                        voice_clone_audio_normalized = voice_clone_audio
                
                logger.info(f"🎵 Voice file check:")
                logger.info(f"   voice_clone_audio (raw): {repr(voice_clone_audio)}")
                logger.info(f"   voice_clone_audio (normalized): {repr(voice_clone_audio_normalized)}")
                
                if voice_clone_audio_normalized:
                    # User explicitly provided a voice file - use it or error
                    if os.path.exists(voice_clone_audio_normalized):
                        # Clean and normalize the speaker audio before using
                        speaker_wav_arg = clean_speaker_audio(voice_clone_audio_normalized)
                        logger.info(f"🎵 Using provided voice file (cleaned): {speaker_wav_arg}")
                    else:
                        # User provided a path but file doesn't exist - this is an error, don't fall back to discovery
                        logger.error(f"❌ Voice file not found: {voice_clone_audio_normalized}")
                        logger.error(f"   Provided via --voice argument but file does not exist")
                        raise FileNotFoundError(f"Voice file not found: {voice_clone_audio_normalized}")
                else:
                    # No voice file provided - auto-discover based on language
                    speaker_wav_arg = None
                    auto_wav = self._discover_speaker_wav(self.config.language)
                    if auto_wav:
                        # Clean and normalize the auto-discovered speaker audio
                        speaker_wav_arg = clean_speaker_audio(auto_wav)
                        logger.info(f"🎵 Auto-discovered speaker_wav for language '{self.config.language}' (cleaned): {speaker_wav_arg}")
                    else:
                        logger.info(f"🎵 No speaker_wav found - will use default XTTS speaker")
                
                logger.info(f"🎵 XTTS: speaker_wav_arg set to: {speaker_wav_arg}")
                logger.info(f"🎵 XTTS: voice_clone_audio was: {voice_clone_audio}")
                
                # Ensure a valid speaker is passed for XTTS if no reference wav
                requested_speaker = (
                    speaker if (speaker is not None and str(speaker).strip() != "") else self.config.speaker
                )
                xtts_speaker = self._select_xtts_speaker(requested_speaker if speaker_wav_arg is None else None)
                logger.info(f"XTTS selected speaker: {xtts_speaker if speaker_wav_arg is None else 'speaker_wav provided'}")

                def _xtts_call(speaker_value: Optional[str]) -> None:
                    # Avoid passing progress_bar to suppress model_kwargs warnings
                    if speaker_wav_arg is not None:
                        # Validate speaker_wav file before using
                        if not os.path.exists(speaker_wav_arg):
                            raise FileNotFoundError(f"Speaker WAV file not found: {speaker_wav_arg}")
                        
                        # Try to use cached conditioning latents to avoid re-encoding
                        cached_latents = self._get_cached_speaker_latents(speaker_wav_arg, self.config.language)
                        
                        logger.info(f"XTTS synthesis with speaker_wav: {speaker_wav_arg}")
                        logger.info(f"Text length: {len(full_text)} characters")
                        
                        start_time = time.time()
                        try:
                            # If we have cached latents, try to use them directly with the model
                            if cached_latents is not None:
                                try:
                                    # Access the underlying model to use cached latents
                                    if hasattr(self.tts, 'synthesizer') and hasattr(self.tts.synthesizer, 'model'):
                                        model = self.tts.synthesizer.model
                                        # Use model's inference with cached latents
                                        # This bypasses the speaker_wav encoding step
                                        logger.info("Using cached speaker conditioning latents (skipping re-encoding)")
                                        wav = model.inference(
                                            full_text,
                                            self.config.language,
                                            cached_latents[0],  # gpt_cond_latent
                                            cached_latents[1],  # speaker_embedding
                                        )
                                        # Save the generated audio
                                        import soundfile as sf
                                        sf.write(output_path, wav, samplerate=22050)
                                    elif hasattr(self.tts, 'model') and hasattr(self.tts.model, 'inference'):
                                        logger.info("Using cached speaker conditioning latents (skipping re-encoding)")
                                        wav = self.tts.model.inference(
                                            full_text,
                                            self.config.language,
                                            cached_latents[0],  # gpt_cond_latent
                                            cached_latents[1],  # speaker_embedding
                                        )
                                        import soundfile as sf
                                        sf.write(output_path, wav, samplerate=22050)
                                    else:
                                        # Fall back to standard call if model structure is different
                                        raise AttributeError("Model structure not recognized for cached latents")
                                except Exception as e:
                                    logger.warning(f"Failed to use cached latents, falling back to speaker_wav: {e}")
                                    # Fall back to standard call
                                    self.tts.tts_to_file(
                                        text=full_text,
                                        file_path=output_path,
                                        speaker_wav=speaker_wav_arg,
                                        language=self.config.language,
                                    )
                            else:
                                # No cached latents - standard call (will encode speaker_wav)
                                # Ensure GPU is used for inference
                                if torch.cuda.is_available() and self.config.gpu:
                                    # Force GPU context for inference
                                    with torch.cuda.device(0):
                                        self.tts.tts_to_file(
                                            text=full_text,
                                            file_path=output_path,
                                            speaker_wav=speaker_wav_arg,
                                            language=self.config.language,
                                        )
                                else:
                                    self.tts.tts_to_file(
                                        text=full_text,
                                        file_path=output_path,
                                        speaker_wav=speaker_wav_arg,
                                        language=self.config.language,
                                    )
                            elapsed = time.time() - start_time
                            logger.info(f"✅ TTS synthesis completed in {elapsed:.2f} seconds")
                        except Exception as e:
                            logger.error(f"TTS synthesis error: {e}")
                            raise
                    else:
                        # No reference: pass an explicit speaker token
                        chosen_speaker = speaker_value or self.config.speaker or "default"
                        logger.info(f"XTTS synthesis with speaker: {chosen_speaker}")
                        
                        # Ensure GPU is used for inference
                        if torch.cuda.is_available() and self.config.gpu:
                            # Force GPU context for inference
                            with torch.cuda.device(0):
                                self.tts.tts_to_file(
                                    text=full_text,
                                    file_path=output_path,
                                    speaker=chosen_speaker,
                                    language=self.config.language,
                                )
                        else:
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
                
                # Strategy 4: If all strategies fail, raise the error
                # No longer adding padding text - use original text as-is
                
                if not synthesis_success:
                    logger.error(f"All XTTS synthesis strategies failed: {synthesis_errors}")
                    raise Exception(f"XTTS synthesis failed: {'; '.join(synthesis_errors)}")
            else:
                # Non-XTTS models: use speaker registry in voice_dir
                current_speaker = speaker or self.config.speaker
                # Optional: create a named speaker from provided audio for YourTTS-like models
                if voice_clone_audio and os.path.exists(voice_clone_audio):
                    logger.info(f"Cloning voice (registry) from: {voice_clone_audio}")
                    # Clean and normalize the speaker audio before copying to registry
                    cleaned_audio = clean_speaker_audio(voice_clone_audio)
                    speaker_name = os.path.splitext(os.path.basename(voice_clone_audio))[0]
                    speaker_dir = os.path.join(self.config.voice_dir, speaker_name)
                    os.makedirs(speaker_dir, exist_ok=True)
                    speaker_audio_path = os.path.join(speaker_dir, "speaker.wav")
                    shutil.copy2(cleaned_audio, speaker_audio_path)
                    logger.info(f"✅ Copied cleaned speaker audio to registry: {speaker_audio_path}")
                    current_speaker = speaker_name

                # First attempt: whole text
                synthesis_success = False
                try:
                    logger.info(f"Generating audio with speaker: {current_speaker}")
                    self._safe_tts_to_file_non_xtts(
                        text=full_text,
                        file_path=output_path,
                        speaker=current_speaker
                    )
                    synthesis_success = True
                except Exception as e:
                    error_msg = str(e)
                    logger.warning(f"Whole-text synthesis failed: {error_msg}")
                    
                # Fallback 1: If whole-text synthesis failed, try sentence-by-sentence
                # No longer adding padding text - use original text as-is

                # Fallback 2: split into sentences and concatenate
                if not synthesis_success:
                    try:
                        logger.info("Falling back to sentence-by-sentence synthesis and concatenation")
                        self._synthesize_non_xtts_by_sentences(
                            text=full_text,
                            output_path=output_path,
                            speaker=current_speaker
                        )
                        synthesis_success = True
                        logger.info("✅ Non-XTTS synthesis successful by concatenating sentences")
                    except Exception as e3:
                        logger.error(f"Sentence-by-sentence synthesis failed: {e3}")
                        raise e3

                if not synthesis_success:
                    raise Exception("Non-XTTS synthesis failed")
            
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
    

    
    def move_to_cpu(self):
        """Move TTS model to CPU to free VRAM."""
        try:
            if self.tts is None:
                return
            
            if torch.cuda.is_available():
                # Move model to CPU
                if hasattr(self.tts, 'synthesizer') and hasattr(self.tts.synthesizer, 'model'):
                    self.tts.synthesizer.model = self.tts.synthesizer.model.cpu()
                    logger.info("✅ TTS model moved to CPU")
                elif hasattr(self.tts, 'model'):
                    self.tts.model = self.tts.model.cpu()
                    logger.info("✅ TTS model moved to CPU")
                else:
                    try:
                        self.tts = self.tts.to("cpu")
                        logger.info("✅ TTS object moved to CPU")
                    except:
                        logger.warning("⚠️ Could not move TTS to CPU")
                
                # Clear CUDA cache
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                logger.info("💾 TTS pipeline offloaded to CPU (VRAM freed)")
        except Exception as e:
            logger.warning(f"⚠️ Failed to move TTS to CPU: {e}")
    
    def move_to_gpu(self):
        """Move TTS model back to GPU when needed."""
        try:
            if self.tts is None:
                return
            
            if self.config.gpu and torch.cuda.is_available():
                # Move model back to GPU
                if hasattr(self.tts, 'synthesizer') and hasattr(self.tts.synthesizer, 'model'):
                    self.tts.synthesizer.model = self.tts.synthesizer.model.cuda()
                    logger.info("✅ TTS model moved back to GPU")
                elif hasattr(self.tts, 'model'):
                    self.tts.model = self.tts.model.cuda()
                    logger.info("✅ TTS model moved back to GPU")
                else:
                    try:
                        self.tts = self.tts.to("cuda")
                        logger.info("✅ TTS object moved back to GPU")
                    except:
                        logger.warning("⚠️ Could not move TTS to GPU")
        except Exception as e:
            logger.warning(f"⚠️ Failed to move TTS to GPU: {e}")
    
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

    # -------------------------
    # Helpers: text + synthesis
    # -------------------------
    def _sanitize_text(self, text: str) -> str:
        """Sanitize input text to avoid decoder issues.

        - Normalize whitespace
        - Remove stray, unmatched quotes at the end
        - Ensure sentences end with proper punctuation
        - Fix common unmatched single-quote cases like: He said, 'Hello!
        """
        if not text:
            return text

        # Normalize whitespace
        txt = re.sub(r"\s+", " ", text).strip()

        # Replace fancy quotes with straight quotes for stability
        txt = txt.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")

        # Remove isolated single or double quote tokens
        txt = re.sub(r'\s+\'\s+', ' ', txt)
        txt = re.sub(r'\s+"+\s+', ' ', txt)

        # If text ends with a lone quote, drop it
        txt = re.sub(r"([.!?])['\"]?$", r"\1", txt)
        txt = re.sub(r"(^|\s)['\"]($|\s)", r" ", txt).strip()

        # Note: We avoid aggressive quote-matching heuristics to prevent regex errors.

        return txt

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences conservatively and clean stray quotes."""
        if not text:
            return []
        # Primary split on punctuation followed by whitespace
        parts = re.split(r"(?<=[.!?])\s+", text)
        cleaned: List[str] = []
        for p in parts:
            s = p.strip()
            if not s:
                continue
            # Remove trailing lone quotes
            s = re.sub(r"^['\"]+(.*)$", r"\1", s)
            s = re.sub(r"^(.*)['\"]+$", r"\1", s)
            # Ensure terminal punctuation
            if not re.search(r"[.!?]$", s):
                s = s + "."
            cleaned.append(s)
        return cleaned

    def _safe_tts_to_file_non_xtts(self, text: str, file_path: str, speaker: Optional[str]) -> None:
        """Call tts_to_file for non-XTTS models, passing speaker if supported."""
        try:
            # Some models accept speaker token
            self.tts.tts_to_file(
                text=text,
                file_path=file_path,
                voice_dir=self.config.voice_dir,
                speaker=speaker,
            )
        except TypeError:
            # Older single-speaker models may not accept speaker kwarg
            self.tts.tts_to_file(
                text=text,
                file_path=file_path,
                voice_dir=self.config.voice_dir,
            )

    def _synthesize_non_xtts_by_sentences(self, text: str, output_path: str, speaker: Optional[str]) -> None:
        """Synthesize sentence by sentence and concatenate to a single WAV."""
        import tempfile
        import wave

        sentences = self._split_sentences(text)
        if not sentences:
            raise RuntimeError("No sentences to synthesize")

        temp_paths: List[str] = []
        try:
            for idx, sent in enumerate(sentences):
                tmp_path = os.path.join(tempfile.gettempdir(), f"coqui_sent_{os.getpid()}_{idx}.wav")
                self._safe_tts_to_file_non_xtts(sent, tmp_path, speaker)
                if not os.path.exists(tmp_path):
                    raise RuntimeError(f"Failed to synthesize sentence {idx+1}")
                temp_paths.append(tmp_path)

            # Concatenate WAV files
            with wave.open(output_path, 'wb') as out_wav:
                # Initialize with params from first file
                with wave.open(temp_paths[0], 'rb') as first:
                    out_wav.setparams(first.getparams())
                    out_wav.writeframes(first.readframes(first.getnframes()))

                # Append the rest
                for p in temp_paths[1:]:
                    with wave.open(p, 'rb') as w:
                        # If params mismatch, log and still try to append raw frames
                        if w.getparams() != out_wav.getparams():
                            logger.warning("WAV parameter mismatch during concatenation; attempting raw append")
                        out_wav.writeframes(w.readframes(w.getnframes()))
        finally:
            # Cleanup temp files
            for p in temp_paths:
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass

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