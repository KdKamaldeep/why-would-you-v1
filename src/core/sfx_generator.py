#!/usr/bin/env python3
"""
SFX Generator Module

This module generates sound effects from text prompts using AudioLDM or similar models.
"""

import os
import logging
import subprocess
from typing import Optional
from pathlib import Path
import torch
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

try:
    from diffusers import AudioLDMPipeline
    AUDIOLDM_AVAILABLE = True
except ImportError:
    AUDIOLDM_AVAILABLE = False
    logger.warning("AudioLDM not available. SFX generation will use fallback method.")


def _get_local_model_path() -> Optional[Path]:
    """
    Get local model path if available, following the same pattern as WAN and TTS.
    Checks for models in the models/ directory first.
    
    Returns:
        Path to local model directory if exists, None otherwise
    """
    # Check for local model in models/sfx directory (from download_models.sh)
    local_model_path = Path("models/sfx/audioldm-s-full-v2")
    if local_model_path.exists():
        # Check if it's a valid model directory (has model files)
        model_files = list(local_model_path.glob("*.safetensors")) + \
                     list(local_model_path.glob("*.bin")) + \
                     list(local_model_path.glob("*.json"))
        if model_files:
            logger.info(f"📁 Found local AudioLDM model: {local_model_path}")
            return local_model_path
    
    return None


def _get_cache_dir() -> Optional[str]:
    """
    Get the appropriate cache directory for Hugging Face models.
    Prefers /workspace if available (for RunPod and similar environments with attached disks).
    
    Returns:
        Cache directory path as string, or None to use default
    """
    workspace_cache = Path("/workspace/.cache/huggingface")
    if Path("/workspace").exists():
        workspace_cache.mkdir(parents=True, exist_ok=True)
        # Set HF_HOME environment variable as well
        os.environ["HF_HOME"] = str(workspace_cache)
        return str(workspace_cache)
    return None


class SFXGenerator:
    """Generates sound effects from text prompts."""
    
    def __init__(self, model_id: str = "cvssp/audioldm-s-full-v2", device: Optional[str] = None):
        """
        Initialize SFX generator.
        
        Args:
            model_id: HuggingFace model identifier for AudioLDM (or local path)
            device: Device to run on ('cuda' or 'cpu')
        """
        # Check for local model first
        local_model = _get_local_model_path()
        if local_model:
            self.model_id = str(local_model)
            logger.info(f"📦 Using local AudioLDM model from: {self.model_id}")
        else:
            self.model_id = model_id
            logger.info(f"📦 Will download AudioLDM model from HuggingFace: {self.model_id}")
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None
        self._initialized = False
        
    def _initialize_pipeline(self):
        """Lazy initialization of the AudioLDM pipeline."""
        if self._initialized:
            return
            
        if not AUDIOLDM_AVAILABLE:
            logger.warning("AudioLDM not available. Using silent audio fallback.")
            self._initialized = True
            return
            
        try:
            logger.info(f"Loading AudioLDM pipeline: {self.model_id}")
            logger.info(f"Using device: {self.device}")
            
            # Configure Hugging Face cache directory to use /workspace if available
            cache_dir = _get_cache_dir()
            if cache_dir:
                logger.info(f"📁 Using Hugging Face cache directory: {cache_dir}")
            
            # Determine torch dtype based on device
            if self.device == "cuda" and torch.cuda.is_available():
                torch_dtype = torch.float16
                logger.info("✅ Using float16 on CUDA for optimal performance")
            else:
                torch_dtype = torch.float32
                if self.device == "cpu":
                    logger.warning("⚠️ Running on CPU - SFX generation will be slower. Consider using GPU.")
            
            # Load pipeline (from local path or HuggingFace)
            self.pipeline = AudioLDMPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                cache_dir=cache_dir
            )
            self.pipeline = self.pipeline.to(self.device)
            logger.info("✅ AudioLDM pipeline loaded successfully")
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to load AudioLDM pipeline: {e}")
            logger.warning("Will use silent audio fallback for SFX generation")
            self.pipeline = None
            self._initialized = True
    
    def generate_sfx(
        self,
        prompt: str,
        duration: float,
        output_path: str,
        num_inference_steps: int = 10,
        guidance_scale: float = 2.5
    ) -> str:
        """
        Generate sound effect audio file from text prompt.
        
        Args:
            prompt: Text description of the sound effect
            duration: Target duration in seconds
            output_path: Path to save the generated audio file
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for generation
            
        Returns:
            Path to generated audio file
        """
        self._initialize_pipeline()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # If pipeline not available, create silent audio as fallback
        if not self.pipeline:
            logger.warning(f"Creating silent audio fallback for SFX: {prompt}")
            return self._create_silent_audio(duration, str(output_path))
        
        try:
            logger.info(f"Generating SFX: '{prompt}' (target duration: {duration:.2f}s)")
            
            # AudioLDM generates 10-second clips by default
            # We'll generate and loop/trim as needed
            audio = self.pipeline(
                prompt,
                num_inference_steps=num_inference_steps,
                audio_length_in_s=int(min(duration, 10)),  # AudioLDM supports up to 10s
                guidance_scale=guidance_scale
            ).audios[0]
            
            # Convert to mono if stereo, and ensure correct format
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=0)
            
            # Ensure audio is in the right range [-1, 1]
            audio = np.clip(audio, -1.0, 1.0)
            
            # Check if audio has actual content (not silent)
            max_amplitude = np.max(np.abs(audio))
            if max_amplitude < 0.001:
                logger.warning(f"⚠️ Generated SFX appears to be silent (max amplitude: {max_amplitude:.6f})")
                logger.warning(f"   This might indicate an issue with AudioLDM generation for prompt: '{prompt}'")
            else:
                logger.info(f"✅ SFX has audio content (max amplitude: {max_amplitude:.4f})")
            
            # Normalize audio to ensure it's audible (but not clipping)
            # Boost quiet audio to make it more audible
            if max_amplitude > 0:
                target_peak = 0.8  # Target 80% peak to avoid clipping
                if max_amplitude < target_peak:
                    gain = target_peak / max_amplitude
                    audio = audio * gain
                    audio = np.clip(audio, -1.0, 1.0)
                    logger.info(f"🔊 Boosted SFX audio by {gain:.2f}x to improve audibility")
            
            # Save initial audio
            temp_audio = str(output_path).replace('.wav', '_temp.wav')
            sf.write(temp_audio, audio, 16000)  # AudioLDM uses 16kHz
            
            # Extend or trim to match target duration
            final_audio = self._adjust_audio_duration(temp_audio, duration, str(output_path))
            
            # Clean up temp file
            try:
                if os.path.exists(temp_audio) and temp_audio != final_audio:
                    os.remove(temp_audio)
            except Exception:
                pass
            
            logger.info(f"✅ SFX generated: {output_path} ({duration:.2f}s)")
            return final_audio
            
        except Exception as e:
            logger.error(f"Error generating SFX: {e}")
            logger.warning(f"Creating silent audio fallback for: {prompt}")
            return self._create_silent_audio(duration, str(output_path))
    
    def _adjust_audio_duration(self, input_audio: str, target_duration: float, output_audio: str) -> str:
        """
        Adjust audio duration by looping or trimming to match target duration.
        
        Args:
            input_audio: Path to input audio file
            target_duration: Target duration in seconds
            output_audio: Path to output audio file
            
        Returns:
            Path to output audio file
        """
        try:
            # Get current duration
            cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'csv=p=0', input_audio
            ]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            current_duration = float(result.stdout.strip())
            
            if abs(current_duration - target_duration) < 0.1:
                # Duration is close enough, just copy
                import shutil
                shutil.copy2(input_audio, output_audio)
                return output_audio
            
            # Calculate loop count needed
            if current_duration < target_duration:
                # Need to loop
                loop_count = int(target_duration / current_duration) + 1
                
                # Create concat file
                concat_file = output_audio.replace('.wav', '_concat.txt')
                with open(concat_file, 'w') as f:
                    for _ in range(loop_count):
                        f.write(f"file '{os.path.abspath(input_audio)}'\n")
                
                # Concatenate
                looped_audio = output_audio.replace('.wav', '_looped.wav')
                cmd = [
                    'ffmpeg', '-y',
                    '-f', 'concat', '-safe', '0',
                    '-i', concat_file,
                    '-c', 'copy',
                    looped_audio
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                
                # Trim to exact duration
                cmd = [
                    'ffmpeg', '-y',
                    '-i', looped_audio,
                    '-t', str(target_duration),
                    '-c', 'copy',
                    output_audio
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                
                # Cleanup
                try:
                    os.remove(concat_file)
                    os.remove(looped_audio)
                except Exception:
                    pass
            else:
                # Need to trim
                cmd = [
                    'ffmpeg', '-y',
                    '-i', input_audio,
                    '-t', str(target_duration),
                    '-c', 'copy',
                    output_audio
                ]
                subprocess.run(cmd, check=True, capture_output=True)
            
            return output_audio
            
        except Exception as e:
            logger.error(f"Error adjusting audio duration: {e}")
            # Fallback: just copy the input
            import shutil
            shutil.copy2(input_audio, output_audio)
            return output_audio
    
    def _create_silent_audio(self, duration: float, output_path: str) -> str:
        """
        Create silent audio file as fallback.
        
        Args:
            duration: Duration in seconds
            output_path: Path to save the silent audio file
            
        Returns:
            Path to created audio file
        """
        try:
            cmd = [
                'ffmpeg', '-y',
                '-f', 'lavfi',
                '-i', f'anullsrc=channel_layout=mono:sample_rate=44100:duration={duration}',
                '-ar', '44100',
                '-ac', '1',
                output_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Created silent audio: {output_path} ({duration:.2f}s)")
            return output_path
        except Exception as e:
            logger.error(f"Error creating silent audio: {e}")
            return output_path

