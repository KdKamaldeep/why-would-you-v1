#!/usr/bin/env python3
"""
WAN 2.2 Text-Image-to-Video Generator Module
Handles text-to-video generation using Wan-AI/Wan2.2-TI2V-5B
Supports both text-to-video (T2V) and image-to-video (I2V) modes
Uses WanImageToVideoPipeline for I2V mode and WanPipeline for T2V mode
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
import logging
import torch
from pathlib import Path
from typing import Optional, Dict, Any, Union
import gc
import subprocess
import tempfile
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Environment variables for optional FPS interpolation
INTERPOLATE_FPS = os.getenv("INTERPOLATE_FPS", "0").lower() in ("1", "true", "yes")
INTERPOLATE_TARGET_FPS = int(os.getenv("INTERPOLATE_TARGET_FPS", "24"))


def interpolate_video_ffmpeg(input_path, output_path, target_fps=24):
    """
    Interpolate video FPS using ffmpeg minterpolate filter.
    High-quality motion-compensated interpolation.
    
    Args:
        input_path: Input video file path
        output_path: Output video file path (will be overwritten)
        target_fps: Target FPS for interpolation (default: 24)
    
    Raises:
        RuntimeError: If ffmpeg is not found or interpolation fails
        subprocess.CalledProcessError: If ffmpeg command fails
    """
    import subprocess
    import shutil
    
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found")
    
    # High-quality minterpolate settings:
    # - mi_mode=mci: Motion-compensated interpolation
    # - mc_mode=aobmc: Adaptive overlapped block motion compensation
    # - me_mode=bidir: Bidirectional motion estimation
    # - vsbmc=1: Variable-size block motion compensation
    filter_complex = (
        f"minterpolate=fps={target_fps}:"
        f"mi_mode=mci:"
        f"mc_mode=aobmc:"
        f"me_mode=bidir:"
        f"vsbmc=1"
    )
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", filter_complex,
        "-c:v", "libx264",
        "-crf", "18",  # High quality
        "-preset", "slow",  # Better compression efficiency
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path)
    ]
    
    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        error_msg = result.stderr[:500] if result.stderr else 'Unknown error'
        raise subprocess.CalledProcessError(result.returncode, cmd, error_msg)


def write_frames_fast(frames, out_dir):
    """Write frames as JPEG files quickly using OpenCV."""
    import os
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(
            os.path.join(out_dir, f"{i:06d}.jpg"),
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), 92]
        )


def encode_video_ffmpeg_pipe(frames_np, fps, output_path):
    """
    Encode video by piping raw RGB frames directly to ffmpeg via stdin.
    Much faster than writing to disk and re-reading.
    
    Args:
        frames_np: uint8 RGB array [T, H, W, 3]
        fps: Frames per second
        output_path: Output video file path
    
    Raises:
        RuntimeError: If ffmpeg is not found or encoding fails
    """
    import subprocess
    import shutil
    
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found")
    
    # Get dimensions
    num_frames, height, width, channels = frames_np.shape
    if channels != 3:
        raise ValueError(f"Expected RGB frames [T,H,W,3], got shape {frames_np.shape}")
    
    # Ensure frames are contiguous in memory for efficient tobytes()
    if not frames_np.flags['C_CONTIGUOUS']:
        frames_np = np.ascontiguousarray(frames_np)
    
    # Prepare frame data as bytes
    frame_bytes = frames_np.tobytes()
    frame_size = height * width * 3  # RGB24
    
    # Try GPU NVENC first (FAST)
    try:
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "pipe:0",
            "-c:v", "h264_nvenc",
            "-preset", "p1",  # p1 = fastest, p7 = slowest (best quality)
            "-rc", "vbr",  # Variable bitrate mode
            "-b:v", "10M",  # Target bitrate
            "-maxrate", "20M",  # Max bitrate
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path
        ]
        
        # Use Popen to write to stdin (don't use capture_output=True - can block)
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE
        )
        
        # Write all frames to stdin
        process.stdin.write(frame_bytes)
        process.stdin.close()
        
        # Wait for completion and check return code
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode('utf-8', errors='ignore')[:500] if stderr else 'Unknown error'
            raise subprocess.CalledProcessError(process.returncode, cmd, error_msg)
        
        logger.info("✅ Video encoded using GPU NVENC (h264_nvenc) via pipe")
        return
        
    except (subprocess.CalledProcessError, OSError) as e:
        error_msg = str(e)
        if hasattr(e, 'stderr') and e.stderr:
            error_msg = e.stderr.decode('utf-8', errors='ignore')[:500] if isinstance(e.stderr, bytes) else str(e.stderr)[:500]
        logger.warning(f"⚠️ NVENC encoding via pipe failed: {error_msg}")
        logger.info("🔄 Falling back to CPU encoding via pipe...")
    
    # CPU fallback via pipe
    try:
        logger.info("💻 Using CPU encoding (libx264) via pipe")
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "pipe:0",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path
        ]
        
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE
        )
        
        # Write all frames to stdin
        process.stdin.write(frame_bytes)
        process.stdin.close()
        
        # Wait for completion and check return code
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode('utf-8', errors='ignore')[:500] if stderr else 'Unknown error'
            raise subprocess.CalledProcessError(process.returncode, cmd, error_msg)
        
        logger.info("✅ Video encoded using CPU (libx264) via pipe")
        return
        
    except Exception as e:
        error_msg = str(e)
        if hasattr(e, 'stderr') and e.stderr:
            error_msg = e.stderr.decode('utf-8', errors='ignore')[:500] if isinstance(e.stderr, bytes) else str(e.stderr)[:500]
        raise RuntimeError(f"Failed to encode video via pipe: {error_msg}")


def encode_video_ffmpeg(frames_dir, fps, output_path):
    """Encode video using FFmpeg with NVENC (GPU) or CPU fallback."""
    import subprocess
    import shutil

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found")

    # Try GPU NVENC first (FAST)
    try:
        result = subprocess.run([
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", f"{frames_dir}/%06d.jpg",
            "-c:v", "h264_nvenc",
            "-preset", "p1",  # p1 = fastest, p7 = slowest (best quality)
            "-rc", "vbr",  # Variable bitrate mode
            "-b:v", "10M",  # Target bitrate (adjust if needed)
            "-maxrate", "20M",  # Max bitrate
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path
        ], check=True, capture_output=True, text=True)
        logger.info("✅ Video encoded using GPU NVENC (h264_nvenc)")
        return
    except subprocess.CalledProcessError as e:
        logger.warning(f"⚠️ NVENC encoding failed: {e.stderr[:500] if e.stderr else 'Unknown error'}")
        logger.info("🔄 Falling back to CPU encoding...")
    except Exception as e:
        logger.warning(f"⚠️ NVENC encoding error: {e}")
        logger.info("🔄 Falling back to CPU encoding...")

    # CPU fallback (still fast)
    logger.info("💻 Using CPU encoding (libx264)")
    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", f"{frames_dir}/%06d.jpg",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path
    ], check=True)


# Global singleton instances (separate for T2V and I2V)
_wan_pipeline_t2v = None
_wan_pipeline_i2v = None
_wan_vae = None
# Shared components to avoid loading twice
_wan_shared_components = None  # Will store transformer, text_encoder, etc.


def get_cache_dir() -> Optional[str]:
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


def apply_fastwan_lora(pipeline, lora_id_or_path, weight_name=None, scale=1.0):
    """
    Apply FastWan (distilled/lightning) LoRA to the pipeline.
    
    Args:
        pipeline: The WAN pipeline instance
        lora_id_or_path: LoRA repository ID or local path
        weight_name: Optional specific weight file name (e.g., "pytorch_lora_weights.safetensors")
        scale: LoRA scale (default: 1.0)
    
    Returns:
        True if LoRA was successfully applied, False otherwise
    """
    try:
        # Check if diffusers LoRA APIs are available
        if not hasattr(pipeline, 'load_lora_weights'):
            logger.warning("ℹ️ FastWan disabled: LoRA APIs not available in this diffusers version")
            return False
        
        logger.info(f"⚡ Loading FastWan LoRA from: {lora_id_or_path}")
        
        # Load LoRA weights
        if weight_name:
            pipeline.load_lora_weights(lora_id_or_path, weight_name=weight_name)
            logger.info(f"⚡ FastWan LoRA loaded: {lora_id_or_path} (weight: {weight_name})")
        else:
            pipeline.load_lora_weights(lora_id_or_path)
            logger.info(f"⚡ FastWan LoRA loaded: {lora_id_or_path}")
        
        # Try to fuse LoRA for better performance (if available)
        if hasattr(pipeline, 'fuse_lora'):
            try:
                pipeline.fuse_lora(lora_scale=scale)
                logger.info(f"✅ FastWan LoRA fused (scale: {scale})")
            except Exception as e:
                logger.warning(f"⚠️ Could not fuse LoRA (will use unfused): {e}")
        elif hasattr(pipeline, 'set_adapters'):
            # Alternative: use adapter API if available
            try:
                pipeline.set_adapters(["default"], adapter_weights=[scale])
                logger.info(f"✅ FastWan LoRA adapter set (scale: {scale})")
            except Exception as e:
                logger.warning(f"⚠️ Could not set LoRA adapter: {e}")
        
        return True
        
    except ImportError:
        logger.warning("ℹ️ FastWan disabled: LoRA support not available")
        return False
    except Exception as e:
        logger.warning(f"⚠️ Failed to apply FastWan LoRA: {e}")
        return False


def try_enable_teacache(pipeline, threshold=0.12):
    """
    Enable TeaCache-style transformer caching for the pipeline.
    Only wraps pipeline.transformer, never crashes if TeaCache is missing.
    
    Args:
        pipeline: The WAN pipeline instance
        threshold: Cache threshold (default: 0.12)
    
    Returns:
        True if TeaCache was successfully enabled, False otherwise
    """
    try:
        # Try to import TeaCache
        try:
            from teacache import TeaCache
        except ImportError:
            logger.info("ℹ️ TeaCache disabled: teacache package not installed")
            return False
        
        # Check if pipeline has transformer
        if not hasattr(pipeline, 'transformer'):
            logger.warning("⚠️ TeaCache: Pipeline does not have transformer attribute")
            return False
        
        # Wrap transformer with TeaCache (only wraps transformer, preserves other components)
        original_transformer = pipeline.transformer
        pipeline.transformer = TeaCache(original_transformer, threshold=threshold)
        
        logger.info(f"🫖 TeaCache enabled (threshold: {threshold})")
        return True
        
    except ImportError:
        logger.info("ℹ️ TeaCache disabled: teacache package not installed")
        return False
    except Exception as e:
        logger.warning(f"⚠️ Failed to enable TeaCache: {e}")
        return False


def get_wan_pipeline(device: str = None, force_reload: bool = False, use_i2v: bool = False):
    """
    Get or initialize the global WAN pipeline (singleton pattern).
    
    Args:
        device: Device to run on ('cuda' or 'cpu'). Auto-detected if None.
        force_reload: Force reload of the pipeline even if already loaded.
        use_i2v: If True, load WanImageToVideoPipeline for I2V mode. If False, load WanPipeline for T2V mode.
        
    Returns:
        WanPipeline or WanImageToVideoPipeline instance or None if loading fails
    """
    global _wan_pipeline_t2v, _wan_pipeline_i2v, _wan_vae
    
    # Select the appropriate pipeline based on mode
    if use_i2v:
        if _wan_pipeline_i2v is not None and not force_reload:
            logger.info("♻️ Reusing existing WAN I2V pipeline (singleton) - model already loaded")
            return _wan_pipeline_i2v
        pipeline_var = "_wan_pipeline_i2v"
        pipeline_name = "I2V"
    else:
        if _wan_pipeline_t2v is not None and not force_reload:
            logger.info("♻️ Reusing existing WAN T2V pipeline (singleton) - model already loaded")
            return _wan_pipeline_t2v
        pipeline_var = "_wan_pipeline_t2v"
        pipeline_name = "T2V"
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        from diffusers import AutoencoderKLWan, WanPipeline
        # Import WanImageToVideoPipeline for I2V mode
        try:
            from diffusers import WanImageToVideoPipeline
            i2v_available = True
        except ImportError:
            logger.warning("⚠️ WanImageToVideoPipeline not available in this diffusers version. I2V mode will not work.")
            i2v_available = False
            if use_i2v:
                logger.error("❌ Cannot use I2V mode: WanImageToVideoPipeline not available")
                return None
        
        model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
        
        # Configure Hugging Face cache directory to use /workspace if available
        # This is important for RunPod and similar environments with attached disks
        cache_dir = get_cache_dir()
        if cache_dir:
            logger.info(f"📁 Using Hugging Face cache directory: {cache_dir}")
        else:
            # Use default cache
            default_cache = Path.home() / ".cache" / "huggingface"
            logger.info(f"📁 Using default Hugging Face cache: {default_cache}")
        
        logger.info(f"🔄 Loading WAN 2.2 TI2V-5B model: {model_id}")
        logger.info(f"💻 Device: {device}")
        logger.info(f"🎬 Mode: {pipeline_name} ({'Image-to-Video' if use_i2v else 'Text-to-Video'})")
        logger.info("📝 Note: Wan2.2-TI2V-5B is a dense model (no MoE expert switching)")
        
        # Determine torch dtype based on device
        if device == 'cuda' and torch.cuda.is_available():
            # Use bfloat16 on CUDA for maximum quality (full BF16 for 48GB+ VRAM)
            torch_dtype = torch.float16
            vae_dtype = torch.float16  # VAE in FP16 for decode VRAM stability
            logger.info("✅ Using full BF16 precision on CUDA for maximum quality (48GB+ VRAM optimized)")
        else:
            torch_dtype = torch.float32
            vae_dtype = torch.float32
            if device == 'cpu':
                logger.warning("⚠️ Running on CPU - this will be very slow. Consider using GPU.")
        
        # Clear GPU cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Load VAE with wan2.2_vae.safetensors (16x16x4 compression ratio)
        # The 16x16x4 compression ratio provides 64x overall compression (4x temporal, 16x spatial)
        # VAE is shared between T2V and I2V pipelines
        if _wan_vae is None:
            logger.info("📦 Loading WAN 2.2 VAE with 16x16x4 compression ratio...")
            logger.info("📦 VAE: wan2.2_vae.safetensors (temporal: 4x, spatial: 16x16 = 64x total)")
            logger.info(f"📦 VAE dtype: FP16 (optimal for decode VRAM stability)")
            _wan_vae = AutoencoderKLWan.from_pretrained(
                model_id,
                subfolder="vae",
                torch_dtype=torch.float16,  # FP16 for decode VRAM stability
                cache_dir=cache_dir
            )
        else:
            logger.info("♻️ Reusing existing VAE (shared between T2V and I2V pipelines)")
        
        # Load appropriate pipeline based on mode
        # If loading I2V and T2V already exists, reuse its components to avoid loading twice
        global _wan_shared_components
        
        if use_i2v and i2v_available:
            # Check if T2V pipeline exists - if so, reuse its components
            if _wan_pipeline_t2v is not None:
                logger.info("♻️ Reusing components from T2V pipeline to avoid loading model weights twice...")
                logger.info("📦 Constructing I2V pipeline from shared components...")
                # Extract shared components from T2V pipeline
                shared_kwargs = {
                    "vae": _wan_vae,
                }
                # Copy shared components (transformer, text encoders, etc.)
                if hasattr(_wan_pipeline_t2v, "transformer"):
                    shared_kwargs["transformer"] = _wan_pipeline_t2v.transformer
                if hasattr(_wan_pipeline_t2v, "text_encoder"):
                    shared_kwargs["text_encoder"] = _wan_pipeline_t2v.text_encoder
                if hasattr(_wan_pipeline_t2v, "text_encoder_2"):
                    shared_kwargs["text_encoder_2"] = _wan_pipeline_t2v.text_encoder_2
                if hasattr(_wan_pipeline_t2v, "tokenizer"):
                    shared_kwargs["tokenizer"] = _wan_pipeline_t2v.tokenizer
                if hasattr(_wan_pipeline_t2v, "tokenizer_2"):
                    shared_kwargs["tokenizer_2"] = _wan_pipeline_t2v.tokenizer_2
                
                # Load I2V pipeline with shared components (only loads I2V-specific parts)
                pipeline = WanImageToVideoPipeline.from_pretrained(
                    model_id,
                    **shared_kwargs,
                    torch_dtype=torch_dtype,
                    cache_dir=cache_dir
                )
                logger.info("✅ I2V pipeline constructed using shared components (no duplicate model loading)")
            else:
                logger.info("📦 Loading WAN 2.2 TI2V-5B I2V pipeline (dense architecture)...")
                pipeline = WanImageToVideoPipeline.from_pretrained(
                    model_id,
                    vae=_wan_vae,
                    torch_dtype=torch_dtype,
                    cache_dir=cache_dir
                )
        else:
            logger.info("📦 Loading WAN 2.2 TI2V-5B T2V pipeline (dense architecture)...")
            pipeline = WanPipeline.from_pretrained(
                model_id,
                vae=_wan_vae,
                torch_dtype=torch_dtype,
                cache_dir=cache_dir
            )
        
        # Move to device
        pipeline = pipeline.to(device)
        
        # Keep VAE on GPU permanently (48GB VRAM target - no CPU↔GPU shuffling)
        if device == 'cuda':
            logger.info("💾 VAE kept on GPU permanently (48GB VRAM optimized)")
        
        # Enable VAE optimizations (slicing and tiling for memory efficiency)
        # Enable directly on VAE object, not pipeline wrapper
        if device == 'cuda':
            try:
                pipeline.vae.enable_slicing()
                logger.info("✅ Enabled VAE slicing (temporal chunking for memory efficiency)")
            except Exception as e:
                logger.warning(f"⚠️ Could not enable VAE slicing: {e}")
            
            try:
                pipeline.vae.enable_tiling()
                logger.info("✅ Enabled VAE tiling (spatial chunking for 720p+ resolution)")
            except Exception as e:
                logger.warning(f"⚠️ Could not enable VAE tiling: {e}")
        
        # Enable memory optimizations if available
        if device == 'cuda':
            try:
                if hasattr(pipeline, 'enable_memory_efficient_attention'):
                    pipeline.enable_memory_efficient_attention()
                    logger.info("✅ Enabled memory efficient attention")
            except Exception as e:
                logger.warning(f"⚠️ Could not enable memory efficient attention: {e}")
            
            try:
                if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
                    pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("✅ Enabled xformers memory efficient attention")
            except Exception as e:
                logger.info("ℹ️ xFormers not available; continuing without it")
        
        # Apply FastWan LoRA if enabled via environment variable
        fastwan_lora = os.getenv("FASTWAN_LORA", "")
        if fastwan_lora:
            fastwan_weight = os.getenv("FASTWAN_WEIGHT", None)
            fastwan_scale = float(os.getenv("FASTWAN_SCALE", "1.0"))
            apply_fastwan_lora(pipeline, fastwan_lora, weight_name=fastwan_weight, scale=fastwan_scale)
        else:
            logger.info("ℹ️ FastWan disabled (set FASTWAN_LORA to enable)")
        
        # Enable TeaCache if enabled via environment variable
        teacache_enabled = os.getenv("TEACACHE", "0").lower() in ("1", "true", "yes")
        if teacache_enabled:
            teacache_threshold = float(os.getenv("TEACACHE_T", "0.12"))
            try_enable_teacache(pipeline, threshold=teacache_threshold)
        else:
            logger.info("ℹ️ TeaCache disabled (set TEACACHE=1 to enable)")
        
        # Store pipeline in appropriate global variable
        if use_i2v:
            globals()[pipeline_var] = pipeline
            _wan_pipeline_i2v = pipeline
        else:
            globals()[pipeline_var] = pipeline
            _wan_pipeline_t2v = pipeline
        
        logger.info(f"✅ WAN 2.2 TI2V-5B {pipeline_name} pipeline loaded successfully")
        logger.info("📦 WAN pipeline initialized ONCE - will be reused for all subsequent generations")
        return pipeline
        
    except ImportError as e:
        logger.error(f"❌ diffusers library not available or WAN not supported: {e}")
        logger.info("💡 Install with: pip install diffusers>=0.34.0 transformers>=4.45.0")
        return None
    except Exception as e:
        logger.error(f"❌ Error loading WAN model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


class WanT2VGenerator:
    """Handles text-to-video and text-image-to-video generation using WAN 2.2 TI2V-5B."""
    
    def __init__(self, 
                 width: int = 1280,
                 height: int = 720,
                 num_frames: int = 25,
                 fps: int = 24,
                 num_inference_steps: int = 30,
                 guidance_scale: float = 6.0,
                 negative_prompt: str = "text, subtitles, watermark, blurry, low quality, cartoon, anime, manga, illustration, painting, drawing, sketch, bad anatomy, distorted, deformed, ugly",
                 device: str = None):
        """
        Initialize the WAN 2.2 TI2V-5B generator.
        
        Args:
            width: Video width (default: 1280 for 720p)
            height: Video height (default: 720 for 720p)
            num_frames: Number of frames to generate (default: 25)
            fps: Frames per second for output video (default: 24)
            num_inference_steps: Number of denoising steps (default: 30)
            guidance_scale: Guidance scale for prompt adherence (default: 6.0)
            negative_prompt: Negative prompt (default excludes cartoon/anime/illustration for realistic videos)
            device: Device to run on ('cuda' or 'cpu'). Auto-detected if None.
        """
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.fps = fps
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.negative_prompt = negative_prompt
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.pipeline = None
        self.pipeline_i2v = None  # Separate pipeline for I2V mode
        
        # Load T2V pipeline by default (singleton, shared across instances)
        self.pipeline = get_wan_pipeline(device=self.device, use_i2v=False)
        if self.pipeline is None:
            logger.warning("⚠️ WAN T2V pipeline not available")
    
    def generate_video(self, 
                      prompt: str, 
                      output_path: str,
                      seed: Optional[int] = None,
                      negative_prompt: Optional[str] = None,
                      duration: Optional[float] = None,
                      num_frames: Optional[int] = None,
                      scene_id: Optional[str] = None,
                      visual_reference: Optional[str] = None,
                      slug: Optional[str] = None,
                      best_frame_filename: Optional[str] = None,
                      image: Optional[Union[str, np.ndarray, torch.Tensor]] = None) -> Union[str, Dict[str, Any]]:
        """
        Generate a video from a text prompt (T2V) or text + image (TI2V).
        
        Args:
            prompt: Text prompt describing the video
            output_path: Path to save the output MP4 file
            seed: Random seed for reproducibility (optional)
            negative_prompt: Override default negative prompt (optional)
            duration: Target duration in seconds (deprecated - use num_frames instead).
            num_frames: Number of frames to generate. If provided, this takes precedence over duration calculation.
                      If None, uses the default num_frames from initialization.
            scene_id: Scene identifier for frame extraction (optional)
            visual_reference: Visual reference description from storyboard (optional)
            slug: Slug for best frame filename (optional, used if best_frame_filename not provided)
            best_frame_filename: Explicit filename for best frame (optional, takes precedence over slug/scene_id)
            image: Optional input image for TI2V mode. Can be:
                   - Path to image file (str)
                   - numpy array (np.ndarray)
                   - torch tensor (torch.Tensor)
                   If None, uses pure T2V mode (text-only)
            
        Returns:
            Dictionary with:
            - video_path: Path to the generated video file
            - best_frame_path: Path to the best frame PNG (if scene_id provided)
            - scene_id: Scene identifier
            - visual_reference: Visual reference from storyboard
            If scene_id not provided, returns just video_path as string (backward compatible)
        """
        if self.pipeline is None:
            raise RuntimeError("WAN pipeline not available. Cannot generate video.")
        
        # Use num_frames from scene/command-line if provided, otherwise use default
        num_frames_to_use = num_frames if num_frames is not None else self.num_frames
        logger.info(f"🎞️ Using {num_frames_to_use} frames @ {self.fps}fps (~{num_frames_to_use/self.fps:.2f}s)")
        
        try:
            # Determine mode: T2V (text-only) or TI2V (text + image)
            mode = "I2V" if image is not None else "T2V"
            logger.info(f"🎬 Generating video with WAN 2.2 TI2V-5B ({mode} mode)...")
            logger.info(f"📝 Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
            logger.info(f"📐 Dimensions: {self.width}x{self.height} (720p)")
            logger.info(f"🎞️ FPS: {self.fps} (24fps configured)")
            
            # Use provided negative prompt or default
            neg_prompt = negative_prompt or self.negative_prompt
            if neg_prompt:
                logger.info(f"🚫 Negative prompt: {neg_prompt[:100]}{'...' if len(neg_prompt) > 100 else ''}")
            
            # Process image input for I2V mode
            image_input = None
            use_i2v_mode = False
            if image is not None:
                logger.info("🖼️ Processing input image for I2V mode...")
                
                if isinstance(image, str):
                    # Load from file path
                    image_input = Image.open(image).convert("RGB")
                    logger.info(f"📷 Loaded image from: {image}")
                elif isinstance(image, np.ndarray):
                    # Convert numpy array to PIL Image
                    image_input = Image.fromarray(image)
                    logger.info("📷 Converted numpy array to PIL Image")
                elif isinstance(image, torch.Tensor):
                    # Convert torch tensor to PIL Image
                    # Assuming tensor is in [C, H, W] format and normalized [0, 1]
                    if image.dim() == 3:
                        image_np = image.cpu().numpy().transpose(1, 2, 0)
                        if image_np.max() <= 1.0:
                            image_np = (image_np * 255).astype(np.uint8)
                        image_input = Image.fromarray(image_np)
                        logger.info("📷 Converted torch tensor to PIL Image")
                    else:
                        logger.warning("⚠️ Unsupported tensor format, skipping image input")
                else:
                    logger.warning(f"⚠️ Unsupported image type: {type(image)}, skipping image input")
                
                if image_input is not None:
                    use_i2v_mode = True
                    # Resize image to match pipeline dimensions (1280x704 or 704x1280)
                    image_input = image_input.resize((self.width, self.height), Image.Resampling.LANCZOS)
                    logger.info(f"📐 Resized image to {self.width}x{self.height} for 5B model compatibility")
            
            # Load I2V pipeline if image is provided, otherwise use T2V pipeline
            if use_i2v_mode:
                if self.pipeline_i2v is None:
                    logger.info("🔄 Loading I2V pipeline for image-to-video generation...")
                    self.pipeline_i2v = get_wan_pipeline(device=self.device, use_i2v=True)
                    if self.pipeline_i2v is None:
                        logger.warning("⚠️ I2V pipeline not available, falling back to T2V mode")
                        logger.warning("⚠️ Creating dummy black frame for T2V mode...")
                        use_i2v_mode = False
                        # Create a dummy black frame for T2V fallback
                        image_input = Image.new("RGB", (self.width, self.height), color=(0, 0, 0))
                        logger.info("✅ Using T2V mode with dummy black frame (I2V not available)")
                current_pipeline = self.pipeline_i2v
            else:
                current_pipeline = self.pipeline
                if image_input is None:
                    logger.info("✅ Using T2V mode: prompt only (no image provided)")
                else:
                    logger.info("✅ Using T2V mode with dummy frame (I2V pipeline not available)")
            
            # Set seed if provided
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                logger.info(f"🎲 Using seed: {seed}")
            
            # Check if FastWan is enabled and clamp steps if needed
            fastwan_lora = os.getenv("FASTWAN_LORA", "")
            num_inference_steps = self.num_inference_steps
            if fastwan_lora:
                fastwan_steps = os.getenv("FASTWAN_STEPS", "")
                if fastwan_steps:
                    try:
                        fastwan_steps = int(fastwan_steps)
                        if num_inference_steps > fastwan_steps:
                            logger.info(f"⚡ FastWan enabled: clamping steps from {num_inference_steps} to {fastwan_steps}")
                            num_inference_steps = fastwan_steps
                    except ValueError:
                        logger.warning(f"⚠️ Invalid FASTWAN_STEPS value: {fastwan_steps}, using default")
            
            # Generate video (T2V or I2V mode)
            # Optimized for 5B model resolution (1280x704 or 704x1280) @ 24fps with 16x16x4 VAE compression
            logger.info(f"🎬 Running inference ({'I2V' if use_i2v_mode else 'T2V'} mode)...")
            logger.info(f"📊 Sampling config: {num_frames_to_use} frames @ {self.fps}fps, {self.width}x{self.height}px")
            logger.info(f"⚙️ Steps: {num_inference_steps}, Guidance: {self.guidance_scale}")
            pipeline_kwargs = {
                "prompt": prompt,
                "negative_prompt": neg_prompt,
                "height": self.height,  # 5B model optimized (704 or 1280)
                "width": self.width,   # 5B model optimized (1280 or 704)
                "num_frames": num_frames_to_use,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": self.guidance_scale
            }
            
            # Add image input for I2V mode (WanImageToVideoPipeline accepts 'image' parameter)
            if use_i2v_mode and image_input is not None:
                pipeline_kwargs["image"] = image_input
                logger.info("✅ Using I2V mode: prompt + image")
            elif image_input is not None and not use_i2v_mode:
                # T2V mode with dummy frame (shouldn't happen, but handle gracefully)
                logger.warning("⚠️ Image provided but I2V pipeline not available, using T2V mode")
            
            # Call the appropriate pipeline
            output = current_pipeline(**pipeline_kwargs)
            
            # Extract frames
            frames = output.frames[0]
            
            # Fast export using FFmpeg (NVENC GPU or CPU fallback)
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert frames to numpy if needed
            if isinstance(frames, torch.Tensor):
                frames_np = frames.cpu().numpy()
            elif isinstance(frames, list):
                frames_np = np.array([np.array(f) for f in frames])
            else:
                frames_np = np.array(frames)
            
            # Ensure frames are in RGB format (0-255 uint8)
            if frames_np.dtype != np.uint8:
                if frames_np.max() <= 1.0:
                    frames_np = (frames_np * 255).astype(np.uint8)
                else:
                    frames_np = frames_np.astype(np.uint8)
            
            # Ensure frames are in [T, H, W, 3] format for pipe encoding
            if frames_np.ndim == 4:
                # Already in correct format [T, H, W, 3]
                pass
            elif frames_np.ndim == 3:
                # Single frame or wrong format - reshape if needed
                # This shouldn't happen, but handle gracefully
                if frames_np.shape[2] == 3:
                    # [H, W, 3] - add time dimension
                    frames_np = frames_np[np.newaxis, ...]
                else:
                    raise ValueError(f"Unexpected frame shape: {frames_np.shape}")
            else:
                raise ValueError(f"Unexpected frame dimensions: {frames_np.ndim}, shape: {frames_np.shape}")
            
            num_frames = frames_np.shape[0]
            logger.info(f"💾 Exporting {num_frames} frames to: {output_path}")
            
            # Try fast pipe-based encoding first (no disk I/O)
            try:
                encode_video_ffmpeg_pipe(frames_np, self.fps, str(output_path))
                logger.info("🚀 Video exported using fast pipe method (no disk I/O)")
            except Exception as e:
                # Fallback to disk-based method if pipe fails
                logger.warning(f"⚠️ Pipe encoding failed: {e}")
                logger.info("🔄 Falling back to disk-based encoding...")
                import shutil
                temp_frames_dir = tempfile.mkdtemp(prefix="wan_frames_")
                try:
                    write_frames_fast(frames_np, temp_frames_dir)
                    encode_video_ffmpeg(temp_frames_dir, self.fps, str(output_path))
                finally:
                    shutil.rmtree(temp_frames_dir, ignore_errors=True)
            
            # Explicitly delete frames and output to free memory
            del frames
            del output
            frames = None
            output = None
            
            logger.info(f"✅ Video generated successfully: {output_path}")
            
            # Optional FPS interpolation (if enabled via environment variable)
            if INTERPOLATE_FPS:
                logger.info("🎞️ FPS interpolation enabled via env")
                
                # Skip interpolation if already at or above target FPS
                if self.fps >= INTERPOLATE_TARGET_FPS:
                    logger.info(f"ℹ️ Skipping interpolation (already at target fps: {self.fps} >= {INTERPOLATE_TARGET_FPS})")
                else:
                    try:
                        # Generate temporary interpolated file
                        output_path_obj = Path(output_path)
                        interp_path = output_path_obj.parent / f"{output_path_obj.stem}_interp{output_path_obj.suffix}"
                        
                        logger.info(f"🎞️ Interpolating from {self.fps}fps to {INTERPOLATE_TARGET_FPS}fps...")
                        interpolate_video_ffmpeg(str(output_path), str(interp_path), target_fps=INTERPOLATE_TARGET_FPS)
                        
                        # Replace original with interpolated file
                        import shutil
                        shutil.move(str(interp_path), str(output_path))
                        logger.info("✅ FPS interpolation complete")
                    except Exception as e:
                        logger.warning(f"⚠️ FPS interpolation failed, keeping original video: {e}")
                        # Original video remains unchanged
            
            # Extract best frame if scene_id is provided
            result = {"video_path": str(output_path)}
            if scene_id is not None:
                try:
                    best_frame_path = self._extract_best_frame(
                        video_path=str(output_path),
                        scene_id=scene_id,
                        slug=slug,
                        best_frame_filename=best_frame_filename,
                        output_dir=output_path.parent
                    )
                    result.update({
                        "best_frame_path": best_frame_path,
                        "scene_id": scene_id,
                        "visual_reference": visual_reference
                    })
                    logger.info(f"✅ Best frame extracted: {best_frame_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to extract best frame: {e}")
                    # Continue without best frame - video generation succeeded
            
            # Return dict if scene_id provided, otherwise string for backward compatibility
            return result if scene_id is not None else str(output_path)
            
        except Exception as e:
            logger.error(f"❌ Error generating video: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _extract_best_frame(self, 
                           video_path: str,
                           scene_id: str,
                           slug: Optional[str] = None,
                           best_frame_filename: Optional[str] = None,
                           output_dir: Optional[Path] = None,
                           extraction_fps: float = 9.0) -> str:
        """
        Extract frames from video and select the best frame using Laplacian variance.
        
        Args:
            video_path: Path to the video file
            scene_id: Scene identifier
            slug: Slug for filename (optional, used if best_frame_filename not provided)
            best_frame_filename: Explicit filename for best frame (optional, takes precedence)
            output_dir: Directory to save the best frame (optional, uses video parent dir)
            extraction_fps: FPS for frame extraction (6-12 fps range, default 9)
            
        Returns:
            Path to the saved best frame PNG
        """
        video_path_obj = Path(video_path)
        if output_dir is None:
            output_dir = video_path_obj.parent
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create temporary directory for extracted frames
        temp_frames_dir = tempfile.mkdtemp(prefix="wan_frames_")
        temp_frames_dir_path = Path(temp_frames_dir)
        
        try:
            # Extract frames using ffmpeg at specified FPS
            logger.info(f"📸 Extracting frames from video at {extraction_fps} fps...")
            frame_pattern = str(temp_frames_dir_path / "frame_%04d.png")
            
            cmd = [
                'ffmpeg', '-y',
                '-i', str(video_path),
                '-vf', f'fps={extraction_fps}',
                frame_pattern
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Find all extracted frames
            frame_files = sorted(temp_frames_dir_path.glob("frame_*.png"))
            if not frame_files:
                raise ValueError("No frames extracted from video")
            
            logger.info(f"📸 Extracted {len(frame_files)} frames")
            
            # Calculate Laplacian variance for each frame (sharpness metric)
            best_frame_path = None
            best_sharpness = -1
            
            for frame_file in frame_files:
                try:
                    # Read frame as grayscale
                    img = cv2.imread(str(frame_file), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    
                    # Calculate Laplacian variance (sharpness metric)
                    laplacian = cv2.Laplacian(img, cv2.CV_64F)
                    sharpness = laplacian.var()
                    
                    if sharpness > best_sharpness:
                        best_sharpness = sharpness
                        best_frame_path = frame_file
                except Exception as e:
                    logger.warning(f"⚠️ Error processing frame {frame_file}: {e}")
                    continue
            
            if best_frame_path is None:
                raise ValueError("Could not find a valid frame")
            
            logger.info(f"✅ Best frame selected (sharpness: {best_sharpness:.2f})")
            
            # Use best_frame_filename from scene if provided, otherwise generate from slug/scene_id
            if best_frame_filename:
                output_filename = best_frame_filename
            elif slug:
                output_filename = f"{slug}__{scene_id}.png"
            else:
                output_filename = f"{scene_id}.png"
            
            output_frame_path = output_dir / output_filename
            
            # Copy best frame to output location
            import shutil
            shutil.copy2(best_frame_path, output_frame_path)
            logger.info(f"💾 Saved best frame: {output_frame_path}")
            
            return str(output_frame_path)
            
        finally:
            # Clean up temporary frames directory
            try:
                import shutil
                shutil.rmtree(temp_frames_dir, ignore_errors=True)
                logger.debug(f"🧹 Cleaned up temporary frames directory: {temp_frames_dir}")
            except Exception as e:
                logger.warning(f"⚠️ Could not clean up temp directory: {e}")
    
    def is_available(self) -> bool:
        """Check if WAN pipeline is available."""
        return self.pipeline is not None


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test WAN generator
    generator = WanT2VGenerator(
        width=832,
        height=480,
        num_frames=49,
        fps=12
    )
    
    if generator.is_available():
        print("✅ WAN generator initialized successfully")
        
        # Test generation
        test_prompt = "A cat walks on the grass, realistic"
        output_path = generator.generate_video(
            prompt=test_prompt,
            output_path="test_wan_output.mp4"
        )
        print(f"✅ Test video generated: {output_path}")
    else:
        print("❌ WAN generator not available")
