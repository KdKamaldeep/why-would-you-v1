#!/usr/bin/env python3
"""
WAN 2.2 Text-Image-to-Video Generator Module
Handles text-to-video generation using Wan-AI/Wan2.2-TI2V-5B
Supports both text-to-video (T2V) and text-image-to-video (TI2V) modes
"""

import os
# Allocator env var handling (new PyTorch naming)
if not os.getenv("PYTORCH_ALLOC_CONF") and os.getenv("PYTORCH_CUDA_ALLOC_CONF"):
    os.environ["PYTORCH_ALLOC_CONF"] = os.environ["PYTORCH_CUDA_ALLOC_CONF"]
if not os.getenv("PYTORCH_ALLOC_CONF"):
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
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

# Model IDs
WAN_MODEL_ID_5B = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
WAN_MODEL_ID_14B = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"

# Global singleton instance
_wan_pipeline = None
_wan_vae = None


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


def get_wan_pipeline(device: str = None, force_reload: bool = False, model_size: str = "5b"):
    """
    Get or initialize the global WAN pipeline (singleton pattern).
    
    Args:
        device: Device to run on ('cuda' or 'cpu'). Auto-detected if None.
        force_reload: Force reload of the pipeline even if already loaded.
        model_size: Model size to use ("5b" or "14b", default "5b").
        
    Returns:
        WanPipeline instance or None if loading fails
    """
    global _wan_pipeline, _wan_vae
    
    if _wan_pipeline is not None and not force_reload:
        logger.info("♻️ Reusing existing WAN pipeline (singleton) - model already loaded")
        return _wan_pipeline
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Determine model ID based on model_size
    model_size_lower = model_size.lower() if model_size else "5b"
    if model_size_lower == "14b":
        model_id = WAN_MODEL_ID_14B
        is_moe = True
    else:
        model_id = WAN_MODEL_ID_5B
        is_moe = False
    
    # Check GPU memory for 14B model
    if is_moe and device == 'cuda' and torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory_gb < 40:
            logger.warning(f"⚠️ WARNING: Selected 14B MoE model but GPU memory is {gpu_memory_gb:.1f}GB (< 40GB). This may cause OOM errors.")
    
    try:
        from diffusers import AutoencoderKLWan, WanPipeline
        
        # Configure Hugging Face cache directory to use /workspace if available
        # This is important for RunPod and similar environments with attached disks
        cache_dir = get_cache_dir()
        if cache_dir:
            logger.info(f"📁 Using Hugging Face cache directory: {cache_dir}")
        else:
            # Use default cache
            default_cache = Path.home() / ".cache" / "huggingface"
            logger.info(f"📁 Using default Hugging Face cache: {default_cache}")
        
        logger.info(f"🔄 Loading WAN 2.2 TI2V model: {model_id}")
        logger.info(f"💻 Device: {device}")
        logger.info(f"📝 Model type: {'MoE (14B)' if is_moe else 'Dense (5B)'}")
        
        # Determine torch dtype based on device
        if device == 'cuda' and torch.cuda.is_available():
            # Use bfloat16 on CUDA for maximum quality (full BF16 for 48GB+ VRAM)
            torch_dtype = torch.bfloat16
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
        logger.info("📦 Loading WAN 2.2 VAE with 16x16x4 compression ratio...")
        logger.info("📦 VAE: wan2.2_vae.safetensors (temporal: 4x, spatial: 16x16 = 64x total)")
        logger.info(f"📦 VAE dtype: FP16 (optimal for decode VRAM stability)")
        _wan_vae = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=torch.float16,  # FP16 for decode VRAM stability
            cache_dir=cache_dir
        )
        
        # Load pipeline (MoE-safe: no device_map="auto", no low_cpu_mem_usage=False)
        logger.info(f"📦 Loading WAN 2.2 TI2V pipeline ({'MoE' if is_moe else 'Dense'} architecture)...")
        _wan_pipeline = WanPipeline.from_pretrained(
            model_id,
            vae=_wan_vae,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir
        )
        
        # Enable attention slicing if available
        if hasattr(_wan_pipeline, 'enable_attention_slicing'):
            try:
                _wan_pipeline.enable_attention_slicing("max")
                logger.info("✅ Enabled attention slicing (max)")
            except Exception as e:
                logger.warning(f"⚠️ Could not enable attention slicing: {e}")
        
        # Move to device
        _wan_pipeline = _wan_pipeline.to(device)
        
        # Offload VAE to CPU when idle to reduce peak VRAM overlap
        if device == 'cuda':
            _wan_pipeline.vae.to("cpu")
            logger.info("💾 VAE offloaded to CPU (will be moved to GPU only during inference)")
        
        # Enable VAE optimizations (slicing and tiling for memory efficiency) - mandatory
        # Enable directly on VAE object, not pipeline wrapper
        if device == 'cuda':
            if hasattr(_wan_pipeline.vae, 'enable_slicing'):
                try:
                    _wan_pipeline.vae.enable_slicing()
                    logger.info("✅ Enabled VAE slicing (temporal chunking for memory efficiency)")
                except Exception as e:
                    logger.warning(f"⚠️ Could not enable VAE slicing: {e}")
            
            if hasattr(_wan_pipeline.vae, 'enable_tiling'):
                try:
                    _wan_pipeline.vae.enable_tiling()
                    logger.info("✅ Enabled VAE tiling (spatial chunking for 720p+ resolution)")
                except Exception as e:
                    logger.warning(f"⚠️ Could not enable VAE tiling: {e}")
        
        # Enable memory optimizations if available
        if device == 'cuda':
            try:
                if hasattr(_wan_pipeline, 'enable_memory_efficient_attention'):
                    _wan_pipeline.enable_memory_efficient_attention()
                    logger.info("✅ Enabled memory efficient attention")
            except Exception as e:
                logger.warning(f"⚠️ Could not enable memory efficient attention: {e}")
            
            try:
                if hasattr(_wan_pipeline, 'enable_xformers_memory_efficient_attention'):
                    _wan_pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("✅ Enabled xformers memory efficient attention")
            except Exception as e:
                logger.info("ℹ️ xFormers not available; continuing without it")
        
        logger.info(f"✅ WAN 2.2 TI2V pipeline loaded successfully ({'MoE' if is_moe else 'Dense'})")
        logger.info("📦 WAN pipeline initialized ONCE - will be reused for all subsequent generations")
        return _wan_pipeline
        
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
                 device: str = None,
                 model_size: str = None,
                 vae_tiling: bool = True,
                 vae_slicing: bool = True,
                 compile_unet: bool = False,
                 offload_text_encoders: bool = None):
        """
        Initialize the WAN 2.2 TI2V generator.
        
        Args:
            width: Video width (default: 1280 for 720p)
            height: Video height (default: 720 for 720p)
            num_frames: Number of frames to generate (default: 25)
            fps: Frames per second for output video (default: 24)
            num_inference_steps: Number of denoising steps (default: 30)
            guidance_scale: Guidance scale for prompt adherence (default: 6.0)
            negative_prompt: Negative prompt (default excludes cartoon/anime/illustration for realistic videos)
            device: Device to run on ('cuda' or 'cpu'). Auto-detected if None.
            model_size: Model size ("5b" or "14b", default from env WAN_MODEL_SIZE or "5b").
            vae_tiling: Enable VAE tiling (default: True)
            vae_slicing: Enable VAE slicing (default: True)
            compile_unet: Compile UNet with torch.compile (default: False)
            offload_text_encoders: Offload text encoders to CPU after embeddings (default: from env OFFLOAD_TEXT_ENCODERS or True)
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
        
        # Determine model_size from arg, env var, or default
        if model_size is None:
            model_size = os.getenv("WAN_MODEL_SIZE", "5b").lower()
        self.model_size = model_size.lower()
        
        # Determine offload_text_encoders from arg, env var, or default
        if offload_text_encoders is None:
            offload_env = os.getenv("OFFLOAD_TEXT_ENCODERS", "true").lower()
            offload_text_encoders = offload_env in ("true", "1", "yes")
        self.offload_text_encoders = offload_text_encoders
        
        self.vae_tiling = vae_tiling
        self.vae_slicing = vae_slicing
        self.compile_unet = compile_unet
        
        # Load pipeline (singleton, shared across instances)
        self.pipeline = get_wan_pipeline(device=self.device, model_size=self.model_size)
        if self.pipeline is None:
            logger.warning("⚠️ WAN pipeline not available")
        
        logger.info(f"📦 Model size: {self.model_size.upper()}")
        logger.info(f"💾 Text encoder offload: {'enabled' if self.offload_text_encoders else 'disabled'}")
    
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
            mode = "TI2V" if image is not None else "T2V"
            logger.info(f"🎬 Generating video with WAN 2.2 TI2V ({mode} mode, {self.model_size.upper()} model)...")
            logger.info(f"📝 Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
            logger.info(f"📐 Dimensions: {self.width}x{self.height} (720p)")
            logger.info(f"🎞️ FPS: {self.fps} (24fps configured)")
            logger.info(f"⚙️ Steps: {self.num_inference_steps}, Guidance: {self.guidance_scale}")
            
            # Use provided negative prompt or default
            neg_prompt = negative_prompt or self.negative_prompt
            if neg_prompt:
                logger.info(f"🚫 Negative prompt: {neg_prompt[:100]}{'...' if len(neg_prompt) > 100 else ''}")
            
            # Process image input for TI2V mode
            image_input = None
            if image is not None:
                logger.info("🖼️ Processing input image for TI2V mode...")
                
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
            
            # Set seed if provided
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                logger.info(f"🎲 Using seed: {seed}")
            
            # Clear GPU cache before generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Move VAE to GPU only for inference
            if self.device == 'cuda' and torch.cuda.is_available():
                self.pipeline.vae.to(self.device)
                torch.cuda.empty_cache()
                gc.collect()
                logger.info("🚀 VAE moved to GPU for inference")
            
            # Prepare pipeline kwargs
            pipeline_kwargs = {
                "prompt": prompt,
                "negative_prompt": neg_prompt,
                "height": self.height,  # 720p optimized
                "width": self.width,   # 720p optimized
                "num_frames": num_frames_to_use,  # Calculated for 24fps
                "num_inference_steps": self.num_inference_steps,
                "guidance_scale": self.guidance_scale
            }
            
            # Add image input for TI2V mode
            if image_input is not None:
                pipeline_kwargs["image"] = image_input
                logger.info("✅ Using TI2V mode: prompt + image")
            else:
                logger.info("✅ Using T2V mode: prompt only")
            
            # Determine torch dtype for inference
            torch_dtype = torch.bfloat16 if (self.device == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16
            
            # Generate video with VRAM discipline
            # Optimized for 720p (1280x720) @ 24fps with 16x16x4 VAE compression
            logger.info("🎬 Running inference (optimized for 720p @ 24fps)...")
            logger.info(f"📊 Sampling config: {num_frames_to_use} frames @ {self.fps}fps, {self.width}x{self.height}px")
            
            # Text encoder offload: encode prompts first, then move to CPU
            if self.offload_text_encoders and self.device == 'cuda' and torch.cuda.is_available():
                # Try to encode prompts separately if pipeline supports it
                # Otherwise, we'll offload after the first forward pass
                logger.info("💾 Text encoder offload enabled - will move to CPU after encoding")
            
            # Run inference with autocast and inference_mode
            with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch_dtype, enabled=(self.device == 'cuda')):
                output = self.pipeline(**pipeline_kwargs)
            
            # Offload text encoders to CPU after embeddings are ready (after inference)
            if self.offload_text_encoders and self.device == 'cuda' and torch.cuda.is_available():
                if hasattr(self.pipeline, "text_encoder") and self.pipeline.text_encoder is not None:
                    self.pipeline.text_encoder.to("cpu")
                if hasattr(self.pipeline, "text_encoder_2") and self.pipeline.text_encoder_2 is not None:
                    self.pipeline.text_encoder_2.to("cpu")
                torch.cuda.empty_cache()
                gc.collect()
                logger.info("💾 Text encoders moved to CPU (VRAM freed)")
            
            # Extract frames
            frames = output.frames[0]
            
            # Export to video with slicing/chunking for memory efficiency
            from diffusers.utils import export_to_video
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Enable slicing for large videos (720p @ 24fps can be memory-intensive)
            # Process frames in chunks to avoid OOM errors
            num_frames = len(frames)
            chunk_size = 100  # Process 100 frames at a time (adjust based on VRAM)
            
            logger.info(f"💾 Saving video to: {output_path}")
            logger.info(f"📊 Total frames: {num_frames}, Chunk size: {chunk_size}")
            
            if num_frames > chunk_size:
                # Use chunked export for large videos
                logger.info(f"🔪 Using chunked export (slicing enabled) for {num_frames} frames")
                # Convert frames to numpy if needed and process in chunks
                
                # Ensure frames are in the right format
                if isinstance(frames, torch.Tensor):
                    frames_np = frames.cpu().numpy()
                elif isinstance(frames, list):
                    frames_np = np.array([np.array(f) for f in frames])
                else:
                    frames_np = np.array(frames)
                
                # Process in chunks to save memory
                temp_dir = tempfile.mkdtemp(prefix="wan_export_")
                chunk_files = []
                
                try:
                    for i in range(0, num_frames, chunk_size):
                        chunk_end = min(i + chunk_size, num_frames)
                        chunk = frames_np[i:chunk_end]
                        chunk_file = os.path.join(temp_dir, f"chunk_{i:04d}.mp4")
                        chunk_files.append(chunk_file)
                        
                        logger.info(f"📦 Processing chunk {i//chunk_size + 1}/{(num_frames-1)//chunk_size + 1}: frames {i}-{chunk_end-1}")
                        export_to_video(chunk, chunk_file, fps=self.fps)
                    
                    # Concatenate chunks using FFmpeg
                    logger.info("🔗 Concatenating video chunks...")
                    concat_file = os.path.join(temp_dir, "concat_list.txt")
                    with open(concat_file, 'w') as f:
                        for chunk_file in chunk_files:
                            f.write(f"file '{os.path.abspath(chunk_file)}'\n")
                    
                    cmd = [
                        'ffmpeg', '-y',
                        '-f', 'concat',
                        '-safe', '0',
                        '-i', concat_file,
                        '-c', 'copy',  # Fast copy without re-encoding
                        str(output_path)
                    ]
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                    
                    # Clean up temp files
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    logger.info("✅ Chunked export completed successfully")
                except Exception as e:
                    # Fallback to direct export if chunking fails
                    logger.warning(f"⚠️ Chunked export failed: {e}, falling back to direct export")
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    export_to_video(frames, str(output_path), fps=self.fps)
            else:
                # Direct export for smaller videos
                logger.info("✅ Using direct export (no slicing needed)")
                export_to_video(frames, str(output_path), fps=self.fps)
            
            # Move VAE back to CPU after saving/export is done
            if self.device == 'cuda' and torch.cuda.is_available():
                self.pipeline.vae.to("cpu")
                torch.cuda.empty_cache()
                gc.collect()
                logger.info("💾 VAE moved back to CPU (idle)")
            
            # Explicitly delete frames and output to free memory
            del frames
            del output
            frames = None
            output = None
            
            # Clear GPU cache after generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for all GPU operations to complete
            gc.collect()  # Force Python garbage collection
            
            logger.info(f"✅ Video generated successfully: {output_path}")
            
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
