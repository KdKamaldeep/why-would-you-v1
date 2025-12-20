#!/usr/bin/env python3
"""
WAN 2.1 Text-to-Video Generator Module
Handles text-to-video generation using Wan-AI/Wan2.1-T2V-1.3B-Diffusers
"""

import os
import logging
import torch
from pathlib import Path
from typing import Optional
import gc

logger = logging.getLogger(__name__)

# Global singleton instance
_wan_pipeline = None
_wan_vae = None


def get_wan_pipeline(device: str = None, force_reload: bool = False):
    """
    Get or initialize the global WAN pipeline (singleton pattern).
    
    Args:
        device: Device to run on ('cuda' or 'cpu'). Auto-detected if None.
        force_reload: Force reload of the pipeline even if already loaded.
        
    Returns:
        WanPipeline instance or None if loading fails
    """
    global _wan_pipeline, _wan_vae
    
    if _wan_pipeline is not None and not force_reload:
        return _wan_pipeline
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        from diffusers import AutoencoderKLWan, WanPipeline
        
        model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        
        # Configure Hugging Face cache directory to use /workspace if available
        # This is important for RunPod and similar environments with attached disks
        cache_dir = get_cache_dir()
        if cache_dir:
            logger.info(f"📁 Using Hugging Face cache directory: {cache_dir}")
        else:
            # Use default cache
            default_cache = Path.home() / ".cache" / "huggingface"
            logger.info(f"📁 Using default Hugging Face cache: {default_cache}")
        
        logger.info(f"🔄 Loading WAN 2.1 T2V model: {model_id}")
        logger.info(f"💻 Device: {device}")
        
        # Determine torch dtype based on device
        if device == 'cuda' and torch.cuda.is_available():
            # Use bfloat16 on CUDA for better performance and memory efficiency
            torch_dtype = torch.bfloat16
            vae_dtype = torch.float32  # VAE typically uses float32
            logger.info("✅ Using bfloat16 on CUDA for optimal performance")
        else:
            torch_dtype = torch.float32
            vae_dtype = torch.float32
            if device == 'cpu':
                logger.warning("⚠️ Running on CPU - this will be very slow. Consider using GPU.")
        
        # Clear GPU cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Load VAE
        logger.info("📦 Loading WAN VAE...")
        _wan_vae = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=vae_dtype,
            cache_dir=cache_dir
        )
        
        # Load pipeline
        logger.info("📦 Loading WAN pipeline...")
        _wan_pipeline = WanPipeline.from_pretrained(
            model_id,
            vae=_wan_vae,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir
        )
        
        # Move to device
        _wan_pipeline = _wan_pipeline.to(device)
        
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
        
        logger.info("✅ WAN 2.1 T2V pipeline loaded successfully")
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
    """Handles text-to-video generation using WAN 2.1."""
    
    def __init__(self, 
                 width: int = 832,
                 height: int = 480,
                 num_frames: int = 49,
                 fps: int = 12,
                 num_inference_steps: int = 30,
                 guidance_scale: float = 6.0,
                 negative_prompt: str = "text, subtitles, watermark, blurry, low quality",
                 device: str = None):
        """
        Initialize the WAN T2V generator.
        
        Args:
            width: Video width (default: 832)
            height: Video height (default: 480)
            num_frames: Number of frames to generate (default: 49)
            fps: Frames per second for output video (default: 12)
            num_inference_steps: Number of denoising steps (default: 30)
            guidance_scale: Guidance scale for prompt adherence (default: 6.0)
            negative_prompt: Negative prompt (default: "text, subtitles, watermark, blurry, low quality")
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
        
        # Load pipeline (singleton, shared across instances)
        self.pipeline = get_wan_pipeline(device=self.device)
        if self.pipeline is None:
            logger.warning("⚠️ WAN pipeline not available")
    
    def generate_video(self, 
                      prompt: str, 
                      output_path: str,
                      seed: Optional[int] = None,
                      negative_prompt: Optional[str] = None) -> str:
        """
        Generate a video from a text prompt.
        
        Args:
            prompt: Text prompt describing the video
            output_path: Path to save the output MP4 file
            seed: Random seed for reproducibility (optional)
            negative_prompt: Override default negative prompt (optional)
            
        Returns:
            Path to the generated video file
        """
        if self.pipeline is None:
            raise RuntimeError("WAN pipeline not available. Cannot generate video.")
        
        try:
            logger.info(f"🎬 Generating video with WAN 2.1 T2V...")
            logger.info(f"📝 Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
            logger.info(f"📐 Dimensions: {self.width}x{self.height}")
            logger.info(f"🎞️ Frames: {self.num_frames} @ {self.fps}fps (~{self.num_frames/self.fps:.1f}s)")
            logger.info(f"⚙️ Steps: {self.num_inference_steps}, Guidance: {self.guidance_scale}")
            
            # Use provided negative prompt or default
            neg_prompt = negative_prompt or self.negative_prompt
            if neg_prompt:
                logger.info(f"🚫 Negative prompt: {neg_prompt[:100]}{'...' if len(neg_prompt) > 100 else ''}")
            
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
            
            # Generate video
            logger.info("🎬 Running inference...")
            output = self.pipeline(
                prompt=prompt,
                negative_prompt=neg_prompt,
                height=self.height,
                width=self.width,
                num_frames=self.num_frames,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale
            )
            
            # Extract frames
            frames = output.frames[0]
            
            # Export to video
            from diffusers.utils import export_to_video
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"💾 Saving video to: {output_path}")
            export_to_video(frames, str(output_path), fps=self.fps)
            
            # Clear GPU cache after generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            logger.info(f"✅ Video generated successfully: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"❌ Error generating video: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
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
