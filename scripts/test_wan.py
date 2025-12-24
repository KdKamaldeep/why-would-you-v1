#!/usr/bin/env python3
"""
Test script for WAN 2.1 Text-to-Video generation
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def main():
    """Main function to test WAN T2V generation."""
    parser = argparse.ArgumentParser(description="Test WAN 2.1 Text-to-Video generation")
    parser.add_argument(
        "--prompt",
        type=str,
        default="A cat walks on the grass, realistic",
        help="Text prompt for video generation (default: 'A cat walks on the grass, realistic')"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/test_wan.mp4",
        help="Output video path (default: 'outputs/test_wan.mp4')"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Video width (default: 832)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Video height (default: 480)"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=49,
        help="Number of frames to generate (default: 49)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=12,
        help="Output FPS (default: 12)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Inference steps (default: 30)"
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=6.0,
        help="Guidance scale (default: 6.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (optional)"
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="text, subtitles, watermark, blurry, low quality",
        help="Negative prompt (default: 'text, subtitles, watermark, blurry, low quality')"
    )
    
    args = parser.parse_args()
    
    print("🧪 Testing WAN 2.1 Text-to-Video Generation")
    print("=" * 60)
    print(f"📝 Prompt: {args.prompt}")
    print(f"📐 Dimensions: {args.width}x{args.height}")
    print(f"🎞️ Frames: {args.num_frames} @ {args.fps}fps (~{args.num_frames/args.fps:.1f}s)")
    print(f"⚙️ Steps: {args.steps}, Guidance: {args.guidance}")
    if args.seed:
        print(f"🎲 Seed: {args.seed}")
    print(f"🚫 Negative prompt: {args.negative_prompt}")
    print(f"📁 Output: {args.out}")
    print("=" * 60)
    
    try:
        from core.wan_t2v import WanT2VGenerator
        
        # Create output directory
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize generator
        print("\n🔄 Initializing WAN generator...")
        generator = WanT2VGenerator(
            width=args.width,
            height=args.height,
            num_frames=args.num_frames,
            fps=args.fps,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            negative_prompt=args.negative_prompt
        )
        
        if not generator.is_available():
            print("\n❌ WAN generator is not available!")
            print("💡 Make sure you have installed the required dependencies:")
            print("   pip install diffusers>=0.34.0 transformers>=4.45.0 accelerate>=0.33.0")
            sys.exit(1)
        
        print("✅ WAN generator initialized successfully\n")
        
        # Generate video
        print("🎬 Generating video...")
        output_file = generator.generate_video(
            prompt=args.prompt,
            output_path=args.out,
            seed=args.seed,
            negative_prompt=args.negative_prompt
        )
        
        print("\n" + "=" * 60)
        print("✅ Test completed successfully!")
        print(f"📁 Output video: {output_file}")
        
        # Check file size
        if Path(output_file).exists():
            file_size = Path(output_file).stat().st_size / (1024 * 1024)  # MB
            print(f"📊 File size: {file_size:.2f} MB")
        
        print("=" * 60)
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("💡 Make sure you have installed the required dependencies:")
        print("   pip install diffusers>=0.34.0 transformers>=4.45.0 accelerate>=0.33.0 safetensors>=0.4.0 imageio imageio-ffmpeg")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
