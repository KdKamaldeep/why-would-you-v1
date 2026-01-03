#!/usr/bin/env python3
"""
Test script for WAN 2.2 with Gemini image generation
Tests the full pipeline: Gemini generates initial frame -> WAN generates video
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
if not os.getenv("GEMINI_API_KEY") and Path("config.env").exists():
    load_dotenv("config.env")

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging to display in console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main function to test WAN with Gemini image generation."""
    parser = argparse.ArgumentParser(description="Test WAN 2.2 with Gemini image generation")
    parser.add_argument(
        "--visual-prompt",
        type=str,
        default="first person point of view, slowly rising above thick clouds, endless sky opening ahead, soft golden sunlight, fantasy cinematic lighting",
        help="Visual prompt for Gemini image generation"
    )
    parser.add_argument(
        "--motion-prompt",
        type=str,
        default="slow vertical ascent as if floating upward, extremely smooth motion",
        help="Motion prompt for WAN video generation"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/test_wan_gemini.mp4",
        help="Output video path (default: 'outputs/test_wan_gemini.mp4')"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help="Video width (default: 768)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1344,
        help="Video height (default: 1344)"
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
        default=24,
        help="Inference steps (default: 24)"
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
        default="text, subtitles, watermark, blurry, low quality, cartoon, anime, manga, illustration, painting, drawing, sketch, bad anatomy, distorted, deformed, ugly",
        help="Negative prompt"
    )
    parser.add_argument(
        "--skip-gemini",
        action="store_true",
        help="Skip Gemini image generation and use T2V mode only"
    )
    
    args = parser.parse_args()
    
    print("🧪 Testing WAN 2.2 with Gemini Image Generation")
    print("=" * 60)
    print(f"🎨 Visual Prompt (Gemini): {args.visual_prompt}")
    print(f"🎬 Motion Prompt (WAN): {args.motion_prompt}")
    print(f"📐 Dimensions: {args.width}x{args.height}")
    print(f"🎞️ Frames: {args.num_frames} @ {args.fps}fps (~{args.num_frames/args.fps:.1f}s)")
    print(f"⚙️ Steps: {args.steps}, Guidance: {args.guidance}")
    if args.seed:
        print(f"🎲 Seed: {args.seed}")
    print(f"📁 Output: {args.out}")
    print("=" * 60)
    
    try:
        from core.gemini_image_generator import GeminiImageGenerator
        from core.wan_t2v import WanT2VGenerator
        
        # Create output directory
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        initial_image_path = None
        
        # Step 1: Generate initial frame with Gemini
        if not args.skip_gemini:
            print("\n🔄 Step 1: Generating initial frame with Gemini...")
            gemini_generator = GeminiImageGenerator()
            
            if not gemini_generator.available:
                print("⚠️ Gemini not available, skipping image generation")
                print("💡 Make sure GEMINI_API_KEY is set and google-genai is installed")
            else:
                initial_image_path = output_path.parent / "initial_frame.png"
                print(f"📝 Generating image from visual prompt...")
                generated_image = gemini_generator.generate_image(
                    prompt=args.visual_prompt,
                    output_path=str(initial_image_path),
                    width=args.width,
                    height=args.height
                )
                
                if generated_image:
                    initial_image_path = generated_image
                    print(f"✅ Initial frame generated: {initial_image_path}")
                else:
                    print("⚠️ Failed to generate initial frame, will use T2V mode")
                    initial_image_path = None
        else:
            print("\n⏭️ Skipping Gemini image generation (--skip-gemini)")
        
        # Step 2: Generate video with WAN
        print("\n🔄 Step 2: Generating video with WAN...")
        
        # Combine visual_prompt and motion_prompt for WAN
        combined_prompt = f"{args.visual_prompt}. Motion: {args.motion_prompt}"
        
        wan_generator = WanT2VGenerator(
            width=args.width,
            height=args.height,
            num_frames=args.num_frames,
            fps=args.fps,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            negative_prompt=args.negative_prompt
        )
        
        if not wan_generator.is_available():
            print("\n❌ WAN generator is not available!")
            print("💡 Make sure you have installed the required dependencies")
            sys.exit(1)
        
        print("✅ WAN generator initialized successfully")
        
        # Generate video
        print(f"\n🎬 Generating video ({'I2V' if initial_image_path else 'T2V'} mode)...")
        output_file = wan_generator.generate_video(
            prompt=combined_prompt,
            output_path=args.out,
            seed=args.seed,
            negative_prompt=args.negative_prompt,
            image=str(initial_image_path) if initial_image_path else None
        )
        
        print("\n" + "=" * 60)
        print("✅ Test completed successfully!")
        print(f"📁 Output video: {output_file}")
        
        # Check file size
        if Path(output_file).exists():
            file_size = Path(output_file).stat().st_size / (1024 * 1024)  # MB
            print(f"📊 File size: {file_size:.2f} MB")
        
        if initial_image_path and Path(initial_image_path).exists():
            image_size = Path(initial_image_path).stat().st_size / 1024  # KB
            print(f"🖼️ Initial frame: {initial_image_path} ({image_size:.2f} KB)")
        
        print("=" * 60)
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("💡 Make sure you have installed the required dependencies:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

