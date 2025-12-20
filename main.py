#!/usr/bin/env python3
"""
Cartoon Shorts Generator - Main Entry Point
Professional AI-powered cartoon video generation system
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Main entry point for the cartoon generator."""
    print("🎬 Video Reel Generator - Professional Edition")
    print("=" * 50)
    
    try:
        # Import the main generator
        from core.generate_cartoon_short import CartoonShortsGenerator, VideoConfig
        
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Generate platform-ready vertical Reels/Shorts videos")
        parser.add_argument("prompt", nargs="?", help="Story prompt (e.g., 'A dragon learns to bake cookies')")
        parser.add_argument("--video-format", choices=["shorts", "normal"], default="shorts",
                           help="Video format: 'shorts' for 9:16 YouTube Shorts, 'normal' for 16:9 standard videos")
        parser.add_argument("--duration", type=int, default=30, help="Video duration in seconds")
        parser.add_argument("--output", default="output", help="Output directory")
        
        args = parser.parse_args()
        
        # Check if prompt is provided
        if args.prompt:
            print(f"🎯 Generating video reel for: {args.prompt}")
            print(f"📐 Video format: {args.video_format}")
            print(f"⏱️ Duration: {args.duration} seconds")
            
            # Create configuration
            config = VideoConfig(
                prompt=args.prompt,
                duration=args.duration,
                video_format=args.video_format,
                output_path=args.output
            )
            
            # Generate cartoon
            generator = CartoonShortsGenerator(config)
            output_path = generator.generate()
            
            print(f"✅ Video reel generated successfully: {output_path}")
            
        else:
            print("📝 Usage: python main.py 'Your story prompt here' [options]")
            print("💡 Example: python main.py 'A dragon learns to bake cookies'")
            print("💡 Example: python main.py 'A dragon learns to bake cookies' --video-format normal")
            print("\n🎨 Available options:")
            print("   --video-format: shorts (9:16) or normal (16:9)")
            print("   --duration: video duration in seconds")
            print("   --output: output directory")
            print("\n🎨 For more options, use the interface scripts:")
            print("   python -m src.interfaces.simple_cartoon_generator")
            print("   python -m src.interfaces.quick_start")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Please install dependencies: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
