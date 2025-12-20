#!/usr/bin/env python3
"""
Quick Start Script - Generate a video reel with minimal setup
"""

import os
from pathlib import Path

def quick_generate():
    """Quick generation with default settings."""
    
    print("🚀 Quick Video Reel Generator")
    print("=" * 40)
    
    # Get user input
    prompt = input("📝 Enter your story idea: ").strip()
    
    if not prompt:
        print("❌ Please provide a story prompt!")
        return
    
    print(f"\n🎬 Generating video reel for: '{prompt}'")
    print("⏳ This may take several minutes...")
    print("-" * 40)
    
    try:
        # Import and run the generator
        from src.core.generate_cartoon_short import CartoonShortsGenerator, VideoConfig
        
        # Create simple config
        config = VideoConfig(
            prompt=prompt,
            duration=30,
            style="realistic",
            output_path="output"
        )
        
        # Generate
        generator = CartoonShortsGenerator(config)
        output_path = generator.generate()
        
        print("-" * 40)
        print(f"🎉 Done! Video saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Tips:")
        print("1. Make sure you've run: pip install -r requirements.txt")
        print("2. Download models with: download_models.bat")
        print("3. Set up your API keys in .env file")

def main():
    """Entry point wrapper to satisfy package imports."""
    quick_generate()

if __name__ == "__main__":
    quick_generate()
