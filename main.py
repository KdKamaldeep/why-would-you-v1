#!/usr/bin/env python3
"""
Cartoon Shorts Generator - Main Entry Point
Professional AI-powered cartoon video generation system
"""

import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Main entry point for the cartoon generator."""
    print("🎬 Cartoon Shorts Generator - Professional Edition")
    print("=" * 50)
    
    try:
        # Import the main generator
        from core.generate_cartoon_short import CartoonShortsGenerator, VideoConfig
        
        # Check if command line arguments are provided
        if len(sys.argv) > 1:
            prompt = " ".join(sys.argv[1:])
            print(f"🎯 Generating cartoon for: {prompt}")
            
            # Create configuration
            config = VideoConfig(
                prompt=prompt,
                duration=30,
                output_path="output"
            )
            
            # Generate cartoon
            generator = CartoonShortsGenerator(config)
            output_path = generator.generate()
            
            print(f"✅ Cartoon generated successfully: {output_path}")
            
        else:
            print("📝 Usage: python main.py 'Your cartoon prompt here'")
            print("💡 Example: python main.py 'A dragon learns to bake cookies'")
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
