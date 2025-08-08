#!/usr/bin/env python3
"""
Simple Cartoon Generator - Easy-to-use script for generating cartoon videos
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_requirements():
    """Check if all required components are installed."""
    print("🔍 Checking requirements...")
    
    # Check if .env file exists
    if not Path(".env").exists():
        print("❌ .env file not found. Please copy config.env to .env and add your API keys.")
        return False
    
    # Check API keys
    openai_key = os.getenv('OPENAI_API_KEY')
    elevenlabs_key = os.getenv('ELEVENLABS_API_KEY')
    
    if not openai_key or openai_key == 'your_openai_api_key_here':
        print("❌ OpenAI API key not configured in .env file")
        return False
        
    if not elevenlabs_key or elevenlabs_key == 'your_elevenlabs_api_key_here':
        print("❌ ElevenLabs API key not configured in .env file")
        return False
    
    # Check if models directory exists
    if not Path("models").exists():
        print("❌ Models directory not found. Please run download_models.bat first.")
        return False
    
    # Check for essential model files
    essential_models = [
        "models/toonyou_beta6.safetensors",
        "models/meina_mix.safetensors",
        "models/mm_sd_v15_v2.safetensors"
    ]
    
    for model in essential_models:
        if not Path(model).exists():
            print(f"❌ Model not found: {model}")
            print("Please run download_models.bat to download required models.")
            return False
    
    print("✅ All requirements satisfied!")
    return True

def generate_cartoon(prompt, style="cartoon", duration=30):
    """Generate a cartoon video with the given prompt."""
    try:
        # Import the main generator
        from generate_cartoon_short import CartoonShortsGenerator, VideoConfig
        
        print(f"🎬 Starting cartoon generation...")
        print(f"📝 Prompt: {prompt}")
        print(f"🎨 Style: {style}")
        print(f"⏱️ Duration: {duration} seconds")
        print("-" * 50)
        
        # Create video configuration
        config = VideoConfig(
            prompt=prompt,
            duration=duration,
            style=style,
            output_path="output"
        )
        
        # Initialize generator
        generator = CartoonShortsGenerator(config)
        
        # Generate the video
        output_path = generator.generate()
        
        print("-" * 50)
        print(f"🎉 Video generation completed!")
        print(f"📁 Output file: {output_path}")
        print(f"🎬 You can now view your cartoon video!")
        
        return output_path
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required Python packages are installed.")
        print("Run: pip install -r requirements.txt")
        return None
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        return None

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate cartoon videos with AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python simple_cartoon_generator.py --prompt "A baby lion opens a smoothie shop"
  python simple_cartoon_generator.py --prompt "A robot learns to dance" --style anime
  python simple_cartoon_generator.py --prompt "Magic forest adventure" --duration 45
        """
    )
    
    parser.add_argument(
        "--prompt", "-p",
        required=True,
        help="The story prompt for your cartoon video"
    )
    
    parser.add_argument(
        "--style", "-s",
        choices=["cartoon", "anime"],
        default="cartoon",
        help="Visual style for the cartoon (default: cartoon)"
    )
    
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=30,
        help="Video duration in seconds (default: 30)"
    )
    
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check requirements, don't generate video"
    )
    
    args = parser.parse_args()
    
    print("🎨 Simple Cartoon Generator")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Setup incomplete. Please fix the issues above and try again.")
        sys.exit(1)
    
    if args.check_only:
        print("\n✅ All requirements are satisfied! You're ready to generate cartoons.")
        sys.exit(0)
    
    # Generate cartoon
    output_path = generate_cartoon(args.prompt, args.style, args.duration)
    
    if output_path:
        print(f"\n🎊 Success! Your cartoon is ready at: {output_path}")
    else:
        print("\n❌ Generation failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
