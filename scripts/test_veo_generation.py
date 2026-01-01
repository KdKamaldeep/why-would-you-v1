#!/usr/bin/env python3
"""
Test script for Google Veo video generation
Tests the Veo generator with a simple prompt
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from src.core.veo_generator import VeoGenerator

# Load environment variables
load_dotenv()

def test_veo_generation():
    """Test Veo video generation with a simple prompt."""
    
    print("=" * 60)
    print("🧪 Testing Google Veo Video Generation")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY not found in environment variables")
        print("💡 Set it in your .env file or export it:")
        print("   export GOOGLE_API_KEY=your_api_key_here")
        return False
    
    print(f"✅ API Key found: {api_key[:10]}...{api_key[-4:]}")
    
    # Test parameters
    test_prompt = "A cat walks on the grass, realistic, cinematic"
    output_path = "output/test_veo_video.mp4"
    
    print(f"\n📝 Test Prompt: {test_prompt}")
    print(f"📁 Output Path: {output_path}")
    print(f"📐 Dimensions: 768x1024 (9:16 aspect ratio)")
    print(f"⏱️  Duration: 5 seconds (Veo will use default)")
    print("\n" + "-" * 60)
    
    try:
        # Initialize Veo generator
        print("🔧 Initializing Veo generator...")
        generator = VeoGenerator(
            api_key=api_key,
            width=768,
            height=1024,
            duration=5,
            negative_prompt="text, subtitles, watermark, blurry, low quality"
        )
        
        if not generator.is_available():
            print("❌ Veo generator is not available")
            return False
        
        print("✅ Veo generator initialized successfully")
        print("\n" + "-" * 60)
        
        # Generate video
        print("🎬 Starting video generation...")
        print("⏳ This may take several minutes...")
        print("-" * 60)
        
        result = generator.generate_video(
            prompt=test_prompt,
            output_path=output_path
        )
        
        print("\n" + "=" * 60)
        print("✅ Video generation completed successfully!")
        print(f"📁 Video saved to: {result}")
        print("=" * 60)
        
        # Check if file exists
        if Path(result).exists():
            file_size = Path(result).stat().st_size / (1024 * 1024)  # Size in MB
            print(f"📊 File size: {file_size:.2f} MB")
            return True
        else:
            print("⚠️  Warning: Output file not found")
            return False
            
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Install required package: pip install google-genai")
        return False
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error during video generation: {e}")
        import traceback
        print("\n📋 Full traceback:")
        traceback.print_exc()
        return False


def test_multiple_prompts():
    """Test Veo with multiple prompts."""
    
    print("\n" + "=" * 60)
    print("🧪 Testing Multiple Prompts")
    print("=" * 60)
    
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY not found")
        return False
    
    test_prompts = [
        "A serene sunset over mountains, cinematic, 4K",
        "A robot walking in a futuristic city, sci-fi style",
        "Ocean waves crashing on a beach, peaceful, realistic"
    ]
    
    generator = VeoGenerator(
        api_key=api_key,
        width=768,
        height=1024,
        duration=5
    )
    
    results = []
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📝 Test {i}/{len(test_prompts)}: {prompt}")
        output_path = f"output/test_veo_{i}.mp4"
        
        try:
            result = generator.generate_video(
                prompt=prompt,
                output_path=output_path
            )
            results.append((i, True, result))
            print(f"✅ Test {i} completed: {result}")
        except Exception as e:
            results.append((i, False, str(e)))
            print(f"❌ Test {i} failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    successful = sum(1 for _, success, _ in results if success)
    print(f"✅ Successful: {successful}/{len(test_prompts)}")
    print(f"❌ Failed: {len(test_prompts) - successful}/{len(test_prompts)}")
    
    return successful == len(test_prompts)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Google Veo video generation")
    parser.add_argument(
        "--multiple",
        action="store_true",
        help="Test with multiple prompts"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom test prompt (overrides default)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/test_veo_video.mp4",
        help="Output path for test video"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.multiple:
        success = test_multiple_prompts()
    else:
        # Override prompt if provided
        if args.prompt:
            # Temporarily modify the test function
            import types
            original_test = test_veo_generation
            def custom_test():
                api_key = os.getenv('GOOGLE_API_KEY')
                if not api_key:
                    print("❌ Error: GOOGLE_API_KEY not found")
                    return False
                
                generator = VeoGenerator(
                    api_key=api_key,
                    width=768,
                    height=1024,
                    duration=5
                )
                
                print(f"📝 Custom Prompt: {args.prompt}")
                print(f"📁 Output: {args.output}")
                
                result = generator.generate_video(
                    prompt=args.prompt,
                    output_path=args.output
                )
                
                print(f"✅ Video saved to: {result}")
                return Path(result).exists()
            
            success = custom_test()
        else:
            success = test_veo_generation()
    
    sys.exit(0 if success else 1)

