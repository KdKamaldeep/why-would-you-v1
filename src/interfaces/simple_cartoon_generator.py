#!/usr/bin/env python3
"""
Simple Cartoon Generator - Easy-to-use script for generating cartoon videos with face-based character generation
"""

import os
import sys
import argparse
import json
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
    # Coqui TTS does not require an API key when using local models
    
    if not openai_key or openai_key == 'your_openai_api_key_here':
        print("❌ OpenAI API key not configured in .env file")
        return False
        
    # No ElevenLabs key needed; ensure TTS model directory exists if using local models
    
    # Check if models directory exists
    if not Path("models").exists():
        print("❌ Models directory not found. Please run the model download script first.")
        return False
    
    # Check for at least one supported model (others optional)
    preferred_models = [
        "models/toonyou_beta6.safetensors",
        "models/meina_mix.safetensors"
    ]
    if not any(Path(m).exists() for m in preferred_models):
        print("⚠️ No preferred SD models found (toonyou or meina). The app will use placeholder images.")
    
    print("✅ All requirements satisfied!")
    return True

def extract_character_faces_from_cast(cast_list):
    """Extract character face mappings from the cast array in storyboard."""
    character_faces = {}
    
    if not cast_list:
        return character_faces
    
    for cast_member in cast_list:
        if isinstance(cast_member, dict):
            character_name = cast_member.get('name', '')
            face_path = cast_member.get('face', '')  # New field for face image path
            
            if character_name and face_path:
                if Path(face_path).exists():
                    character_faces[character_name] = face_path
                    print(f"✅ Character '{character_name}' will use face: {face_path}")
                else:
                    print(f"⚠️ Face image not found for character '{character_name}': {face_path}")
            elif character_name:
                print(f"ℹ️ Character '{character_name}' will use auto-generated face")
        elif isinstance(cast_member, str):
            print(f"ℹ️ Character '{cast_member}' will use auto-generated face")
    
    if character_faces:
        print(f"✅ Found {len(character_faces)} characters with custom faces")
    
    return character_faces

def generate_cartoon(prompt, style="cartoon", duration=30, language="en", enable_prompt_enhancement=True, video_format="shorts", character_faces=None):
    """Generate a cartoon video with the given prompt and character faces."""
    try:
        # Import the main generator
        from ..core.generate_cartoon_short import CartoonShortsGenerator, VideoConfig
        
        print(f"🎬 Starting cartoon generation...")
        print(f"📝 Prompt: {prompt}")
        print(f"🎨 Style: {style}")
        print(f"⏱️ Duration: {duration} seconds")
        print(f"🗣️ Language: {language}")
        print(f"🎯 Prompt enhancement: {'Enabled' if enable_prompt_enhancement else 'Disabled'}")
        print(f"📐 Video format: {video_format}")
        if character_faces:
            print(f"👥 Character faces: {len(character_faces)} characters mapped")
            for char, face in character_faces.items():
                print(f"   - {char}: {face}")
        print("-" * 50)
        
        # Create video configuration
        config = VideoConfig(
            prompt=prompt,
            duration=duration,
            style=style,
            video_format=video_format,
            output_path="output",
            add_subtitles=False,
            language=language,
            enable_prompt_enhancement=enable_prompt_enhancement,
            character_faces=character_faces or {}
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
        description="Generate cartoon videos with AI and face-based character generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic generation
  python simple_cartoon_generator.py --prompt "A baby lion opens a smoothie shop"
  
  # Custom style and duration
  python simple_cartoon_generator.py --prompt "A robot learns to dance" --style anime --duration 45
  
  # With storyboard (includes character faces from cast)
  python simple_cartoon_generator.py --prompt "Magic forest adventure" --storyboard storyboards/tillu.json

Storyboard Cast Format (with face images):
  {
    "cast": [
      {"name": "Lion", "role": "main character", "face": "faces/lion_face.jpg"},
      {"name": "Robot", "role": "helper", "face": "faces/robot_face.jpg"},
      {"name": "Princess", "role": "customer"}
    ]
  }
        """
    )
    
    parser.add_argument(
        "--prompt", "-p",
        required=True,
        help="The story prompt for your cartoon video"
    )
    
    parser.add_argument(
        "--style", "-s",
        choices=["cartoon", "anime", "indian", "indian_cartoon", "desi", "bollywood"],
        default="cartoon",
        help="Visual style (cartoon, anime, indian). Use 'indian' for Indian children's-book style"
    )
    
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=30,
        help="Video duration in seconds (default: 30)"
    )

    parser.add_argument(
        "--video-format", "-f",
        choices=["shorts", "normal"],
        default="shorts",
        help="Video format: 'shorts' for YouTube Shorts (9:16) or 'normal' for standard video (16:9)"
    )

    parser.add_argument(
        "--language", "-l",
        default="en",
        help="Narration language (e.g., en, hi, es). For Hindi use 'hi'"
    )

    parser.add_argument(
        "--storyboard",
        type=str,
        help="Path to a JSON file with custom storyboard scenes (title, description, scenes[])"
    )
    

    
    parser.add_argument(
        "--no-reuse",
        action="store_true",
        help="Force regeneration of all assets (ignore cached outputs)"
    )
    
    parser.add_argument(
        "--no-prompt-enhancement",
        action="store_true",
        help="Disable prompt enhancement (use raw prompts without AI enhancement)"
    )
    
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check requirements, don't generate video"
    )
    
    args = parser.parse_args()
    
    print("🎨 Simple Cartoon Generator with Face-Based Characters")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Setup incomplete. Please fix the issues above and try again.")
        sys.exit(1)
    
    if args.check_only:
        print("\n✅ All requirements are satisfied! You're ready to generate cartoons.")
        sys.exit(0)
    

    
    # Generate cartoon
    if args.storyboard:
        try:
            from ..core.generate_cartoon_short import CartoonShortsGenerator, VideoConfig
            with open(args.storyboard, 'r', encoding='utf-8') as f:
                data = json.load(f)
            scenes = data.get('scenes', [])
            title = data.get('title')
            description = data.get('description')
            scene_duration = data.get('scene_duration', 8)
            
            # Extract character faces from cast array
            cast_list = data.get('cast', []) or []
            character_faces = extract_character_faces_from_cast(cast_list)
            
            # Normalize characters: map names in scenes to structured cast entries
            name_to_cast = {}
            for entry in cast_list:
                if isinstance(entry, dict) and entry.get('name'):
                    name_to_cast[entry['name']] = entry
                elif isinstance(entry, str):
                    name_to_cast[entry] = {"name": entry, "role": "character"}

            normalized_scenes = []
            for scene in scenes:
                scene_copy = dict(scene)
                scene_chars = scene_copy.get('characters', [])
                structured_chars = []
                for ch in scene_chars:
                    if isinstance(ch, dict):
                        structured_chars.append(ch)
                    elif isinstance(ch, str):
                        base = name_to_cast.get(ch, {"name": ch, "role": "character"})
                        structured_chars.append({
                            "name": base.get("name", ch),
                            "role": base.get("role", "character")
                        })
                # Enforce two-character focus
                scene_copy['characters'] = structured_chars[:2]
                normalized_scenes.append(scene_copy)

            config = VideoConfig(
                prompt=args.prompt,
                duration=args.duration,
                style=args.style,
                video_format=args.video_format,
                output_path="output",
                title=title,
                description=description,
                custom_scenes=normalized_scenes,
                scene_duration=scene_duration,
                reuse_existing=(not args.no_reuse),
                add_subtitles=False,
                language=args.language,
                enable_prompt_enhancement=(not args.no_prompt_enhancement),
                character_faces=character_faces
            )
            generator = CartoonShortsGenerator(config)
            output_path = generator.generate()
        except Exception as e:
            print(f"❌ Failed to use storyboard: {e}")
            output_path = None
    else:
        output_path = generate_cartoon(
            args.prompt, 
            args.style, 
            args.duration, 
            args.language, 
            not args.no_prompt_enhancement, 
            args.video_format,
            {}  # No character faces for non-storyboard generation
        )
    
    if output_path:
        print(f"\n🎊 Success! Your cartoon is ready at: {output_path}")
    else:
        print("\n❌ Generation failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
