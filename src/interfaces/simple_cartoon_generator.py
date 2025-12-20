#!/usr/bin/env python3
"""
Video Reel Generator - Easy-to-use script for generating platform-ready vertical Reels/Shorts videos
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
    
    # Note: WAN model will be downloaded automatically from Hugging Face when first used
    
    print("✅ All requirements satisfied!")
    return True

def get_model_path_for_type(model_type: str) -> str | None:
    """Get appropriate model path based on selected model type."""
    if model_type == "cartoon":
        # Cartoon models
        cartoon_models = [
            "models/toonyou_beta6.safetensors",
            "models/anything-v4.5.safetensors", 
            "models/counterfeit-v3.0.safetensors"
        ]
        for model in cartoon_models:
            if Path(model).exists():
                return model
    else:  # realistic
        # Realistic models
        realistic_models = [
            "models/realistic-vision-v4.safetensors",
            "models/deliberate-v3.safetensors",
            "models/realistic-vision-v5.1.safetensors",
            "models/dreamshaper-v8.safetensors"
        ]
        for model in realistic_models:
            if Path(model).exists():
                return model
    
    return None

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

def generate_cartoon(prompt, style="cartoon", duration=30, language="en", enable_prompt_enhancement=True, video_format="shorts", wan_width=832, wan_height=480, wan_num_frames=49, wan_fps=12, wan_steps=30, wan_guidance=6.0, wan_negative_prompt="text, subtitles, watermark, blurry, low quality", seed=None):
    """Generate a cartoon video with the given prompt and character faces."""
    try:
        # Import the main generator
        from ..core.generate_cartoon_short import CartoonShortsGenerator, VideoConfig
        
        # Log all function arguments
        print("🔍 GENERATE_CARTOON FUNCTION ARGUMENTS:")
        print("=" * 50)
        print(f"📝 Prompt: {prompt}")
        print(f"🎨 Style: {style}")
        print(f"⏱️ Duration: {duration}")
        print(f"🗣️ Language: {language}")
        print(f"✨ Enable Prompt Enhancement: {enable_prompt_enhancement}")
        print(f"📐 Video Format: {video_format}")
        print(f"🎬 WAN Width: {wan_width}, Height: {wan_height}")
        print(f"🎞️ WAN Frames: {wan_num_frames} @ {wan_fps}fps")
        print(f"⚙️ WAN Steps: {wan_steps}, Guidance: {wan_guidance}")
        if seed:
            print(f"🎲 Seed: {seed}")
        if character_faces:
            print(f"👥 Character faces: {len(character_faces)} characters mapped")
            for char, face in character_faces.items():
                print(f"   - {char}: {face}")
        else:
            print(f"👥 Character faces: None")
        print("=" * 50)
        
        print(f"🎬 Starting video generation with WAN 2.1 T2V...")
        print(f"📝 Prompt: {prompt}")
        print(f"🎨 Style: {style}")
        print(f"⏱️ Duration: {duration} seconds")
        print(f"🗣️ Language: {language}")
        print(f"🎯 Prompt enhancement: {'Enabled' if enable_prompt_enhancement else 'Disabled'}")
        print(f"📐 Video format: {video_format}")
        print(f"🎬 WAN settings: {wan_width}x{wan_height}, {wan_num_frames} frames @ {wan_fps}fps")
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
            wan_width=wan_width,
            wan_height=wan_height,
            wan_num_frames=wan_num_frames,
            wan_fps=wan_fps,
            wan_steps=wan_steps,
            wan_guidance=wan_guidance,
            wan_negative_prompt=wan_negative_prompt,
            wan_seed=seed,
            create_reel=create_reel,
            vertical_mode=vertical_mode,
            reel_width=reel_width,
            reel_height=reel_height,
            reel_fps=reel_fps,
            music_path=music_path,
            music_volume=music_volume,
            voice_volume=voice_volume,
            verbose_ffmpeg=verbose_ffmpeg
        )
        
        # Initialize generator
        generator = CartoonShortsGenerator(config)
        
        # Generate the video
        output_path = generator.generate()
        
        print("-" * 50)
        print(f"🎉 Video generation completed!")
        print(f"📁 Output file: {output_path}")
        print(f"🎬 You can now view your video reel!")
        
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
  
  # Process only the first scene from storyboard
  python simple_cartoon_generator.py --prompt "Magic forest adventure" --storyboard storyboards/tillu.json --scene 1
  
  # Custom WAN settings
  python simple_cartoon_generator.py --prompt "Adventure story" --wan-width 832 --wan-height 480 --wan-num-frames 49

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
        help="The story prompt for your video reel"
    )
    
    parser.add_argument(
        "--style", "-s",
        choices=["cartoon", "anime", "indian", "indian_cartoon", "desi", "bollywood"],
        default="cartoon",
        help="Visual style (cartoon, anime, indian). Use 'indian' for Indian children's-book style"
    )
    
    # Note: --model-type removed (WAN 2.1 is the only model now)
    
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
        "--scene",
        type=int,
        help="Process only a specific scene number (1-based index). Use with --storyboard to process single scene."
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
    
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Enable animation (default: enabled)"
    )
    
    parser.add_argument(
        "--wan-width",
        type=int,
        default=832,
        help="WAN video width (default: 832)"
    )
    
    parser.add_argument(
        "--wan-height",
        type=int,
        default=480,
        help="WAN video height (default: 480)"
    )
    
    parser.add_argument(
        "--wan-num-frames",
        type=int,
        default=49,
        help="WAN number of frames to generate (default: 49)"
    )
    
    parser.add_argument(
        "--wan-fps",
        type=int,
        default=12,
        help="WAN output FPS (default: 12)"
    )
    
    parser.add_argument(
        "--wan-steps",
        type=int,
        default=30,
        help="WAN inference steps (default: 30)"
    )
    
    parser.add_argument(
        "--wan-guidance",
        type=float,
        default=6.0,
        help="WAN guidance scale (default: 6.0)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for video generation (optional)"
    )
    
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="text, subtitles, watermark, blurry, low quality",
        help="Negative prompt for video generation (default: 'text, subtitles, watermark, blurry, low quality')"
    )
    
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="Skip audio generation (create video without narration)"
    )
    
    # Reel rendering arguments
    parser.add_argument(
        "--format",
        choices=["reel", "normal"],
        default="reel",
        help="Output format: 'reel' for platform-ready vertical reel (default), 'normal' for standard output"
    )
    
    parser.add_argument(
        "--vertical",
        action="store_true",
        help="Enable vertical reel output (same as --format reel)"
    )
    
    parser.add_argument(
        "--no-reel",
        action="store_true",
        help="Disable reel creation (use stitched video only)"
    )
    
    parser.add_argument(
        "--vertical-mode",
        choices=["pad", "crop"],
        default="pad",
        help="Vertical mode: 'pad' (safe, no cropping) or 'crop' (fills frame)"
    )
    
    parser.add_argument(
        "--out-width",
        type=int,
        default=1080,
        help="Output reel width (default: 1080)"
    )
    
    parser.add_argument(
        "--out-height",
        type=int,
        default=1920,
        help="Output reel height (default: 1920)"
    )
    
    parser.add_argument(
        "--out-fps",
        type=int,
        default=30,
        help="Output reel FPS (default: 30)"
    )
    
    parser.add_argument(
        "--music",
        type=str,
        default=None,
        help="Path to background music file (optional)"
    )
    
    parser.add_argument(
        "--music-volume",
        type=float,
        default=0.12,
        help="Background music volume (default: 0.12)"
    )
    
    parser.add_argument(
        "--voice-volume",
        type=float,
        default=1.0,
        help="Voice volume (default: 1.0)"
    )
    
    parser.add_argument(
        "--verbose-ffmpeg",
        action="store_true",
        help="Print FFmpeg commands for debugging"
    )
    
    args = parser.parse_args()
    
    # Log all arguments for debugging
    print("🔍 ARGUMENT LOGGING:")
    print("=" * 60)
    print(f"📝 Prompt: {args.prompt}")
    print(f"🎨 Style: {args.style}")
    print(f"⏱️ Duration: {args.duration} seconds")
    print(f"📐 Video Format: {args.video_format}")
    print(f"🗣️ Language: {args.language}")
    print(f"📋 Storyboard: {args.storyboard}")
    print(f"🎬 Scene: {args.scene if args.scene else 'All scenes'}")
    print(f"🔄 No Reuse: {args.no_reuse}")
    print(f"✨ No Prompt Enhancement: {args.no_prompt_enhancement}")
    print(f"🔍 Check Only: {args.check_only}")
    print(f"🎬 WAN Width: {args.wan_width}, Height: {args.wan_height}")
    print(f"🎞️ WAN Frames: {args.wan_num_frames} @ {args.wan_fps}fps")
    print(f"⚙️ WAN Steps: {args.wan_steps}, Guidance: {args.wan_guidance}")
    if args.seed:
        print(f"🎲 Seed: {args.seed}")
    print(f"🔇 Skip Audio: {args.skip_audio}")
    # Determine if reel should be created
    create_reel = not args.no_reel and (args.format == 'reel' or args.vertical)
    print(f"🎬 Reel: {'Enabled' if create_reel else 'Disabled'}")
    if create_reel:
        print(f"📐 Reel: {args.out_width}x{args.out_height} @ {args.out_fps}fps, mode={args.vertical_mode}")
        if args.music:
            print(f"🎵 Music: {args.music} (vol={args.music_volume})")
    print("=" * 60)
    
    print("🎨 Simple Cartoon Generator with Face-Based Characters")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Setup incomplete. Please fix the issues above and try again.")
        sys.exit(1)
    
    if args.check_only:
        print("\n✅ All requirements are satisfied! You're ready to generate video reels.")
        sys.exit(0)
    

    
    # Generate cartoon
    if args.storyboard:
        try:
            from ..core.generate_cartoon_short import CartoonShortsGenerator, VideoConfig
            
            # Log storyboard processing arguments
            print("🔍 STORYBOARD PROCESSING ARGUMENTS:")
            print("=" * 50)
            print(f"📋 Storyboard file: {args.storyboard}")
            print(f"🎬 Scene: {args.scene if args.scene else 'All scenes'}")
            print(f"📝 Prompt: {args.prompt}")
            print(f"🎨 Style: {args.style}")
            print(f"⏱️ Duration: {args.duration}")
            print(f"📐 Video Format: {args.video_format}")
            print(f"🗣️ Language: {args.language}")
            print(f"🎬 WAN Width: {args.wan_width}, Height: {args.wan_height}")
            print(f"🎞️ WAN Frames: {args.wan_num_frames} @ {args.wan_fps}fps")
            print(f"⚙️ WAN Steps: {args.wan_steps}, Guidance: {args.wan_guidance}")
            if args.seed:
                print(f"🎲 Seed: {args.seed}")
            print(f"🔇 Skip Audio: {args.skip_audio}")
            print(f"🔄 No Reuse: {args.no_reuse}")
            print(f"✨ No Prompt Enhancement: {args.no_prompt_enhancement}")
            print("=" * 50)
            
            with open(args.storyboard, 'r', encoding='utf-8') as f:
                data = json.load(f)
            all_scenes = data.get('scenes', [])
            title = data.get('title')
            description = data.get('description')
            scene_duration = data.get('scene_duration', 8)
            
            # Filter scenes based on --scene argument
            if args.scene:
                if args.scene < 1 or args.scene > len(all_scenes):
                    print(f"❌ Scene {args.scene} not found. Available scenes: 1-{len(all_scenes)}")
                    sys.exit(1)
                scenes = [all_scenes[args.scene - 1]]  # Convert to 0-based index
                print(f"🎬 Processing only scene {args.scene}: {scenes[0].get('description', 'No description')}")
            else:
                scenes = all_scenes
                print(f"🎬 Processing all {len(scenes)} scenes from storyboard")
            
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

            # Determine if reel should be created
            create_reel = not args.no_reel and (args.format == "reel" or args.vertical)
            
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
                wan_width=args.wan_width,
                wan_height=args.wan_height,
                wan_num_frames=args.wan_num_frames,
                wan_fps=args.wan_fps,
                wan_steps=args.wan_steps,
                wan_guidance=args.wan_guidance,
                wan_negative_prompt=args.negative_prompt,
                wan_seed=args.seed,
                skip_audio=args.skip_audio,
                create_reel=create_reel,
                vertical_mode=args.vertical_mode,
                reel_width=args.out_width,
                reel_height=args.out_height,
                reel_fps=args.out_fps,
                music_path=args.music,
                music_volume=args.music_volume,
                voice_volume=args.voice_volume,
                verbose_ffmpeg=args.verbose_ffmpeg
            )
            generator = CartoonShortsGenerator(config)
            output_path = generator.generate()
        except Exception as e:
            print(f"❌ Failed to use storyboard: {e}")
            output_path = None
    else:
        # Determine if reel should be created
        create_reel = not args.no_reel and (args.format == "reel" or args.vertical)
        
        output_path = generate_cartoon(
            args.prompt, 
            args.style, 
            args.duration, 
            args.language, 
            not args.no_prompt_enhancement, 
            args.video_format,
            args.wan_width,
            args.wan_height,
            args.wan_num_frames,
            args.wan_fps,
            args.wan_steps,
            args.wan_guidance,
            args.negative_prompt,
            args.seed,
            create_reel=create_reel,
            vertical_mode=args.vertical_mode,
            reel_width=args.out_width,
            reel_height=args.out_height,
            reel_fps=args.out_fps,
            music_path=args.music,
            music_volume=args.music_volume,
            voice_volume=args.voice_volume,
            verbose_ffmpeg=args.verbose_ffmpeg
        )
    
    if output_path:
        print(f"\n🎊 Success! Your cartoon is ready at: {output_path}")
    else:
        print("\n❌ Generation failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
