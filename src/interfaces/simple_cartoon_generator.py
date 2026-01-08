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
        
    # No ElevenLabs key needed; TTS and WAN models will be downloaded automatically from Hugging Face when first used
    # Note: Models directory will be created automatically if needed
    
    print("✅ All requirements satisfied!")
    return True

def get_model_path_for_type(model_type: str) -> str | None:
    """
    Legacy function - no longer used.
    WAN 2.1 T2V is the only model and it downloads automatically from Hugging Face.
    This function is kept for backward compatibility but always returns None.
    """
    # WAN 2.1 T2V model is handled automatically by diffusers library
    # Model ID: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    # It downloads automatically on first use to Hugging Face cache
    return None

def generate_cartoon(prompt, style="realistic", duration=30, language="en", enable_prompt_enhancement=False, video_format="shorts", wan_width=832, wan_height=480, wan_num_frames=49, wan_fps=12, wan_steps=30, wan_guidance=6.0, wan_negative_prompt="text, subtitles, watermark, blurry, low quality, cartoon, anime, manga, illustration, painting, drawing, sketch, bad anatomy, distorted, deformed, ugly", seed=None, create_reel=True, vertical_mode="pad", reel_width=1080, reel_height=1920, reel_fps=30, music_path=None, music_volume=0.12, voice_volume=1.0, verbose_ffmpeg=False, voice_file=None, auto_sub=False, voice_speed=1.0):
    """Generate a video reel with the given prompt."""
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
        print("=" * 50)
        
        print(f"🎬 Starting video generation with WAN 2.1 T2V...")
        print(f"📝 Prompt: {prompt}")
        print(f"🎨 Style: {style}")
        print(f"⏱️ Duration: {duration} seconds")
        print(f"🗣️ Language: {language}")
        # Prompt enhancement disabled
        print(f"📐 Video format: {video_format}")
        print(f"🎬 WAN settings: {wan_width}x{wan_height}, {wan_num_frames} frames @ {wan_fps}fps")
        if create_reel:
            print(f"🎬 Reel enabled: {reel_width}x{reel_height} @ {reel_fps}fps, mode={vertical_mode}")
        print("-" * 50)
        
        # Create video configuration
        config = VideoConfig(
            prompt=prompt,
            duration=duration,
            style=style,
            video_format=video_format,
            output_path="output",
            add_subtitles=auto_sub,  # Enable subtitles if requested
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
            voice_speed=voice_speed,
            verbose_ffmpeg=verbose_ffmpeg,
            voice_id=voice_file if voice_file else ""  # Path to reference speaker WAV for Coqui TTS (empty string = no voice)
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
        choices=["realistic", "anime", "indian", "desi", "bollywood"],
        default="realistic",
        help="Visual style (realistic, anime, indian, etc.)"
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
        help="Path to a JSON file with custom storyboard scenes (title, description, scenes[]) or a file with 'stories' array for bulk generation"
    )
    
    parser.add_argument(
        "--scene",
        type=int,
        help="Process only a specific scene number (1-based index). Use with --storyboard to process single scene."
    )
    
    parser.add_argument(
        "--gen-bulk",
        action="store_true",
        help="Enable bulk generation from a single storyboard JSON file with multiple stories in a 'stories' array"
    )
    
    parser.add_argument(
        "--auto-sub",
        action="store_true",
        help="Enable automatic subtitle generation (disabled by default)"
    )
    
    parser.add_argument(
        "--add-hooks",
        action="store_true",
        help="Enable hook text rendering (top_hook_text, bottom_hook_text from storyboard, and scene hook_text)"
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
        default="text, subtitles, watermark, blurry, low quality, cartoon, anime, manga, illustration, painting, drawing, sketch, bad anatomy, distorted, deformed",
        help="Negative prompt for video generation (default excludes cartoon/anime/illustration for realistic videos)"
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
        "--voice",
        type=str,
        default=None,
        help="Path to reference speaker WAV file for Coqui TTS voice cloning (optional). If not provided, uses default voice or voice specified in storyboard JSON."
    )
    
    parser.add_argument(
        "--voice-speed",
        type=float,
        default=1.0,
        help="Voice speed multiplier (default: 1.0, 0.8 = slower, 1.2 = faster)"
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
            
            # Bulk generation mode
            if args.gen_bulk:
                # Debug: Log voice argument to help diagnose issues
                print(f"🔍 DEBUG: args.voice = {repr(args.voice)}")
                print(f"🔍 DEBUG: args.voice type = {type(args.voice)}")
                if args.voice:
                    print(f"🔍 DEBUG: args.voice file exists: {os.path.exists(args.voice)}")
                    print(f"🔍 DEBUG: args.voice absolute path: {os.path.abspath(args.voice) if args.voice else None}")
                else:
                    print(f"⚠️ WARNING: args.voice is None or empty! Check command line argument parsing.")
                
                storyboard_file = Path(args.storyboard)
                if not storyboard_file.exists() or not storyboard_file.is_file():
                    print(f"❌ Storyboard file does not exist: {storyboard_file}")
                    sys.exit(1)
                
                # Load the storyboard JSON file
                try:
                    with open(storyboard_file, 'r', encoding='utf-8') as f:
                        storyboard_data = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"❌ Invalid JSON in storyboard file: {e}")
                    sys.exit(1)
                except Exception as e:
                    print(f"❌ Failed to read storyboard file: {e}")
                    sys.exit(1)
                
                # Extract stories array
                stories = storyboard_data.get('stories', [])
                if not stories:
                    print(f"❌ No 'stories' array found in storyboard file. Expected format: {{'stories': [...]}}")
                    sys.exit(1)
                
                print(f"📁 Found {len(stories)} stories in {storyboard_file.name}")
                
                # Initialize pipelines once (singleton pattern ensures they're shared)
                print("🔄 Initializing pipelines (will be reused for all stories)...")
                base_config = VideoConfig(
                    prompt="",  # Will be overridden per story
                    duration=args.duration,
                    video_format=args.video_format,
                    output_path="output",
                    style=args.style,
                    language=args.language,
                    enable_prompt_enhancement=False,
                    add_subtitles=args.auto_sub,  # Only enable if --auto-sub is provided
                    wan_width=args.wan_width,
                    wan_height=args.wan_height,
                    wan_num_frames=args.wan_num_frames,
                    wan_fps=args.wan_fps,
                    wan_steps=args.wan_steps,
                    wan_guidance=args.wan_guidance,
                    wan_negative_prompt=args.negative_prompt,
                    wan_seed=args.seed,
                    skip_audio=args.skip_audio,
                    create_reel=not args.no_reel and (args.format == "reel" or args.vertical),
                    vertical_mode=args.vertical_mode,
                    reel_width=args.out_width,
                    reel_height=args.out_height,
                    reel_fps=args.out_fps,
                    music_path=args.music,
                    music_volume=args.music_volume,
                    voice_volume=args.voice_volume,
                    voice_speed=args.voice_speed,
                    verbose_ffmpeg=args.verbose_ffmpeg,
                    add_hooks=args.add_hooks,  # Enable hook text rendering
                    voice_id=args.voice if args.voice else ""  # Path to reference speaker WAV for Coqui TTS
                )
                
                # Debug: Verify voice_id was set correctly
                print(f"🔍 DEBUG: base_config.voice_id = {repr(base_config.voice_id)}")
                if base_config.voice_id:
                    print(f"🔍 DEBUG: base_config.voice_id file exists: {os.path.exists(base_config.voice_id)}")
                
                # Pre-initialize generator to load pipelines once
                print("📦 Loading WAN pipeline (singleton - will be reused)...")
                print("📦 Loading TTS model (singleton - will be reused)...")
                temp_generator = CartoonShortsGenerator(base_config)
                print("✅ Pipelines initialized and ready for bulk generation")
                
                # Process each story in the stories array
                successful = []
                failed = []
                
                for i, story in enumerate(stories, 1):
                    print(f"\n{'='*70}")
                    print(f"Processing story {i}/{len(stories)}")
                    print(f"{'='*70}")
                    
                    try:
                        # Use story title if available, otherwise use top-level title, or fallback to Story_{i}
                        title = story.get('title') or storyboard_data.get('title') or f'Story_{i}'
                        description = story.get('description', '') or storyboard_data.get('description', '')
                        scenes = story.get('scenes', [])
                        total_duration = story.get('total_duration', args.duration)
                        
                        if not scenes:
                            print(f"⚠️ No scenes found in story {i}, skipping")
                            failed.append((title, "No scenes found"))
                            continue
                        
                        # Calculate scene duration if not provided
                        for scene in scenes:
                            if 'duration' not in scene:
                                scene['duration'] = total_duration // len(scenes)
                        
                        # Create output folder by title
                        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
                        safe_title = safe_title.replace(' ', '_')[:50]
                        if not safe_title:
                            safe_title = f"Story_{i}"
                        output_folder = Path("output") / safe_title
                        output_folder.mkdir(parents=True, exist_ok=True)
                        
                        print(f"📁 Output folder: {output_folder}")
                        print(f"📋 Title: {title}")
                        print(f"🎬 Scenes: {len(scenes)}")
                        
                        # Normalize scenes
                        cast_list = story.get('cast', []) or storyboard_data.get('cast', []) or []
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
                            scene_copy['characters'] = structured_chars[:2]
                            normalized_scenes.append(scene_copy)
                        
                        # Update generator config (reuse same instance)
                        temp_generator.config.prompt = title
                        temp_generator.config.title = title
                        temp_generator.config.description = description
                        temp_generator.config.custom_scenes = normalized_scenes
                        temp_generator.config.duration = total_duration
                        # Extract hook texts from story
                        temp_generator.config.top_hook_text = story.get('top_hook_text')
                        temp_generator.config.bottom_hook_text = story.get('bottom_hook_text')
                        temp_generator.config.add_hooks = args.add_hooks
                        if normalized_scenes and 'duration' in normalized_scenes[0]:
                            temp_generator.config.scene_duration = normalized_scenes[0].get('duration', 8)
                        else:
                            temp_generator.config.scene_duration = total_duration // len(normalized_scenes) if normalized_scenes else 8
                        temp_generator.config.output_path = str(output_folder)
                        temp_generator.output_dir = output_folder
                        temp_generator.output_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Generate video using the same generator instance
                        output_path = temp_generator.generate()
                        successful.append((title, output_path))
                        print(f"✅ Successfully generated: {output_path}")
                        
                    except Exception as e:
                        print(f"❌ Failed to process story {i} ({title if 'title' in locals() else 'Unknown'}): {e}")
                        import traceback
                        traceback.print_exc()
                        failed.append((title if 'title' in locals() else f"Story_{i}", str(e)))
                
                # Print summary
                print(f"\n{'='*70}")
                print("BULK GENERATION SUMMARY")
                print(f"{'='*70}")
                print(f"✅ Successful: {len(successful)}/{len(stories)}")
                for title, output in successful:
                    print(f"   ✓ {title} -> {output}")
                
                if failed:
                    print(f"\n❌ Failed: {len(failed)}/{len(stories)}")
                    for title, error in failed:
                        print(f"   ✗ {title}: {error}")
                
                print(f"{'='*70}")
                sys.exit(0 if not failed else 1)
            
            # Single storyboard mode (existing logic)
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
            # Prompt enhancement disabled
            print("=" * 50)
            
            with open(args.storyboard, 'r', encoding='utf-8') as f:
                data = json.load(f)
            all_scenes = data.get('scenes', [])
            title = data.get('title')
            description = data.get('description')
            scene_duration = data.get('scene_duration', 8)
            top_hook_text = data.get('top_hook_text')
            bottom_hook_text = data.get('bottom_hook_text')
            
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
            
            # Normalize characters: map names in scenes to structured cast entries
            cast_list = data.get('cast', []) or []
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
                add_subtitles=args.auto_sub,  # Only enable if --auto-sub is provided
                add_hooks=args.add_hooks,  # Enable hook text rendering
                top_hook_text=top_hook_text,
                bottom_hook_text=bottom_hook_text,
                language=args.language,
                enable_prompt_enhancement=False,  # Prompt enhancement disabled
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
                voice_speed=args.voice_speed,
                verbose_ffmpeg=args.verbose_ffmpeg,
                voice_id=args.voice or ""  # Path to reference speaker WAV for Coqui TTS
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
            verbose_ffmpeg=args.verbose_ffmpeg,
            voice_file=args.voice,
            voice_speed=args.voice_speed
        )
    
    if output_path:
        print(f"\n🎊 Success! Your cartoon is ready at: {output_path}")
    else:
        print("\n❌ Generation failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
