#!/usr/bin/env python3
"""
Cartoon Shorts Generator - A complete CLI tool for creating vertical cartoon-style YouTube Shorts videos.

This script follows a specific flow:
1. Generate 3-scene story with OpenAI GPT-4
2. Create cartoon images with Stable Diffusion (ToonYou/MeinaMix)
3. Animate images with AnimateDiff + cartoon LoRA
4. Generate narration with Coqui TTS (XTTS v2)
5. Add audio to video clips
6. Add subtitles and background music
7. Compile final vertical video

Usage:
    python generate_cartoon_short.py --prompt "A baby lion opens a smoothie shop in the jungle"
"""

import os
import sys
import json
import time
import argparse
import subprocess
import requests
from pathlib import Path
from typing import List, Dict
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict as DictType
from dotenv import load_dotenv

# Import modular classes
from .script_generator import ScriptGenerator
from .image_generator import ImageGenerator
from .coqui_voice_synthesizer import CoquiVoiceSynthesizer, CoquiVoiceConfig
from .animation_generator import AnimationGenerator
from .video_processor import VideoProcessor, VideoConfig as VPConfig


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cartoon_shorts.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class VideoConfig:
    """Configuration for video generation."""
    prompt: str
    duration: int = 30
    fps: int = 15
    video_format: str = "shorts"  # "shorts" for 9:16, "normal" for 16:9
    width: int = 768
    height: int = 1024  # Vertical format for Shorts
    output_path: str = "output"
    style: str = "cartoon"
    voice_id: str = ""  # Optional path to reference speaker WAV for Coqui XTTS
    language: str = "en"
    num_scenes: int = 3
    # Optional storyboard support
    title: Optional[str] = None
    description: Optional[str] = None
    custom_scenes: Optional[List[DictType]] = None  # Each item may contain: title, visual_prompt, duration (optional)
    scene_duration: int = 8  # Used when custom_scenes is provided and per-scene duration not specified
    # Reuse assets to speed up repeated runs
    reuse_existing: bool = True
    # Control subtitle rendering
    add_subtitles: bool = True
    # Control prompt enhancement
    enable_prompt_enhancement: bool = True
    # Control pause between scenes (in seconds)
    scene_pause_duration: float = 0.0  # Default 0.0 second pause between scenes (no black screens)
    # Control image validation and automatic prompt adjustment
    enable_image_validation: bool = True  # Enable automatic blank image detection and prompt adjustment

    def __post_init__(self):
        """Set dimensions based on video format."""
        if self.video_format.lower() == "shorts":
            # YouTube Shorts: 9:16 aspect ratio
            self.width = 768
            self.height = 1024
        elif self.video_format.lower() == "normal":
            # Normal video: 16:9 aspect ratio
            self.width = 1920
            self.height = 1080
        else:
            # Default to shorts if invalid format
            self.video_format = "shorts"
            self.width = 768
            self.height = 1024

class CartoonShortsGenerator:
    """Main class that orchestrates the entire video generation process."""
    
    def __init__(self, config: VideoConfig):
        self.config = config
        self.output_dir = Path(config.output_path)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components using modular classes
        self.script_generator = ScriptGenerator(os.getenv('OPENAI_API_KEY', ''))
        # Choose a more neutral/non-anime base when Indian style is requested
        default_model = "models/toonyou_beta6.safetensors"
        indian_pref_model = os.getenv("INDIAN_STYLE_MODEL", default_model)
        model_path = indian_pref_model if (config.style or "").lower() in {"indian", "indian_cartoon", "desi", "bollywood"} else default_model
        lora_path = os.getenv("INDIAN_STYLE_LORA", "") or None
        try:
            self.image_generator = ImageGenerator(
                model_path=model_path, 
                lora_path=lora_path, 
                lora_scale=0.85,
                enable_prompt_enhancement=config.enable_prompt_enhancement,
                width=config.width,
                height=config.height
            )
        except TypeError:
            # Fallback for older ImageGenerator signature
            self.image_generator = ImageGenerator(model_path=model_path)
        self.animation_generator = AnimationGenerator(width=config.width, height=config.height)
        # Initialize Coqui TTS voice synthesizer
        self.voice_synthesizer = CoquiVoiceSynthesizer(
            CoquiVoiceConfig(language=config.language)
        )

        
        # Create video config for processor
        video_config = VPConfig(
            fps=config.fps,
            width=config.width,
            height=config.height
        )
        self.video_processor = VideoProcessor(video_config)
        
    def generate(self) -> str:
        """Generate the complete cartoon short video following the specified flow."""
        logger.info(f"Starting video generation for prompt: {self.config.prompt}")
        
        try:
            # Early exit if final video already exists and reuse is enabled
            output_filename = "final_short.mp4" if self.config.video_format == "shorts" else "final_video.mp4"
            final_output = self.output_dir / output_filename
            if self.config.reuse_existing and final_output.exists():
                logger.info(f"Final video already exists and reuse is enabled: {final_output}")
                return str(final_output)
            # Step 1: Generate story
            script_path = self.output_dir / "script.json"
            if self.config.reuse_existing and script_path.exists():
                logger.info(f"Reusing existing script: {script_path}")
                with open(script_path, 'r', encoding='utf-8') as f:
                    script = json.load(f)
                try:
                    self.config.duration = int(script.get('total_duration', self.config.duration))
                except Exception:
                    pass
            elif self.config.custom_scenes and len(self.config.custom_scenes) > 0:
                logger.info("Step 1: Using custom storyboard scenes provided by user...")
                script = self.script_generator.generate_script_from_custom(
                    title=self.config.title or f"Story: {self.config.prompt}",
                    description=self.config.description or f"An adventure about: {self.config.prompt}",
                    scenes=self.config.custom_scenes,
                    default_scene_duration=self.config.scene_duration
                )
                # Overwrite duration based on resulting script
                try:
                    total_duration = sum(scene.get('duration', self.config.scene_duration) for scene in script['scenes'])
                    self.config.duration = max(self.config.duration, total_duration)
                except Exception:
                    pass
                # Save script for reuse
                with open(script_path, 'w', encoding='utf-8') as f:
                    json.dump(script, f, indent=2)
            else:
                logger.info("Step 1: Generating 3-scene story...")
                script = self.script_generator.generate_script(self.config.prompt, self.config.duration, language=self.config.language)
                with open(script_path, 'w', encoding='utf-8') as f:
                    json.dump(script, f, indent=2)
            
            # Save a human-friendly storyboard alongside the raw script
            storyboard_path = self.output_dir / "storyboard.json"
            try:
                with open(storyboard_path, 'w', encoding='utf-8') as f:
                    json.dump(script, f, indent=2)
                logger.info(f"Saved storyboard: {storyboard_path}")
            except Exception as e:
                logger.warning(f"Failed to save storyboard: {e}")

            # Step 2: Create audio clips at the beginning
            logger.info("Step 2: Creating audio clips from each scene's narration...")
            logger.info(f"📊 Total scenes to process: {len(script['scenes'])}")
            scene_audio_paths = []
            actual_scene_durations = []  # Track actual audio durations
            
            # Generate audio clips from each scene's narration
            for i, scene in enumerate(script['scenes']):
                scene_audio = self.output_dir / f"audio_scene_{i+1}.wav"
                narration_text = scene.get('narration', '')
                logger.info(f"🎵 Scene {i+1}: Processing narration ({len(narration_text)} characters)")
                
                if not (self.config.reuse_existing and scene_audio.exists()):
                    logger.info(f"🎵 Scene {i+1}: Generating new audio clip...")
                    generated_audio = self.voice_synthesizer.synthesize_voice(
                        [narration_text],
                        str(scene_audio),
                        speaker=None,
                        voice_clone_audio=self.config.voice_id or None,
                    )
                    # Use actual generated path (may switch extension on fallback)
                    scene_audio = Path(generated_audio)
                    logger.info(f"🎵 Scene {i+1}: Audio generation completed: {scene_audio}")
                else:
                    logger.info(f"🎵 Scene {i+1}: Reusing existing audio: {scene_audio}")
                
                scene_audio_paths.append(str(scene_audio))
                logger.info(f"Scene {i+1}: Audio clip ready: {scene_audio}")
            
            # Detect length of each audio clip
            logger.info("📏 Detecting length of each audio clip...")
            total_audio_duration = 0
            for i, scene_audio in enumerate(scene_audio_paths):
                logger.info(f"📏 Scene {i+1}: Analyzing audio duration...")
                actual_duration = self.video_processor.get_audio_duration(scene_audio)
                actual_scene_durations.append(actual_duration)
                total_audio_duration += actual_duration
                logger.info(f"Scene {i+1}: Audio clip length: {actual_duration:.1f}s")
            
            logger.info(f"✅ Generated {len(scene_audio_paths)} audio clips for narration")
            logger.info(f"📊 Total audio duration: {total_audio_duration:.1f}s")
            logger.info(f"📊 Average audio duration per scene: {total_audio_duration/len(actual_scene_durations):.1f}s")
            
            # Step 3: Generate cartoon images for scenes
            logger.info("Step 3: Generating cartoon images...")
            logger.info(f"🖼️ Total images to generate: {len(script['scenes'])}")
            image_paths = []
            for i, scene in enumerate(script['scenes']):
                image_path = self.output_dir / f"scene_{i+1}.png"
                logger.info(f"🖼️ Scene {i+1}: Processing image generation...")
                
                if self.config.reuse_existing and image_path.exists():
                    logger.info(f"Skipping image generation (exists): {image_path}")
                    final_image_path = str(image_path)
                else:
                    prompt, negative_prompt = self._compose_image_prompt(scene)
                    logger.info(f"🖼️ Scene {i+1}: Generating image with prompt ({len(prompt)} characters)")
                    logger.info(f"🖼️ Scene {i+1}: Prompt preview: {prompt[:100]}...")
                    if negative_prompt:
                        logger.info(f"🖼️ Scene {i+1}: Using negative prompt ({len(negative_prompt)} characters)")
                    
                    # Use validation method if enabled, otherwise use standard generation
                    if self.config.enable_image_validation:
                        final_image_path = self.image_generator.generate_cartoon_image_with_validation(prompt, str(image_path), max_attempts=3, negative_prompt=negative_prompt)
                    else:
                        final_image_path = self.image_generator.generate_cartoon_image(prompt, str(image_path), negative_prompt=negative_prompt)
                    logger.info(f"🖼️ Scene {i+1}: Image generation completed: {final_image_path}")
                
                image_paths.append(final_image_path)
                logger.info(f"Scene {i+1}: Image ready: {final_image_path}")
            
            # Step 4: Animate images with AnimateDiff (using audio clip lengths)
            logger.info("Step 4: Creating animations with audio clip timing...")
            logger.info(f"🎬 Total animations to create: {len(image_paths)}")
            scene_prompts = [scene['visual_prompt'] for scene in script['scenes']]
            
            # Calculate frames per scene based on actual audio clip lengths
            frames_per_scene = []
            total_frames_needed = 0
            logger.info("📊 Calculating frames per scene based on audio durations...")
            for i, audio_duration in enumerate(actual_scene_durations):
                # Use actual audio duration to determine frame count
                total_frames = max(30, int(audio_duration * self.config.fps))
                frames_per_scene.append(total_frames)
                total_frames_needed += total_frames
                logger.info(f"📊 Scene {i+1}: {audio_duration:.1f}s → {total_frames} frames")
            
            logger.info(f"📊 Audio clip lengths: {actual_scene_durations} seconds")
            logger.info(f"📊 Frames per scene: {frames_per_scene}")
            logger.info(f"📊 Total frames needed: {total_frames_needed}")
            logger.info(f"📊 FPS setting: {self.config.fps}")
            logger.info("🎬 Using audio clip timing for animation frames")
            
            frame_dirs: List[str] = []
            for i, image_path in enumerate(image_paths):
                frames_dir = self.output_dir / f"scene_{i+1}_frames"
                expected_frames = frames_per_scene[i]
                logger.info(f"🎬 Scene {i+1}: Starting animation process...")
                logger.info(f"🎬 Scene {i+1}: Target frames: {expected_frames} ({actual_scene_durations[i]:.1f}s @ {self.config.fps}fps)")
                
                if self.config.reuse_existing and frames_dir.exists():
                    # Count frames
                    existing = list(frames_dir.glob("frame_*.png"))
                    logger.info(f"🎬 Scene {i+1}: Checking existing frames: {len(existing)} found")
                    if len(existing) >= expected_frames:
                        logger.info(f"Skipping animation (frames ready): {frames_dir} ({len(existing)} frames)")
                        frame_dirs.append(str(frames_dir))
                        continue
                    else:
                        logger.info(f"🎬 Scene {i+1}: Existing frames insufficient ({len(existing)} < {expected_frames}), regenerating...")
                
                # Generate frames based on audio clip length
                logger.info(f"🎬 Scene {i+1}: Generating {expected_frames} frames from image...")
                dir_path = self.animation_generator.animate_image(
                    image_path,
                    str(frames_dir),
                    num_frames=expected_frames,
                    prompt=scene_prompts[i] if i < len(scene_prompts) else ""
                )
                frame_dirs.append(dir_path)
                logger.info(f"🎬 Scene {i+1}: Animation completed: {dir_path}")
            
            # Step 5: Convert frames to MP4 videos (already expanded to match audio)
            logger.info("Step 5: Converting frames to MP4 videos (expanded to match audio)...")
            logger.info(f"📹 Total videos to create: {len(frame_dirs)}")
            video_clips: List[str] = []
            total_video_duration = 0
            
            for i, frames_dir in enumerate(frame_dirs):
                clip_path = self.output_dir / f"scene_{i+1}.mp4"
                expected_duration = actual_scene_durations[i]
                logger.info(f"📹 Scene {i+1}: Processing video creation...")
                logger.info(f"📹 Scene {i+1}: Expected duration: {expected_duration:.1f}s")
                
                if self.config.reuse_existing and clip_path.exists():
                    logger.info(f"Skipping frames->video (exists): {clip_path}")
                    video_clips.append(str(clip_path))
                    total_video_duration += expected_duration
                    continue
                
                logger.info(f"📹 Scene {i+1}: Converting {frames_per_scene[i]} frames to MP4...")
                video_path = self.video_processor.frames_to_video(frames_dir, str(clip_path), fps=self.config.fps)
                video_clips.append(video_path)
                total_video_duration += expected_duration
                logger.info(f"Scene {i+1}: Created MP4 with {actual_scene_durations[i]:.1f}s duration")
                logger.info(f"📹 Scene {i+1}: Video file: {video_path}")
            
            logger.info(f"📊 Total video duration created: {total_video_duration:.1f}s")
            logger.info(f"📊 Video clips ready: {len(video_clips)}")
            
            # Step 6: Add pauses between scenes if configured
            logger.info("Step 6: Adding pauses between scenes...")
            final_clips = []
            final_audio_paths = []
            
            if self.config.scene_pause_duration > 0:
                logger.info(f"⏸️ Adding {self.config.scene_pause_duration:.2f}s pauses between scenes")
                
                for i, (video_clip, audio_clip) in enumerate(zip(video_clips, scene_audio_paths)):
                    # Add scene video and audio
                    final_clips.append(video_clip)
                    final_audio_paths.append(audio_clip)
                    
                    # Add pause between scenes (except after the last scene)
                    if i < len(video_clips) - 1:
                        pause_video = self.output_dir / f"pause_{i+1}.mp4"
                        pause_audio = self.output_dir / f"pause_audio_{i+1}.aac"
                        
                        if not (self.config.reuse_existing and pause_video.exists()):
                            logger.info(f"⏸️ Creating pause {i+1}: {self.config.scene_pause_duration:.2f}s")
                            self.video_processor.create_pause_video(
                                self.config.scene_pause_duration, 
                                str(pause_video)
                            )
                            self.video_processor.create_silent_audio(
                                self.config.scene_pause_duration, 
                                str(pause_audio)
                            )
                        else:
                            logger.info(f"⏸️ Reusing existing pause {i+1}")
                        
                        final_clips.append(str(pause_video))
                        final_audio_paths.append(str(pause_audio))
            else:
                logger.info("⏸️ No pauses configured - scenes will transition directly")
                final_clips = video_clips
                final_audio_paths = scene_audio_paths
            
            logger.info(f"📊 Final clips to compile: {len(final_clips)} (including pauses)")
            logger.info(f"📊 Final audio clips: {len(final_audio_paths)} (including silence)")
            
            # Step 7: Create subtitles (optional) - accounting for pauses
            logger.info("Step 7: Creating subtitles...")
            subtitles_path = self.output_dir / "subtitles.srt"
            if self.config.add_subtitles:
                logger.info("📝 Subtitles enabled - creating SRT file...")
                if self.config.reuse_existing and subtitles_path.exists():
                    logger.info(f"Skipping subtitles (exists): {subtitles_path}")
                else:
                    logger.info(f"📝 Generating subtitles for {len(script['scenes'])} scenes (with pauses)...")
                    self._create_subtitles_with_pauses(script, str(subtitles_path))
                    logger.info(f"📝 Subtitles created: {subtitles_path}")
            else:
                logger.info("Step 7: Subtitles disabled; skipping SRT generation and overlay")
            
            # Step 8: Select background music
            logger.info("Step 8: Adding background music...")
            background_music = self._get_background_music()
            if background_music:
                logger.info(f"🎵 Background music selected: {background_music}")
            else:
                logger.info("🎵 No background music found - proceeding without music")
            
            # Step 9: Compile final video
            logger.info("Step 9: Compiling final video...")
            logger.info(f"🎬 Compiling {len(final_clips)} video clips...")
            logger.info(f"🎵 Using {len(final_audio_paths)} audio clips...")
            logger.info(f"📝 Subtitles: {'Enabled' if self.config.add_subtitles else 'Disabled'}")
            logger.info(f"🎵 Background music: {'Yes' if background_music else 'No'}")
            logger.info(f"📁 Final output: {final_output}")
            
            self.video_processor.compile_final_video(
                final_clips,
                final_audio_paths,  # Pass audio paths with pauses included
                background_music,
                str(subtitles_path) if self.config.add_subtitles else None,
                str(final_output)
            )
            
            # Step 10: Generate metadata
            logger.info("Step 10: Generating metadata...")
            self._generate_metadata(script, str(final_output))
            
            logger.info(f"🎉 Video generation completed: {final_output}")
            logger.info(f"📊 Final video duration: {total_video_duration:.1f}s")
            logger.info(f"📊 Total processing time: Audio={total_audio_duration:.1f}s, Video={total_video_duration:.1f}s")
            return str(final_output)
            
        except Exception as e:
            logger.warning(f"Per-scene audio detection failed; falling back to single track: {e}")
            logger.info("🔄 Switching to single-track audio generation mode...")
            
            # Fallback: Generate single narration track
            narration_path = self.output_dir / "narration_single.m4a"
            logger.info(f"🎵 Fallback: Creating single narration track: {narration_path}")
            
            if not (self.config.reuse_existing and narration_path.exists()):
                narration_lines = [scene.get('narration', '') for scene in script.get('scenes', [])]
                logger.info(f"🎵 Fallback: Generating single audio for {len(narration_lines)} scenes...")
                total_chars = sum(len(line) for line in narration_lines)
                logger.info(f"🎵 Fallback: Total characters to synthesize: {total_chars}")
                
                generated_audio = self.voice_synthesizer.synthesize_voice(
                    narration_lines,
                    str(narration_path),
                    speaker=None,
                    voice_clone_audio=self.config.voice_id or None,
                )
                narration_path = Path(generated_audio)
                logger.info(f"🎵 Fallback: Single audio generation completed: {narration_path}")
            else:
                logger.info(f"🎵 Fallback: Reusing existing single audio: {narration_path}")
            
            # Detect actual duration of the single track
            logger.info("📏 Fallback: Detecting single track duration...")
            actual_duration = self.video_processor.get_audio_duration(str(narration_path))
            script['total_duration'] = actual_duration
            self.config.duration = max(self.config.duration, actual_duration)
            
            # For single track, we need to extend videos to match the actual audio duration
            logger.info(f"✅ Single track duration detected: {actual_duration:.1f}s")
            
            # Store the actual audio duration for video adjustment
            actual_scene_durations = [actual_duration / len(script['scenes'])] * len(script['scenes'])
            logger.info(f"📊 Fallback: Distributed duration per scene: {actual_scene_durations}")
            
            # Update script durations to match actual audio
            scene_count = len(script['scenes'])
            if scene_count > 0:
                per_scene_duration = actual_duration / scene_count
                logger.info(f"🔄 Distributing single track duration ({actual_duration:.1f}s) across {scene_count} scenes")
                for i, scene in enumerate(script['scenes']):
                    original_duration = scene.get('duration', 8)
                    scene['original_duration'] = original_duration
                    scene['duration'] = per_scene_duration
                    logger.info(f"  Scene {i+1}: {original_duration:.1f}s → {per_scene_duration:.1f}s")
            
            # Videos are already created with correct duration matching audio clips
            final_clips = video_clips
            logger.info("✅ Videos already created with correct duration matching audio clips")
            logger.info(f"📊 Fallback: Using {len(final_clips)} video clips")
            
            # Use single narration track for final compilation
            scene_audio_paths = [str(narration_path)]
            logger.info(f"🎵 Fallback: Using single audio track: {narration_path}")
            
            # Step 7: Create subtitles (optional)
            subtitles_path = self.output_dir / "subtitles.srt"
            if self.config.add_subtitles:
                logger.info("Step 7: Creating subtitles...")
                if self.config.reuse_existing and subtitles_path.exists():
                    logger.info(f"Skipping subtitles (exists): {subtitles_path}")
                else:
                    self.video_processor.create_subtitles_srt(script, str(subtitles_path))
            else:
                logger.info("Step 7: Subtitles disabled; skipping SRT generation and overlay")
            
            # Step 8: Select background music
            logger.info("Step 8: Adding background music...")
            background_music = self._get_background_music()
            
            # Step 9: Compile final video
            logger.info("Step 9: Compiling final video...")
            self.video_processor.compile_final_video(
                final_clips,
                scene_audio_paths,  # Pass audio paths directly - compile_final_video will handle concatenation
                background_music,
                str(subtitles_path) if self.config.add_subtitles else None,
                str(final_output)
            )
            
            # Step 10: Generate metadata
            self._generate_metadata(script, str(final_output))
            
            logger.info(f"Video generation completed: {final_output}")
            return str(final_output)
            
        except Exception as e:
            logger.error(f"Error in video generation: {e}")
            raise
    
    def _get_background_music(self) -> str:
        """Get background music file path."""
        music_dir = Path("music")
        if music_dir.exists():
            music_files = list(music_dir.glob("*.mp3"))
            if music_files:
                return str(music_files[0])
        return None
    
    def _generate_metadata(self, script: Dict, video_path: str):
        """Generate metadata file for the video."""
        metadata = {
            "title": script['title'],
            "description": script['description'],
            "tags": script['tags'],
            "video_path": video_path,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "prompt": self.config.prompt,
                "duration": self.config.duration,
                "style": self.config.style
            }
        }
        
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Generated metadata: {metadata_path}")

    def _create_subtitles_with_pauses(self, script: Dict, output_path: str) -> str:
        """Create SRT subtitle file from script, accounting for pauses between scenes."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                start_time = 0
                for i, scene in enumerate(script['scenes']):
                    # Calculate end time including scene duration
                    scene_duration = scene.get('duration', 8)
                    end_time = start_time + scene_duration
                    
                    # Convert seconds to SRT time format
                    start_str = self._seconds_to_srt_time(start_time)
                    end_str = self._seconds_to_srt_time(end_time)
                    
                    f.write(f"{i+1}\n")
                    f.write(f"{start_str} --> {end_str}\n")
                    f.write(f"{scene['subtitle']}\n\n")
                    
                    # Add pause duration to start time for next scene
                    start_time = end_time + self.config.scene_pause_duration
            
            logger.info(f"Created subtitles with pauses: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating subtitles with pauses: {e}")
            return ""

    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

    def _compose_image_prompt(self, scene: Dict) -> tuple[str, str]:
        """Compose an image prompt and negative prompt using the visual_prompt from script with intelligent aspect ratio adaptation."""
        # Get the visual_prompt from the script (this is the key requirement)
        visual_prompt = scene.get('visual_prompt', '')
        base_prompt = visual_prompt or "Cartoon scene"
        
        # Get the negative prompt from the script
        negative_prompt = scene.get('negative_prompt', '')

        # Determine aspect ratio for intelligent prompt adaptation
        is_16_9_format = self.config.width > self.config.height and self.config.width / self.config.height > 1.5
        
        # If prompt enhancement is enabled, use simple enhancement without format-specific restrictions
        if self.config.enable_prompt_enhancement and hasattr(self, 'image_generator') and self.image_generator.prompt_enhancer:
            # Use simple enhancement for all formats - no format-specific restrictions
            enhanced_prompt = self.image_generator.prompt_enhancer.enhance_prompt(
                base_prompt,
                enhancement_type="cartoon",
                max_tokens=77  # Diffusion model token limit
            )
            logger.info(f"🎯 Enhanced prompt: {enhanced_prompt[:100]}...")
            return enhanced_prompt, negative_prompt
        else:
            logger.info(f"🎯 Using original visual_prompt from script: {base_prompt}")
            return base_prompt, negative_prompt

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate cartoon-style videos (YouTube Shorts or normal format)")
    parser.add_argument("--prompt", required=True, help="Story prompt (e.g., 'A baby lion opens a smoothie shop in the jungle')")
    parser.add_argument("--duration", type=int, default=30, help="Video duration in seconds")
    parser.add_argument("--video-format", choices=["shorts", "normal"], default="shorts", 
                       help="Video format: 'shorts' for 9:16 YouTube Shorts, 'normal' for 16:9 standard videos")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--style", default="cartoon", help="Visual style")
    parser.add_argument("--voice", default="", help="Reference speaker WAV path for Coqui XTTS (optional)")
    parser.add_argument("--language", default="en", help="Language for narration")
    parser.add_argument("--no-prompt-enhancement", action="store_true", help="Disable GPT-2 prompt enhancement")
    parser.add_argument("--scene-pause", type=float, default=0.0, help="Pause duration between scenes in seconds (default: 0.0, no black screens)")
    parser.add_argument("--no-image-validation", action="store_true", help="Disable automatic image validation and prompt adjustment")
    
    args = parser.parse_args()
    
    # Validate environment variables
    required_env_vars = ['OPENAI_API_KEY']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.info("Please set the following environment variables:")
        for var in missing_vars:
            logger.info(f"  {var}")
        sys.exit(1)
    
    # Create configuration
    config = VideoConfig(
        prompt=args.prompt,
        duration=args.duration,
        video_format=args.video_format,
        output_path=args.output,
        style=args.style,
        voice_id=args.voice,
        language=args.language,
        enable_prompt_enhancement=not args.no_prompt_enhancement,
        scene_pause_duration=args.scene_pause,
        enable_image_validation=not args.no_image_validation
    )
    
    # Generate video
    try:
        generator = CartoonShortsGenerator(config)
        output_path = generator.generate()
        print(f"\n🎉 Video generated successfully!")
        print(f"📁 Output: {output_path}")
        print(f"📊 Check the output directory for all generated assets")
        
    except Exception as e:
        logger.error(f"Failed to generate video: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
