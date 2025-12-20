#!/usr/bin/env python3
"""
Cartoon Shorts Generator - A complete CLI tool for creating platform-ready vertical Reels/Shorts videos.

This script follows a specific flow:
1. Generate 3-scene story with OpenAI GPT-4
2. Generate videos directly with WAN 2.1 Text-to-Video (T2V)
3. Generate narration with Coqui TTS (XTTS v2)
4. Stitch scene videos together
5. Create platform-ready reel (1080×1920, H.264/AAC, 30fps)

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
from .wan_t2v import WanT2VGenerator
from .coqui_voice_synthesizer import CoquiVoiceSynthesizer, CoquiVoiceConfig
from .video_processor import VideoProcessor, VideoConfig as VPConfig


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_reel.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class VideoConfig:
    """Configuration for video generation."""
    prompt: str
    duration: int = 30
    fps: int = 10  # Reduced from 15 to 10 for slower playback
    video_format: str = "shorts"  # "shorts" for 9:16, "normal" for 16:9
    width: int = 768
    height: int = 1024  # Vertical format for Shorts
    output_path: str = "output"
    style: str = "realistic"
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
    # WAN T2V settings
    wan_width: int = 832  # WAN video width
    wan_height: int = 480  # WAN video height
    wan_num_frames: int = 49  # WAN number of frames to generate
    wan_fps: int = 12  # WAN output FPS
    wan_steps: int = 30  # WAN inference steps
    wan_guidance: float = 6.0  # WAN guidance scale
    wan_negative_prompt: str = "text, subtitles, watermark, blurry, low quality"  # WAN negative prompt
    wan_seed: Optional[int] = None  # WAN random seed (optional)
    # Audio settings
    skip_audio: bool = False  # Skip audio generation entirely
    # Reel/Shorts rendering settings
    create_reel: bool = True  # Create platform-ready reel (default: True for vertical format)
    vertical_mode: str = "pad"  # "pad" (safe) or "crop" (fills frame)
    reel_width: int = 1080  # Reel output width
    reel_height: int = 1920  # Reel output height
    reel_fps: int = 30  # Reel output FPS
    music_path: Optional[str] = None  # Background music path
    music_volume: float = 0.12  # Music volume (0.0-1.0)
    voice_volume: float = 1.0  # Voice volume (0.0-1.0)
    verbose_ffmpeg: bool = False  # Print FFmpeg commands

    def __post_init__(self):
        """Set dimensions based on video format."""
        if self.video_format.lower() == "shorts":
            # YouTube Shorts: 9:16 aspect ratio
            # Note: WAN uses fixed dimensions, but we'll crop/resize in post-processing
            self.width = 768
            self.height = 1024
        elif self.video_format.lower() == "normal":
            # Normal video: 16:9 aspect ratio
            # Note: WAN uses fixed dimensions, but we'll crop/resize in post-processing
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
        
        # Initialize WAN T2V generator
        self.wan_generator = WanT2VGenerator(
            width=config.wan_width,
            height=config.wan_height,
            num_frames=config.wan_num_frames,
            fps=config.wan_fps,
            num_inference_steps=config.wan_steps,
            guidance_scale=config.wan_guidance,
            negative_prompt=config.wan_negative_prompt
        )
        
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
        """Generate the complete video reel following the specified flow."""
        logger.info(f"Starting video generation for prompt: {self.config.prompt}")
        
        try:
            # Early exit if final video already exists and reuse is enabled
            if self.config.create_reel:
                # Check for reel output
                final_output = self.output_dir / "final_reel.mp4"
            else:
                # Check for stitched output
                final_output = self.output_dir / "stitched.mp4"
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
            
            # Load cast information for character role enhancement
            self.cast = script.get('cast', [])
            if self.cast:
                logger.info(f"🎭 Loaded cast information: {len(self.cast)} characters")
                for character in self.cast:
                    if isinstance(character, dict):
                        name = character.get('name', 'Unknown')
                        role = character.get('role', 'No role specified')
                        logger.info(f"   • {name}: {role}")
            else:
                logger.info("ℹ️ No cast information found in script")

            # Step 2: Create audio clips at the beginning
            if self.config.skip_audio:
                logger.info("Step 2: Skipping audio generation (--skip-audio flag set)")
                logger.info(f"📊 Total scenes to process: {len(script['scenes'])}")
                scene_audio_paths = []
                actual_scene_durations = []
                
                # Create dummy audio durations based on scene durations
                for i, scene in enumerate(script['scenes']):
                    scene_duration = scene.get('duration', self.config.scene_duration)
                    actual_scene_durations.append(scene_duration)
                    scene_audio_paths.append("")  # Empty string for no audio
                    logger.info(f"Scene {i+1}: Using scene duration {scene_duration}s (no audio)")
                
                logger.info(f"✅ Skipped audio generation for {len(script['scenes'])} scenes")
                logger.info(f"📊 Using scene durations: {actual_scene_durations}")
            else:
                logger.info("Step 2: Creating audio clips from each scene's narration...")
                logger.info(f"📊 Total scenes to process: {len(script['scenes'])}")
                scene_audio_paths = []
                actual_scene_durations = []  # Track actual audio durations
                
                # Generate audio clips from each scene's narration
                for i, scene in enumerate(script['scenes']):
                    scene_audio = self.output_dir / f"audio_scene_{i+1}.wav"
                    narration_text = scene.get('narration', '')
                    logger.info(f"🎵 Scene {i+1}: Processing narration ({len(narration_text)} characters)")
                    
                    # Get voice file from scene if available
                    voice_file = None
                    if 'voice' in scene:
                        # Import narration converter to resolve voice files
                        from ..utils.narration_converter import NarrationConverter
                        converter = NarrationConverter()
                        voice_file = converter.resolve_voice_file(scene['voice'])
                        if voice_file:
                            logger.info(f"🎵 Scene {i+1}: Using voice file: {voice_file}")
                            logger.info(f"🎵 Scene {i+1}: Voice property: '{scene['voice']}' -> resolved to: {voice_file}")
                        else:
                            logger.warning(f"🎵 Scene {i+1}: Voice file not found for '{scene['voice']}'")
                    else:
                        logger.info(f"🎵 Scene {i+1}: No voice property found in scene")
                    
                    if not (self.config.reuse_existing and scene_audio.exists()):
                        logger.info(f"🎵 Scene {i+1}: Generating new audio clip...")
                        final_voice_file = voice_file or self.config.voice_id or None
                        logger.info(f"🎵 Scene {i+1}: Final voice_clone_audio parameter: {final_voice_file}")
                        generated_audio = self.voice_synthesizer.synthesize_voice(
                            [narration_text],
                            str(scene_audio),
                            speaker=None,
                            voice_clone_audio=final_voice_file,
                        )
                        # Use actual generated path (may switch extension on fallback)
                        scene_audio = Path(generated_audio)
                        logger.info(f"🎵 Scene {i+1}: Audio generation completed: {scene_audio}")
                    else:
                        logger.info(f"🎵 Scene {i+1}: Reusing existing audio: {scene_audio}")
                    
                    scene_audio_paths.append(str(scene_audio))
                    logger.info(f"Scene {i+1}: Audio clip ready: {scene_audio}")
            
            # Detect length of each audio clip
            if not self.config.skip_audio:
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
            else:
                # Calculate total duration from scene durations
                total_audio_duration = sum(actual_scene_durations)
                logger.info(f"✅ Skipped audio generation")
                logger.info(f"📊 Total scene duration: {total_audio_duration:.1f}s")
                logger.info(f"📊 Average scene duration: {total_audio_duration/len(actual_scene_durations):.1f}s")
            
            # Step 3: Generate videos directly from prompts using WAN T2V
            logger.info("Step 3: Generating videos with WAN 2.1 T2V...")
            logger.info(f"🎬 Total videos to generate: {len(script['scenes'])}")
            video_clips: List[str] = []
            total_video_duration = 0
            
            # Create scenes subdirectory
            scenes_dir = self.output_dir / "scenes"
            scenes_dir.mkdir(parents=True, exist_ok=True)
            
            for i, scene in enumerate(script['scenes']):
                clip_path = scenes_dir / f"scene_{i+1}.mp4"
                logger.info(f"🎬 Scene {i+1}: Processing video generation...")
                
                if self.config.reuse_existing and clip_path.exists():
                    logger.info(f"Skipping video generation (exists): {clip_path}")
                    video_clips.append(str(clip_path))
                    # Get actual duration of existing video
                    existing_duration = self.video_processor.get_audio_duration(str(clip_path))
                    total_video_duration += existing_duration
                    continue
                
                # Compose prompt from scene
                prompt, negative_prompt = self._compose_video_prompt(scene)
                logger.info(f"🎬 Scene {i+1}: Generating video with prompt ({len(prompt)} characters)")
                logger.info(f"🎬 Scene {i+1}: Prompt preview: {prompt[:100]}...")
                if negative_prompt:
                    logger.info(f"🎬 Scene {i+1}: Using negative prompt ({len(negative_prompt)} characters)")
                
                # Generate video with WAN
                try:
                    video_path = self.wan_generator.generate_video(
                        prompt=prompt,
                        output_path=str(clip_path),
                        seed=self.config.wan_seed,
                        negative_prompt=negative_prompt or None
                    )
                    video_clips.append(video_path)
                    
                    # Get actual duration of generated video
                    actual_duration = self.video_processor.get_audio_duration(video_path)
                    total_video_duration += actual_duration
                    logger.info(f"🎬 Scene {i+1}: Video generated: {video_path} ({actual_duration:.1f}s)")
                except Exception as e:
                    logger.error(f"❌ Error generating video for scene {i+1}: {e}")
                    raise
            
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
            
            # Step 5: Create subtitles (optional) - accounting for pauses
            logger.info("Step 5: Creating subtitles...")
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
                logger.info("Step 5: Subtitles disabled; skipping SRT generation and overlay")
            
            # Step 6: Select background music
            logger.info("Step 6: Adding background music...")
            background_music = self._get_background_music()
            if background_music:
                logger.info(f"🎵 Background music selected: {background_music}")
            else:
                logger.info("🎵 No background music found - proceeding without music")
            
            # Step 7: Compile final video
            logger.info("Step 7: Compiling final video...")
            logger.info(f"🎬 Compiling {len(final_clips)} video clips...")
            logger.info(f"🎵 Using {len(final_audio_paths)} audio clips...")
            logger.info(f"📝 Subtitles: {'Enabled' if self.config.add_subtitles else 'Disabled'}")
            logger.info(f"🎵 Background music: {'Yes' if background_music else 'No'}")
            logger.info(f"📁 Final output: {final_output}")
            
            # Compile stitched video (keep existing output)
            stitched_output = self.output_dir / "stitched.mp4"
            self.video_processor.compile_final_video(
                final_clips,
                final_audio_paths,  # Pass audio paths with pauses included
                background_music,
                str(subtitles_path) if self.config.add_subtitles else None,
                str(stitched_output)
            )
            logger.info(f"✅ Stitched video created: {stitched_output}")
            
            # Step 8: Create platform-ready reel (if enabled)
            if self.config.create_reel:
                logger.info("Step 8: Creating platform-ready reel...")
                from ..render.reel_renderer import create_reel
                
                # Prepare voice audio (concatenate if multiple)
                voice_audio_path = None
                if final_audio_paths and any(final_audio_paths):
                    # If multiple audio paths, concatenate them
                    if len(final_audio_paths) > 1:
                        voice_audio_path = str(self.output_dir / "merged_voice.aac")
                        self.video_processor.concat_audios(
                            [p for p in final_audio_paths if p and Path(p).exists()],
                            voice_audio_path
                        )
                    else:
                        voice_audio_path = final_audio_paths[0] if final_audio_paths[0] else None
                
                # Use configured music path or fallback to background_music
                music_file = self.config.music_path or background_music
                
                reel_output = self.output_dir / "final_reel.mp4"
                try:
                    create_reel(
                        stitched_video=str(stitched_output),
                        voice_audio=voice_audio_path,
                        music_path=music_file,
                        output_path=str(reel_output),
                        vertical_mode=self.config.vertical_mode,
                        out_width=self.config.reel_width,
                        out_height=self.config.reel_height,
                        out_fps=self.config.reel_fps,
                        voice_volume=self.config.voice_volume,
                        music_volume=self.config.music_volume,
                        verbose=self.config.verbose_ffmpeg
                    )
                    logger.info(f"🎉 Platform-ready reel created: {reel_output}")
                    logger.info(f"📐 Format: {self.config.reel_width}x{self.config.reel_height} @ {self.config.reel_fps}fps")
                    logger.info(f"🎬 Mode: {self.config.vertical_mode}")
                    final_output = reel_output
                except Exception as e:
                    logger.error(f"❌ Failed to create reel: {e}")
                    logger.warning("⚠️ Falling back to stitched video")
                    final_output = stitched_output
            else:
                logger.info("Step 8: Reel creation disabled, using stitched video")
                final_output = stitched_output
            
            # Step 9: Generate metadata
            logger.info("Step 9: Generating metadata...")
            self._generate_metadata(script, str(final_output))
            
            logger.info(f"🎉 Video generation completed: {final_output}")
            logger.info(f"📊 Final video duration: {total_video_duration:.1f}s")
            logger.info(f"📊 Total processing time: Audio={total_audio_duration:.1f}s, Video={total_video_duration:.1f}s")
            if self.config.create_reel and final_output.name == "final_reel.mp4":
                logger.info(f"📤 Upload-ready: {final_output} (1080x1920, H.264/AAC, 30fps)")
            return str(final_output)
            
        except Exception as e:
            if self.config.skip_audio:
                logger.info("🔄 Audio generation skipped, using scene durations for video timing")
                # Use scene durations for video timing when audio is skipped
                actual_scene_durations = [scene.get('duration', self.config.scene_duration) for scene in script['scenes']]
                total_audio_duration = sum(actual_scene_durations)
                scene_audio_paths = [""] * len(script['scenes'])  # Empty audio paths
                logger.info(f"📊 Using scene durations: {actual_scene_durations}")
                logger.info(f"📊 Total scene duration: {total_audio_duration:.1f}s")
            else:
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
                    
                    # For fallback, use the first available voice file from scenes
                    fallback_voice_file = None
                    if script.get('scenes'):
                        from ..utils.narration_converter import NarrationConverter
                        converter = NarrationConverter()
                        for scene in script['scenes']:
                            if 'voice' in scene:
                                fallback_voice_file = converter.resolve_voice_file(scene['voice'])
                                if fallback_voice_file:
                                    logger.info(f"🎵 Fallback: Using voice file from first scene: {fallback_voice_file}")
                                    break
                    
                    generated_audio = self.voice_synthesizer.synthesize_voice(
                        narration_lines,
                        str(narration_path),
                        speaker=None,
                        voice_clone_audio=fallback_voice_file or self.config.voice_id or None,
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

    def _compose_video_prompt(self, scene: Dict) -> tuple[str, str]:
        """Compose a video prompt and negative prompt using the visual_prompt from script with character role integration."""
        # Get the visual_prompt from the script (this is the key requirement)
        visual_prompt = scene.get('visual_prompt', '')
        base_prompt = visual_prompt or "Cartoon scene"
        
        # Get the negative prompt from the script, or use default
        negative_prompt = scene.get('negative_prompt', self.config.wan_negative_prompt)

        # Enhance prompt with character role information if available
        enhanced_prompt = self._enhance_prompt_with_character_roles(base_prompt, scene)
        
        logger.info(f"🎯 Using enhanced visual_prompt with character roles: {enhanced_prompt}")
        return enhanced_prompt, negative_prompt
    
    def _enhance_prompt_with_character_roles(self, base_prompt: str, scene: Dict) -> str:
        """Enhance the visual prompt with character role information from the cast."""
        try:
            # Get characters mentioned in this scene
            scene_characters = scene.get('characters', [])
            if not scene_characters:
                return base_prompt
            
            # Get cast information if available
            cast_info = getattr(self, 'cast', [])
            if not cast_info:
                return base_prompt
            
            # Create a mapping of character names to their roles
            character_roles = {}
            for cast_member in cast_info:
                if isinstance(cast_member, dict):
                    name = cast_member.get('name', '')
                    role = cast_member.get('role', '')
                    if name and role:
                        character_roles[name] = role
            
            if not character_roles:
                return base_prompt
            
            # Find characters in this scene that have role information
            enhanced_prompt = base_prompt
            character_enhancements = []
            
            for character_name in scene_characters:
                if character_name in character_roles:
                    role = character_roles[character_name]
                    # Add role information to the prompt
                    character_enhancement = f"{character_name} ({role})"
                    character_enhancements.append(character_enhancement)
                    
                    # Replace character name with enhanced version in the prompt
                    # This helps the diffusion model understand the character's role
                    if character_name.lower() in base_prompt.lower():
                        # Replace the character name with role-enhanced version
                        enhanced_prompt = enhanced_prompt.replace(
                            character_name, 
                            character_enhancement
                        )
                    else:
                        # If character name not explicitly mentioned, add role info
                        enhanced_prompt = f"{enhanced_prompt}, {character_enhancement}"
            
            if character_enhancements:
                logger.info(f"🎭 Enhanced prompt with character roles: {character_enhancements}")
            
            return enhanced_prompt
            
        except Exception as e:
            logger.warning(f"⚠️ Error enhancing prompt with character roles: {e}")
            return base_prompt

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate platform-ready vertical Reels/Shorts videos")
    parser.add_argument("--prompt", required=True, help="Story prompt (e.g., 'A baby lion opens a smoothie shop in the jungle')")
    parser.add_argument("--duration", type=int, default=30, help="Video duration in seconds")
    parser.add_argument("--video-format", choices=["shorts", "normal"], default="shorts", 
                       help="Video format: 'shorts' for 9:16 YouTube Shorts, 'normal' for 16:9 standard videos")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--style", default="realistic", help="Visual style (realistic, anime, etc.)")
    parser.add_argument("--voice", default="", help="Reference speaker WAV path for Coqui XTTS (optional)")
    parser.add_argument("--language", default="en", help="Language for narration")
    parser.add_argument("--no-prompt-enhancement", action="store_true", help="Disable GPT-2 prompt enhancement")
    parser.add_argument("--scene-pause", type=float, default=0.0, help="Pause duration between scenes in seconds (default: 0.0, no black screens)")
    # Reel/Shorts rendering flags
    parser.add_argument("--format", choices=["reel", "normal"], default="reel", help="Output format: 'reel' for platform-ready vertical (default), 'normal' for stitched only")
    parser.add_argument("--vertical", action="store_true", help="Create vertical reel (same as --format reel, default)")
    parser.add_argument("--no-reel", action="store_true", help="Disable reel creation (use stitched video only)")
    parser.add_argument("--vertical-mode", choices=["pad", "crop"], default="pad", help="Vertical mode: 'pad' (safe, no cropping) or 'crop' (fills frame)")
    parser.add_argument("--out-width", type=int, default=1080, help="Reel output width (default: 1080)")
    parser.add_argument("--out-height", type=int, default=1920, help="Reel output height (default: 1920)")
    parser.add_argument("--out-fps", type=int, default=30, help="Reel output FPS (default: 30)")
    parser.add_argument("--music", type=str, default=None, help="Path to background music file (optional)")
    parser.add_argument("--music-volume", type=float, default=0.12, help="Background music volume (0.0-1.0, default: 0.12)")
    parser.add_argument("--voice-volume", type=float, default=1.0, help="Voice volume (0.0-1.0, default: 1.0)")
    parser.add_argument("--verbose-ffmpeg", action="store_true", help="Print FFmpeg commands for debugging")
    
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
    
    # Determine if reel should be created
    create_reel = not args.no_reel and (args.format == "reel" or args.vertical)
    
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
