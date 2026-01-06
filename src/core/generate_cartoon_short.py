#!/usr/bin/env python3
"""
Cartoon Shorts Generator - A complete CLI tool for creating platform-ready vertical Reels/Shorts videos.

This script follows a specific flow:
1. Generate 3-scene story with OpenAI GPT-4
2. Generate videos directly with WAN 2.2 Text-Image-to-Video (TI2V-5B)
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
from .gemini_image_generator import GeminiImageGenerator
from .sync_wav2_lip import lipsync_wav2lip


# Load environment variables (try .env first, then config.env as fallback)
load_dotenv()
if not os.getenv("GEMINI_API_KEY") and Path("config.env").exists():
    load_dotenv("config.env")

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


def postprocess_voice(input_wav: str, output_wav: str) -> str:
    filter_chain = "acompressor=threshold=-18dB:ratio=3:attack=20:release=250,highpass=f=80,lowpass=f=9000,equalizer=f=250:t=q:w=1:g=2,equalizer=f=3000:t=q:w=1:g=1.5,loudnorm=I=-16:TP=-1.5:LRA=11"
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", input_wav, "-af", filter_chain, "-y", output_wav],
            capture_output=True,
            text=True,
            shell=False
        )
        if result.returncode != 0:
            logger.error(f"ffmpeg postprocess failed: {result.stderr}")
            return input_wav
        return output_wav
    except Exception as e:
        logger.error(f"Voice postprocessing error: {e}")
        return input_wav

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
    # Control subtitle rendering (disabled by default, enable with --auto-sub)
    add_subtitles: bool = False
    # Control prompt enhancement
    enable_prompt_enhancement: bool = True
    # Control pause between scenes (in seconds)
    scene_pause_duration: float = 0.0  # Default 0.0 second pause between scenes (no black screens)
    # WAN 2.2 TI2V-5B settings (720p @ 24fps)
    wan_width: int = 1280  # WAN video width (720p)
    wan_height: int = 720  # WAN video height (720p)
    wan_num_frames: int = 72  # WAN number of frames to generate (3s @ 24fps)
    wan_fps: int = 24  # WAN output FPS (24fps for 720p)
    wan_steps: int = 30  # WAN inference steps
    wan_guidance: float = 6.0  # WAN guidance scale
    wan_negative_prompt: str = "text, subtitles, watermark, blurry, low quality, cartoon, anime, manga, illustration, painting, drawing, sketch, bad anatomy, distorted, deformed, ugly"  # WAN negative prompt for realistic videos (excludes non-realistic styles)
    wan_seed: Optional[int] = None  # WAN random seed (optional)
    # Audio settings
    skip_audio: bool = False  # Skip audio generation entirely
    # Hook text settings
    add_hooks: bool = False  # Enable hook text rendering (top_hook_text, bottom_hook_text, scene hook_text)
    top_hook_text: Optional[str] = None  # Top hook text to display on black area when upscaling
    bottom_hook_text: Optional[str] = None  # Bottom hook text to display on black area when upscaling
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
        
        # Initialize lightweight components only (no ML pipelines yet - will load after validation)
        self.script_generator = ScriptGenerator(os.getenv('OPENAI_API_KEY', ''))
        
        # Create video config for processor (lightweight)
        video_config = VPConfig(
            fps=config.fps,
            width=config.width,
            height=config.height
        )
        self.video_processor = VideoProcessor(video_config)
        
        # Pipelines will be initialized after validation (lazy loading to save VRAM)
        self.wan_generator = None
        self.voice_synthesizer = None
        self.gemini_generator = None
    
    def _initialize_pipelines(self):
        """Initialize ML pipelines (called after validation passes)."""
        if self.wan_generator is None:
            logger.info("🔄 Initializing WAN T2V generator...")
            self.wan_generator = WanT2VGenerator(
                width=self.config.wan_width,
                height=self.config.wan_height,
                num_frames=self.config.wan_num_frames,
                fps=self.config.wan_fps,
                num_inference_steps=self.config.wan_steps,
                guidance_scale=self.config.wan_guidance,
                negative_prompt=self.config.wan_negative_prompt
            )
        
        if self.voice_synthesizer is None:
            logger.info("🔄 Initializing Coqui TTS voice synthesizer...")
            self.voice_synthesizer = CoquiVoiceSynthesizer(
                CoquiVoiceConfig(language=self.config.language)
            )
        
        if self.gemini_generator is None:
            logger.info("🔄 Initializing Gemini image generator...")
            self.gemini_generator = GeminiImageGenerator()
        
    def generate(self) -> str:
        """Generate the complete video reel following the specified flow."""
        logger.info(f"Starting video generation for prompt: {self.config.prompt}")
        
        # Initialize video_clips early to avoid UnboundLocalError in exception handlers
        video_clips: List[str] = []
        
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
            original_storyboard = None  # Store original storyboard for saving later
            # Prioritize custom_scenes (from storyboard) over reusing existing script
            if self.config.custom_scenes and len(self.config.custom_scenes) > 0:
                logger.info("Step 1: Using custom storyboard scenes provided by user...")
                logger.info(f"📋 Found {len(self.config.custom_scenes)} custom scenes from storyboard")
                # Delete old script.json if it exists to force using storyboard
                if script_path.exists():
                    logger.info(f"🗑️ Removing old script.json to use storyboard instead")
                    try:
                        script_path.unlink()
                    except Exception as e:
                        logger.warning(f"Could not remove old script.json: {e}")
                
                # Store original storyboard structure (same as sent for reel creation)
                original_storyboard = {
                    "title": self.config.title or f"Story: {self.config.prompt}",
                    "description": self.config.description or f"An adventure about: {self.config.prompt}",
                    "scenes": self.config.custom_scenes
                }
                # Add total_duration if available from config
                if hasattr(self.config, 'duration') and self.config.duration:
                    original_storyboard["total_duration"] = self.config.duration
                # Add hook texts from config if present
                if self.config.top_hook_text:
                    original_storyboard["top_hook_text"] = self.config.top_hook_text
                if self.config.bottom_hook_text:
                    original_storyboard["bottom_hook_text"] = self.config.bottom_hook_text
                
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
                    if original_storyboard:
                        original_storyboard["total_duration"] = total_duration
                except Exception:
                    pass
                # Save script for reuse
                with open(script_path, 'w', encoding='utf-8') as f:
                    json.dump(script, f, indent=2)
                logger.info(f"✅ Generated script from storyboard with {len(script.get('scenes', []))} scenes")
            elif self.config.reuse_existing and script_path.exists():
                logger.info(f"Reusing existing script: {script_path}")
                with open(script_path, 'r', encoding='utf-8') as f:
                    script = json.load(f)
                try:
                    self.config.duration = int(script.get('total_duration', self.config.duration))
                except Exception:
                    pass
            else:
                logger.info("Step 1: Generating 3-scene story...")
                script = self.script_generator.generate_script(self.config.prompt, self.config.duration, language=self.config.language)
                with open(script_path, 'w', encoding='utf-8') as f:
                    json.dump(script, f, indent=2)
            
            # Save storyboard: use original storyboard if available (same as sent for reel creation),
            # otherwise use the generated script
            storyboard_path = self.output_dir / "storyboard.json"
            try:
                storyboard_to_save = original_storyboard if original_storyboard else script
                with open(storyboard_path, 'w', encoding='utf-8') as f:
                    json.dump(storyboard_to_save, f, indent=2)
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

            # Initialize pipelines
            self._initialize_pipelines()

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
                # Track which scenes have blank narration (will use 49 frames instead of calculating from audio)
                scenes_with_blank_narration = []
                
                for i, scene in enumerate(script['scenes']):
                    scene_audio = self.output_dir / f"audio_scene_{i+1}.wav"
                    narration_text = scene.get('narration', '')
                    
                    # Check if narration is blank (empty or only whitespace)
                    is_blank_narration = not narration_text or not narration_text.strip()
                    
                    if is_blank_narration:
                        logger.info(f"🎵 Scene {i+1}: Narration is blank - skipping TTS synthesis, will use 49 frames")
                        scenes_with_blank_narration.append(i)
                        # Create empty audio path for blank narration scenes
                        scene_audio_paths.append("")
                        # Calculate duration for 49 frames at wan_fps
                        blank_duration = 73 / self.config.wan_fps
                        actual_scene_durations.append(blank_duration)
                        logger.info(f"🎵 Scene {i+1}: Using default duration {blank_duration:.2f}s (49 frames @ {self.config.wan_fps}fps)")
                        continue
                    
                    logger.info(f"🎵 Scene {i+1}: Processing narration ({len(narration_text)} characters)")
                    
                    # Get voice file from scene if available
                    # Priority: scene voice > --voice argument > None
                    voice_file = None
                    scene_voice_value = scene.get('voice')
                    
                    if scene_voice_value and scene_voice_value.strip():
                        # Scene has a voice property - try to resolve it
                        from ..utils.narration_converter import NarrationConverter
                        converter = NarrationConverter()
                        voice_file = converter.resolve_voice_file(scene_voice_value.strip())
                        if voice_file:
                            logger.info(f"🎵 Scene {i+1}: Using scene voice: '{scene_voice_value}' -> {voice_file}")
                        else:
                            logger.warning(f"🎵 Scene {i+1}: Scene voice '{scene_voice_value}' not found, will fallback to --voice argument")
                    else:
                        logger.info(f"🎵 Scene {i+1}: No voice property in scene, will use --voice argument if provided")
                    
                    if not (self.config.reuse_existing and scene_audio.exists()):
                        logger.info(f"🎵 Scene {i+1}: Generating new audio clip...")
                        # Determine final voice file: scene voice > --voice argument > None
                        final_voice_file = None
                        if voice_file:
                            # Use scene voice if successfully resolved
                            final_voice_file = voice_file
                        elif self.config.voice_id and self.config.voice_id.strip():
                            # Fallback to --voice argument
                            final_voice_file = self.config.voice_id.strip()
                            logger.info(f"🎵 Scene {i+1}: Using --voice argument: {final_voice_file}")
                        else:
                            logger.info(f"🎵 Scene {i+1}: No voice specified (using default TTS voice)")
                        
                        logger.info(f"🎵 Scene {i+1}: Final voice selection: {final_voice_file or 'default TTS voice'}")
                        generated_audio = self.voice_synthesizer.synthesize_voice(
                            [narration_text],
                            str(scene_audio),
                            speaker=None,
                            voice_clone_audio=final_voice_file,
                        )
                        # Use actual generated path (may switch extension on fallback)
                        scene_audio = Path(generated_audio)
                        logger.info(f"🎵 Scene {i+1}: Audio generation completed: {scene_audio}")
                        processed_audio = postprocess_voice(str(scene_audio), str(scene_audio.parent / f"{scene_audio.stem}_pp{scene_audio.suffix}"))
                        scene_audio = Path(processed_audio)
                    else:
                        logger.info(f"🎵 Scene {i+1}: Reusing existing audio: {scene_audio}")
                    
                    scene_audio_paths.append(str(scene_audio))
                    # Duration will be detected in the next step
                    actual_scene_durations.append(0.0)  # Placeholder, will be updated
                    logger.info(f"Scene {i+1}: Audio clip ready: {scene_audio}")
            
            # Detect length of each audio clip (no validation - will set num_frames based on audio length)
            if not self.config.skip_audio:
                logger.info("📏 Detecting length of each audio clip...")
                total_audio_duration = 0
                
                for i, scene_audio in enumerate(scene_audio_paths):
                    scene_num = i + 1
                    scene = script['scenes'][i]
                    
                    # Skip blank narration scenes (already handled above with duration set)
                    if i in scenes_with_blank_narration:
                        logger.info(f"📏 Scene {scene_num}: Blank narration - using pre-calculated duration ({actual_scene_durations[i]:.2f}s)")
                        total_audio_duration += actual_scene_durations[i]
                        continue
                    
                    logger.info(f"📏 Scene {scene_num}: Analyzing audio duration...")
                    actual_duration = self.video_processor.get_audio_duration(scene_audio)
                    # Update the duration at the correct index
                    actual_scene_durations[i] = actual_duration
                    total_audio_duration += actual_duration
                    logger.info(f"Scene {scene_num}: Audio clip length: {actual_duration:.1f}s")
                
                logger.info(f"✅ Generated {len(scene_audio_paths)} audio clips for narration")
                logger.info(f"📊 Total audio duration: {total_audio_duration:.1f}s")
                logger.info(f"📊 Average audio duration per scene: {total_audio_duration/len(actual_scene_durations):.1f}s")
                
                # Move Coqui TTS pipeline to CPU after audio generation to free VRAM for WAN model
                logger.info("💾 Moving Coqui TTS pipeline to CPU to free VRAM for video generation...")
                try:
                    self.voice_synthesizer.move_to_cpu()
                    logger.info("✅ Coqui TTS pipeline moved to CPU (will be moved back to GPU if needed)")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to move TTS pipeline to CPU: {e}")
            else:
                # Calculate total duration from scene durations
                total_audio_duration = sum(actual_scene_durations)
                logger.info(f"✅ Skipped audio generation")
                logger.info(f"📊 Total scene duration: {total_audio_duration:.1f}s")
                logger.info(f"📊 Average scene duration: {total_audio_duration/len(actual_scene_durations):.1f}s")
            
            # Step 3: Generate videos directly from prompts using WAN T2V
            logger.info("Step 3: Generating videos with WAN 2.2 TI2V-5B...")
            logger.info(f"🎬 Total videos to generate: {len(script['scenes'])}")
            total_video_duration = 0
            
            # Create scenes subdirectory
            scenes_dir = self.output_dir / "scenes"
            scenes_dir.mkdir(parents=True, exist_ok=True)
            
            # Track previous image for character consistency across scenes
            previous_image_path = None
            
            for i, scene in enumerate(script['scenes']):
                clip_path = scenes_dir / f"scene_{i+1}.mp4"
                logger.info(f"🎬 Scene {i+1}: Processing video generation...")
                
                if self.config.reuse_existing and clip_path.exists():
                    logger.info(f"⏭️  Reusing existing video: {clip_path}")
                    existing_video_path = str(clip_path)
                    existing_duration = self.video_processor.get_video_duration(existing_video_path)
                    
                    # If audio exists, check if video needs syncing
                    if not self.config.skip_audio and i < len(actual_scene_durations):
                        target_audio_duration = actual_scene_durations[i]
                        duration_diff = abs(existing_duration - target_audio_duration)
                        if duration_diff > 0.1:
                            logger.info(f"🔄 Scene {i+1}: Existing video ({existing_duration:.2f}s) doesn't match audio ({target_audio_duration:.2f}s), syncing...")
                            synced_video_path = str(clip_path).replace('.mp4', '_synced.mp4')
                            if existing_duration < target_audio_duration:
                                self.video_processor.extend_video_duration(
                                    existing_video_path,
                                    target_audio_duration,
                                    synced_video_path
                                )
                            else:
                                cmd = [
                                    'ffmpeg', '-y',
                                    '-i', existing_video_path,
                                    '-t', str(target_audio_duration),
                                    '-c', 'copy',
                                    synced_video_path
                                ]
                                subprocess.run(cmd, check=True, capture_output=True, text=True)
                            existing_video_path = synced_video_path
                            existing_duration = target_audio_duration
                    
                    video_clips.append(existing_video_path)
                    total_video_duration += existing_duration
                    continue
                
                # Compose prompt from scene
                prompt, negative_prompt = self._compose_video_prompt(scene)
                logger.info(f"🎬 Scene {i+1}: Generating video with prompt ({len(prompt)} characters)")
                logger.info(f"🎬 Scene {i+1}: Prompt preview: {prompt[:100]}...")
                if negative_prompt:
                    logger.info(f"🎬 Scene {i+1}: Using negative prompt ({len(negative_prompt)} characters)")
                
                # Generate initial frame with Gemini using visual_prompt
                initial_image_path = None
                visual_prompt = scene.get('visual_prompt', '')
                if visual_prompt:
                    if self.gemini_generator.available:
                        initial_image_path = self.output_dir / f"scene_{i+1}_initial_frame.png"
                        logger.info(f"🎨 Scene {i+1}: Generating initial frame with Gemini...")
                        
                        # Check if scene explicitly disables previous image reference
                        prev_image_ref = scene.get('prev_image_ref', None)
                        reference_image_to_use = None
                        
                        if prev_image_ref is False:
                            # Explicitly disabled - don't use previous image
                            logger.info(f"🖼️ Scene {i+1}: prev_image_ref=false - not using previous image as reference")
                            reference_image_to_use = None
                        elif previous_image_path:
                            # Use previous image if available (default behavior when prev_image_ref is missing/null)
                            logger.info(f"🖼️ Scene {i+1}: Using previous image as reference for character consistency")
                            reference_image_to_use = previous_image_path
                        
                        generated_image = self.gemini_generator.generate_image(
                            prompt=visual_prompt,
                            output_path=str(initial_image_path),
                            width=self.config.wan_width,
                            height=self.config.wan_height,
                            reference_image_path=reference_image_to_use
                        )
                        if generated_image:
                            initial_image_path = generated_image
                            # Update previous_image_path for next scene
                            previous_image_path = generated_image
                            logger.info(f"✅ Scene {i+1}: Initial frame generated: {initial_image_path}")
                        else:
                            logger.warning(f"⚠️ Scene {i+1}: Failed to generate initial frame, using T2V mode")
                            initial_image_path = None
                    else:
                        logger.warning(f"⚠️ Scene {i+1}: Gemini not available (check GEMINI_API_KEY and google-genai package), using T2V mode")
                        initial_image_path = None
                
                # Combine motion_prompt with visual_prompt for WAN if available
                motion_prompt = scene.get('motion_prompt', '')
                if motion_prompt and visual_prompt:
                    prompt = f"{visual_prompt}. Motion: {motion_prompt}"
                    logger.info(f"🎬 Scene {i+1}: Combined visual_prompt + motion_prompt for WAN")
                elif motion_prompt:
                    prompt = f"{prompt}. Motion: {motion_prompt}"
                    logger.info(f"🎬 Scene {i+1}: Added motion_prompt to prompt")
                
                # Generate video with WAN
                try:
                    # Calculate num_frames based on audio duration if available, otherwise use scene/config values
                    num_frames_to_use = None
                    
                    # Check if narration is blank - if so, use 49 frames
                    narration_text = scene.get('narration', '')
                    is_blank_narration = not narration_text or not narration_text.strip()
                    
                    if is_blank_narration:
                        # Use 49 frames for blank narration scenes
                        num_frames_to_use = 49
                        logger.info(f"🎬 Scene {i+1}: Blank narration - using 49 frames")
                    # If audio exists, calculate num_frames from actual audio duration
                    elif not self.config.skip_audio and i < len(actual_scene_durations):
                        actual_duration = actual_scene_durations[i]
                        scene_fps = self.config.wan_fps
                        # Calculate num_frames needed to match audio duration
                        num_frames_to_use = int(actual_duration * scene_fps)
                        logger.info(f"🎬 Scene {i+1}: Setting num_frames to {num_frames_to_use} based on audio duration ({actual_duration:.2f}s @ {scene_fps}fps)")
                    else:
                        # Fallback to scene num_frames or config
                        scene_num_frames = scene.get('num_frames', None)
                        if scene_num_frames is None:
                            # Check generation_profile for num_frames
                            generation_profile = script.get('generation_profile', {})
                            scene_num_frames = generation_profile.get('num_frames', None)
                        
                        # Use scene num_frames if available, otherwise use command-line default
                        num_frames_to_use = scene_num_frames if scene_num_frames is not None else self.config.wan_num_frames
                        
                        if scene_num_frames is not None:
                            logger.info(f"🎬 Scene {i+1}: Using num_frames from scene: {scene_num_frames}")
                        else:
                            logger.info(f"🎬 Scene {i+1}: Using num_frames from command-line: {num_frames_to_use}")
                    
                    # Extract scene metadata
                    scene_id = scene.get('id', f"scene_{i+1}")
                    visual_reference = scene.get('visual_reference', None)
                    # Generate slug from story title (sanitized for filename)
                    story_title = script.get('title', 'story')
                    slug = "".join(c for c in story_title if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_').lower()[:50]
                    
                    # Generate video
                    # Pass initial_image_path to WAN for I2V mode if available
                    result = self.wan_generator.generate_video(
                        prompt=prompt,
                        output_path=str(clip_path),
                        seed=self.config.wan_seed,
                        negative_prompt=negative_prompt or None,
                        num_frames=num_frames_to_use,  # Use num_frames from scene or command-line
                        scene_id=scene_id,
                        visual_reference=visual_reference,
                        slug=slug,
                        image=initial_image_path if initial_image_path else None
                    )
                    
                    # Handle return type: dict (with metadata) or string (backward compatible)
                    if isinstance(result, dict):
                        video_path = result['video_path']
                        # Store metadata for downstream article generation
                        scene['scene_id'] = result.get('scene_id')
                        scene['visual_reference'] = result.get('visual_reference')
                    else:
                        # Backward compatibility: result is a string
                        video_path = result
                    
                    # Get actual video duration
                    actual_duration = self.video_processor.get_video_duration(str(video_path))
                    logger.info(f"✅ Scene {i+1}: Video generated ({actual_duration:.2f}s)")
                    
                    # If we have narration, sync video to match audio clip length exactly
                    if not self.config.skip_audio and i < len(actual_scene_durations):
                        target_audio_duration = actual_scene_durations[i]
                        duration_diff = abs(actual_duration - target_audio_duration)
                        
                        if duration_diff > 0.1:  # If difference > 0.1s, sync them
                            logger.info(f"🎬 Scene {i+1}: Syncing video ({actual_duration:.2f}s) to match audio ({target_audio_duration:.2f}s)")
                            synced_video_path = str(clip_path).replace('.mp4', '_synced.mp4')
                            
                            if actual_duration < target_audio_duration:
                                # Video is shorter than audio - extend by looping
                                self.video_processor.extend_video_duration(
                                    str(video_path),
                                    target_audio_duration,
                                    synced_video_path
                                )
                                logger.info(f"🎬 Scene {i+1}: Extended video from {actual_duration:.2f}s to {target_audio_duration:.2f}s")
                            else:
                                # Video is longer than audio - trim to match
                                cmd = [
                                    'ffmpeg', '-y',
                                    '-i', str(video_path),
                                    '-t', str(target_audio_duration),
                                    '-c', 'copy',
                                    synced_video_path
                                ]
                                subprocess.run(cmd, check=True, capture_output=True, text=True)
                                logger.info(f"🎬 Scene {i+1}: Trimmed video from {actual_duration:.2f}s to {target_audio_duration:.2f}s")
                            
                            video_path = synced_video_path
                            actual_duration = target_audio_duration
                            logger.info(f"✅ Scene {i+1}: Video synced to audio length ({actual_duration:.2f}s)")
                        else:
                            logger.info(f"✅ Scene {i+1}: Video duration ({actual_duration:.2f}s) already matches audio ({target_audio_duration:.2f}s)")
                    
                    # Run lip sync if audio is available and scene has lip_sync enabled
                    scene_lip_sync = scene.get('lip_sync', False)
                    
                    if not self.config.skip_audio and i < len(scene_audio_paths) and scene_audio_paths[i]:
                        if scene_lip_sync:
                            # Run Wav2Lip - uses its own virtual environment Python
                            wav2lip_enabled = os.getenv("WAV2LIP_ENABLED", "true").lower() in ("true", "1", "yes")
                            
                            if wav2lip_enabled:
                                try:
                                    wav2lip_output = str(clip_path).replace('.mp4', '_lipsync.mp4')
                                    logger.info(f"🎙️ Scene {i+1}: lip_sync=ON → Running Wav2Lip...")
                                    
                                    # Pass None to use default paths from sync_wav2_lip.py (which uses env vars or hardcoded defaults)
                                    success = lipsync_wav2lip(
                                        in_video_mp4=str(video_path),
                                        in_audio_wav=scene_audio_paths[i],
                                        out_video_mp4=wav2lip_output,
                                        fps=self.config.wan_fps,
                                        wav2lip_dir=None,  # Use default from sync_wav2_lip.py
                                        checkpoint_path=None,  # Use default from sync_wav2_lip.py
                                        python_cmd=None  # Use Wav2Lip venv Python (default: /workspace/Wav2Lip/venv/bin/python)
                                    )
                                    
                                    if success:
                                        video_path = wav2lip_output
                                        logger.info(f"Scene {i+1}: lip_sync=ON → SUCCESS ({Path(wav2lip_output).name})")
                                    else:
                                        reason = "Wav2Lip inference failed"
                                        logger.warning(f"Scene {i+1}: lip_sync=ON → FAILED (fallback to {Path(video_path).name}): {reason}")
                                except Exception as e:
                                    reason = f"Error: {str(e)}"
                                    logger.warning(f"Scene {i+1}: lip_sync=ON → FAILED (fallback to {Path(video_path).name}): {reason}")
                            else:
                                logger.warning(f"⚠️ Scene {i+1}: WAV2LIP_ENABLED is not set to 'true'. Skipping lip sync.")
                                logger.info(f"Scene {i+1}: lip_sync=ON but WAV2LIP_ENABLED=false → skipped")
                        else:
                            logger.info(f"Scene {i+1}: lip_sync=OFF → skipped")
                    
                    video_clips.append(video_path)
                    total_video_duration += actual_duration
                    logger.info(f"🎬 Scene {i+1}: Video ready: {video_path} ({actual_duration:.1f}s)")
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
            try:
                self.video_processor.compile_final_video(
                    final_clips,
                    final_audio_paths,  # Pass audio paths with pauses included
                    background_music,
                    str(subtitles_path) if self.config.add_subtitles else None,
                    str(stitched_output)
                )
                logger.info(f"✅ Stitched video created: {stitched_output}")
            except Exception as e:
                logger.error(f"❌ Failed to create stitched video: {e}")
                # Clean up corrupted file if it exists
                if stitched_output.exists():
                    try:
                        stitched_output.unlink()
                        logger.info("🧹 Removed corrupted stitched video file")
                    except Exception:
                        pass
                raise
            
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
                
                # Extract hook texts from storyboard
                top_hook = None
                bottom_hook = None
                scene_hooks = []  # List of (start_time, end_time, hook_text) tuples
                
                if self.config.add_hooks:
                    # Try to load storyboard to get hook texts
                    storyboard_path = self.output_dir / "storyboard.json"
                    if storyboard_path.exists():
                        try:
                            with open(storyboard_path, 'r', encoding='utf-8') as f:
                                storyboard_data = json.load(f)
                            top_hook = storyboard_data.get('top_hook_text')
                            bottom_hook = storyboard_data.get('bottom_hook_text')
                        except Exception as e:
                            logger.warning(f"Could not load storyboard for hook texts: {e}")
                    
                    # Extract from config if not in storyboard
                    if not top_hook:
                        top_hook = self.config.top_hook_text
                    if not bottom_hook:
                        bottom_hook = self.config.bottom_hook_text
                
                # Extract scene hook_text from script (always, not just when add_hooks is True)
                for i, scene in enumerate(script['scenes']):
                    hook_text = scene.get('hook_text')
                    if hook_text:
                        # Calculate timing for this scene (accounting for pauses between scenes)
                        scene_duration = scene.get('duration', 8)
                        start_time = sum(s.get('duration', 8) for s in script['scenes'][:i])
                        # Add pause duration for each previous scene (except before first scene)
                        start_time += self.config.scene_pause_duration * i
                        end_time = start_time + scene_duration
                        scene_hooks.append((start_time, end_time, hook_text))
                
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
                        verbose=self.config.verbose_ffmpeg,
                        add_hooks=self.config.add_hooks,
                        top_hook_text=top_hook,
                        bottom_hook_text=bottom_hook,
                        scene_hooks=scene_hooks
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
            logger.error(f"❌ Error in video generation: {e}")
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

        # Use raw prompt without enhancement
        logger.info(f"🎯 Using raw visual_prompt: {base_prompt}")
        return base_prompt, negative_prompt
    

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate platform-ready vertical Reels/Shorts videos")
    parser.add_argument("--prompt", required=False, help="Story prompt (e.g., 'A baby lion opens a smoothie shop in the jungle') - not required with --gen-bulk")
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
    parser.add_argument("--auto-sub", action="store_true", help="Enable automatic subtitle generation (disabled by default)")
    parser.add_argument("--add-hooks", action="store_true", help="Enable hook text rendering (top_hook_text, bottom_hook_text from storyboard, and scene hook_text)")
    
    # Bulk generation arguments
    parser.add_argument("--gen-bulk", action="store_true", help="Enable bulk generation from a single storyboard JSON file with multiple stories")
    parser.add_argument("--storyboard", type=str, help="Path to storyboard JSON file containing a 'stories' array (required with --gen-bulk)")
    
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
    
    # Bulk generation mode
    if args.gen_bulk:
        if not args.storyboard:
            logger.error("❌ --storyboard is required when using --gen-bulk")
            sys.exit(1)
        
        storyboard_file = Path(args.storyboard)
        if not storyboard_file.exists() or not storyboard_file.is_file():
            logger.error(f"❌ Storyboard file does not exist: {storyboard_file}")
            sys.exit(1)
        
        # Load the storyboard JSON file
        try:
            with open(storyboard_file, 'r', encoding='utf-8') as f:
                storyboard_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in storyboard file: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"❌ Failed to read storyboard file: {e}")
            sys.exit(1)
        
        # Extract stories array
        stories = storyboard_data.get('stories', [])
        if not stories:
            logger.error(f"❌ No 'stories' array found in storyboard file. Expected format: {{'stories': [...]}}")
            sys.exit(1)
        
        logger.info(f"📁 Found {len(stories)} stories in {storyboard_file.name}")
        
        # Initialize pipelines once (singleton pattern ensures they're shared)
        logger.info("🔄 Initializing pipelines (will be reused for all stories)...")
        base_config = VideoConfig(
            prompt="",  # Will be overridden per story
            duration=args.duration,
            video_format=args.video_format,
            output_path=args.output,
            style=args.style,
            voice_id=args.voice,
            language=args.language,
            enable_prompt_enhancement=False,
            scene_pause_duration=args.scene_pause,
            add_subtitles=args.auto_sub,  # Only enable if --auto-sub is provided
            add_hooks=args.add_hooks,  # Enable hook text rendering
            create_reel=not args.no_reel and (args.format == "reel" or args.vertical),
            vertical_mode=args.vertical_mode,
            reel_width=args.out_width,
            reel_height=args.out_height,
            reel_fps=args.out_fps,
            music_path=args.music,
            music_volume=args.music_volume,
            voice_volume=args.voice_volume,
            verbose_ffmpeg=args.verbose_ffmpeg
        )
        
        # Pre-initialize generator to load pipelines once (WAN and TTS use singletons)
        logger.info("📦 Loading WAN pipeline (singleton - will be reused)...")
        logger.info("📦 Loading TTS model (singleton - will be reused)...")
        temp_generator = CartoonShortsGenerator(base_config)
        logger.info("✅ Pipelines initialized and ready for bulk generation")
        
        # Process each story in the stories array
        successful = []
        failed = []
        
        for i, story in enumerate(stories, 1):
            logger.info(f"\n{'='*70}")
            logger.info(f"Processing story {i}/{len(stories)}")
            logger.info(f"{'='*70}")
            
            try:
                # Use story title if available, otherwise use top-level title, or fallback to Story_{i}
                title = story.get('title') or storyboard_data.get('title') or f'Story_{i}'
                description = story.get('description', '') or storyboard_data.get('description', '')
                scenes = story.get('scenes', [])
                total_duration = story.get('total_duration', args.duration)
                
                if not scenes:
                    logger.warning(f"⚠️ No scenes found in story {i}, skipping")
                    failed.append((title, "No scenes found"))
                    continue
                
                # Calculate scene duration if not provided in each scene
                # Use total_duration divided by number of scenes, or use scene's own duration
                for scene in scenes:
                    if 'duration' not in scene:
                        scene['duration'] = total_duration // len(scenes)
                
                # Create output folder by title (sanitize filename)
                safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
                safe_title = safe_title.replace(' ', '_')[:50]  # Limit length
                if not safe_title:
                    safe_title = f"Story_{i}"  # Fallback to index
                output_folder = Path(args.output) / safe_title
                output_folder.mkdir(parents=True, exist_ok=True)
                
                logger.info(f"📁 Output folder: {output_folder}")
                logger.info(f"📋 Title: {title}")
                logger.info(f"📝 Description: {description}")
                logger.info(f"🎬 Scenes: {len(scenes)}")
                logger.info(f"⏱️ Total duration: {total_duration}s")
                
                # Normalize scenes (handle characters if present)
                # Check if there's a cast at the story level or file level
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
                    scene_copy['characters'] = structured_chars[:2]  # Limit to 2 characters
                    normalized_scenes.append(scene_copy)
                
                # Update generator config and output directory (reuse same instance)
                # This ensures WAN and TTS pipelines are truly reused (not reloaded)
                temp_generator.config.prompt = title
                temp_generator.config.title = title
                temp_generator.config.description = description
                temp_generator.config.custom_scenes = normalized_scenes
                temp_generator.config.duration = total_duration
                # Use scene duration from first scene if available, otherwise calculate
                if normalized_scenes and 'duration' in normalized_scenes[0]:
                    temp_generator.config.scene_duration = normalized_scenes[0].get('duration', 8)
                else:
                    temp_generator.config.scene_duration = total_duration // len(normalized_scenes) if normalized_scenes else 8
                temp_generator.config.output_path = str(output_folder)
                temp_generator.output_dir = output_folder
                temp_generator.output_dir.mkdir(parents=True, exist_ok=True)
                
                # Update voice synthesizer language if needed
                if args.language != temp_generator.config.language:
                    temp_generator.config.language = args.language
                    # Note: TTS model is already loaded, language change will be handled during synthesis
                
                # Generate video using the same generator instance
                output_path = temp_generator.generate()
                successful.append((title, output_path))
                logger.info(f"✅ Successfully generated: {output_path}")
                
            except Exception as e:
                logger.error(f"❌ Failed to process story {i} ({title if 'title' in locals() else 'Unknown'}): {e}")
                import traceback
                logger.error(traceback.format_exc())
                failed.append((title if 'title' in locals() else f"Story_{i}", str(e)))
        
        # Print summary
        logger.info(f"\n{'='*70}")
        logger.info("BULK GENERATION SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"✅ Successful: {len(successful)}/{len(stories)}")
        for title, output in successful:
            logger.info(f"   ✓ {title} -> {output}")
        
        if failed:
            logger.info(f"\n❌ Failed: {len(failed)}/{len(stories)}")
            for title, error in failed:
                logger.info(f"   ✗ {title}: {error}")
        
        logger.info(f"{'='*70}")
        sys.exit(0 if not failed else 1)
    
    # Single generation mode (existing logic)
    if not args.prompt:
        logger.error("❌ --prompt is required when not using --gen-bulk")
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
        enable_prompt_enhancement=False,  # Prompt enhancement disabled
        scene_pause_duration=args.scene_pause,
        add_subtitles=args.auto_sub,  # Only enable if --auto-sub is provided
        add_hooks=args.add_hooks,  # Enable hook text rendering
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
