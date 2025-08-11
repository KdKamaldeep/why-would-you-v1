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
            self.image_generator = ImageGenerator(model_path=model_path, lora_path=lora_path, lora_scale=0.85)
        except TypeError:
            # Fallback for older ImageGenerator signature
            self.image_generator = ImageGenerator(model_path=model_path)
        self.animation_generator = AnimationGenerator()
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
            final_output = self.output_dir / "final_short.mp4"
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

            # Step 2: Generate cartoon images with Stable Diffusion
            logger.info("Step 2: Generating cartoon images...")
            image_paths = []
            for i, scene in enumerate(script['scenes']):
                image_path = self.output_dir / f"scene_{i+1}.png"
                if self.config.reuse_existing and image_path.exists():
                    logger.info(f"Skipping image generation (exists): {image_path}")
                else:
                    prompt = self._compose_image_prompt(scene)
                    self.image_generator.generate_cartoon_image(prompt, str(image_path))
                image_paths.append(str(image_path))
            
            # Step 3: Animate images with AnimateDiff
            logger.info("Step 3: Creating professional quality animations...")
            # Calculate frames needed for each scene based on duration
            scene_prompts = [scene['visual_prompt'] for scene in script['scenes']]
            scene_durations = [scene['duration'] for scene in script['scenes']]
            
            # Calculate frames per scene: duration * fps
            # Note: Enhanced animation system supports unlimited length!
            frames_per_scene = []
            for duration in scene_durations:
                # Professional quality animation with proper timing
                total_frames = max(30, int(duration * self.config.fps))  # Minimum 2 seconds per scene
                frames_per_scene.append(total_frames)
            
            logger.info(f"Scene durations: {scene_durations} seconds")
            logger.info(f"Frames per scene: {frames_per_scene}")
            logger.info("🎬 Using enhanced animation system (unlimited length capability)")
            
            frame_dirs: List[str] = []
            for i, image_path in enumerate(image_paths):
                frames_dir = self.output_dir / f"scene_{i+1}_frames"
                expected_frames = frames_per_scene[i]
                if self.config.reuse_existing and frames_dir.exists():
                    # Count frames
                    existing = list(frames_dir.glob("frame_*.png"))
                    if len(existing) >= expected_frames:
                        logger.info(f"Skipping animation (frames ready): {frames_dir} ({len(existing)} frames)")
                        frame_dirs.append(str(frames_dir))
                        continue
                # Generate frames
                dir_path = self.animation_generator.animate_image(
                    image_path,
                    str(frames_dir),
                    num_frames=expected_frames,
                    prompt=scene_prompts[i] if i < len(scene_prompts) else ""
                )
                frame_dirs.append(dir_path)
            
            # Step 4: Convert frames to MP4 videos
            logger.info("Step 4: Converting frames to videos...")
            video_clips: List[str] = []
            for i, frames_dir in enumerate(frame_dirs):
                clip_path = self.output_dir / f"scene_{i+1}.mp4"
                if self.config.reuse_existing and clip_path.exists():
                    logger.info(f"Skipping frames->video (exists): {clip_path}")
                    video_clips.append(str(clip_path))
                    continue
                video_path = self.video_processor.frames_to_video(frames_dir, str(clip_path), fps=self.config.fps)
                video_clips.append(video_path)
            
            # Step 5: Generate narration with Coqui TTS
            logger.info("Step 5: Generating narration...")
            narration_path = self.output_dir / "narration.wav"
            scene_audio_paths = []
            actual_scene_durations = []  # Track actual audio durations
            
            try:
                # Generate per-scene audio and calculate actual durations
                for i, scene in enumerate(script['scenes']):
                    scene_audio = self.output_dir / f"audio_scene_{i+1}.wav"
                    if not (self.config.reuse_existing and scene_audio.exists()):
                        generated_audio = self.voice_synthesizer.synthesize_voice(
                            [scene.get('narration', '')],
                            str(scene_audio),
                            speaker=None,
                            voice_clone_audio=self.config.voice_id or None,
                        )
                        # Use actual generated path (may switch extension on fallback)
                        scene_audio = Path(generated_audio)
                    
                    # Calculate actual audio duration
                    actual_duration = self.video_processor.get_audio_duration(str(scene_audio))
                    actual_scene_durations.append(actual_duration)
                    
                    # Update scene duration to match actual audio length
                    script['scenes'][i]['duration'] = actual_duration
                    
                    logger.info(f"Scene {i+1}: Script duration {scene.get('duration', 8):.1f}s, Actual audio: {actual_duration:.1f}s")
                    
                    # Fit each scene audio to scene duration (now using actual duration)
                    fitted_audio = self.output_dir / f"audio_scene_{i+1}_fit.m4a"
                    self.video_processor.adjust_audio_to_duration(str(scene_audio), actual_duration, str(fitted_audio))
                    scene_audio_paths.append(str(fitted_audio))
                
                # Combine per-scene into one track
                merged_audio = self.output_dir / "narration_fitted.m4a"
                self.video_processor.concat_audios(scene_audio_paths, str(merged_audio))
                narration_path = merged_audio
                
                # Update total duration based on actual audio lengths
                total_actual_duration = sum(actual_scene_durations)
                self.config.duration = max(self.config.duration, total_actual_duration)
                script['total_duration'] = total_actual_duration
                
                logger.info(f"Updated total duration: {total_actual_duration:.1f}s (was {self.config.duration}s)")
                logger.info("✅ Narration timing optimized - video will match actual audio length")
                
            except Exception as e:
                logger.warning(f"Per-scene audio fitting failed; falling back to single track: {e}")
                if not (self.config.reuse_existing and narration_path.exists()):
                    narration_lines = [scene.get('narration', '') for scene in script.get('scenes', [])]
                    generated_audio = self.voice_synthesizer.synthesize_voice(
                        narration_lines,
                        str(narration_path),
                        speaker=None,
                        voice_clone_audio=self.config.voice_id or None,
                    )
                    narration_path = Path(generated_audio)
                
                # Calculate actual duration of the single track
                actual_duration = self.video_processor.get_audio_duration(str(narration_path))
                self.config.duration = max(self.config.duration, actual_duration)
                script['total_duration'] = actual_duration
                logger.info(f"Single track duration: {actual_duration:.1f}s")
                logger.info("✅ Narration timing optimized - video will match actual audio length")
            
            # Step 6: Use video clips directly (no lip-sync)
            logger.info("Step 6: Preparing video clips...")
            final_clips = video_clips
            
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
                str(narration_path) if isinstance(narration_path, (str, Path)) else narration_path,
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

    def _compose_image_prompt(self, scene: Dict) -> str:
        """Compose an image prompt that bakes in exactly two character specs if available."""
        base = scene.get('visual_prompt', scene.get('description', ''))
        characters = scene.get('characters', [])
        # Style preset additions
        style_suffix = ""
        style_key = (self.config.style or "").lower()
        if style_key in {"indian", "indian_cartoon", "desi", "bollywood"}:
            style_suffix = (
                " Indian cartoon style, inspired by Indian children's book illustrations and Amar Chitra Katha; "
                "vibrant festive palette (marigold, vermilion, indigo), matte shading, soft outlines; "
                "traditional Indian clothing and accessories where natural (kurta, sari, bangles); "
                "background motifs like bazaars, auto-rickshaws, kites, forts or temples when relevant; "
                "warm sunlight, friendly expression, family-friendly, avoid anime/manga aesthetics."
            )
        else:
            style_suffix = (
                " Vertical 768x1024 cartoon, clean lines, vibrant colors, family-friendly, both characters clearly visible, consistent traits across scenes."
            )
        if characters:
            char_bits = []
            for idx, ch in enumerate(characters[:2], start=1):
                part = (
                    f"({idx}) {ch.get('name','Character')} - {ch.get('role','role')}; "
                    f"appearance: {ch.get('appearance','consistent look')}; "
                    f"clothing: {ch.get('clothing','simple outfit')}; "
                    f"emotion: {ch.get('emotion','neutral')}; "
                    f"action: {ch.get('action','standing')}"
                )
                char_bits.append(part)
            char_text = " Include two characters: " + " | ".join(char_bits) + "."
        else:
            char_text = ""
        return (base or "Cartoon scene") + char_text + style_suffix

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate cartoon-style YouTube Shorts videos")
    parser.add_argument("--prompt", required=True, help="Story prompt (e.g., 'A baby lion opens a smoothie shop in the jungle')")
    parser.add_argument("--duration", type=int, default=30, help="Video duration in seconds")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--style", default="cartoon", help="Visual style")
    parser.add_argument("--voice", default="", help="Reference speaker WAV path for Coqui XTTS (optional)")
    parser.add_argument("--language", default="en", help="Language for narration")
    
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
        output_path=args.output,
        style=args.style,
        voice_id=args.voice,
        language=args.language
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
