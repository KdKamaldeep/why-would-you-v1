#!/usr/bin/env python3
"""
Cartoon Shorts Generator - A complete CLI tool for creating vertical cartoon-style YouTube Shorts videos.

This script follows a specific flow:
1. Generate 3-scene story with OpenAI GPT-4
2. Create cartoon images with Stable Diffusion (ToonYou/MeinaMix)
3. Animate images with AnimateDiff + cartoon LoRA
4. Generate narration with ElevenLabs
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
from dotenv import load_dotenv

# Import modular classes
from .script_generator import ScriptGenerator
from .image_generator import ImageGenerator
from .voice_generator import VoiceGenerator
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
    voice_id: str = "pNInz6obpgDQGcFmaJgB"  # ElevenLabs voice ID
    language: str = "en"
    num_scenes: int = 3

class CartoonShortsGenerator:
    """Main class that orchestrates the entire video generation process."""
    
    def __init__(self, config: VideoConfig):
        self.config = config
        self.output_dir = Path(config.output_path)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components using modular classes
        self.script_generator = ScriptGenerator(os.getenv('OPENAI_API_KEY', ''))
        self.image_generator = ImageGenerator()
        self.animation_generator = AnimationGenerator()
        self.voice_generator = VoiceGenerator(os.getenv('ELEVENLABS_API_KEY', ''))

        
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
            # Step 1: Generate 3-scene story with OpenAI GPT-4
            logger.info("Step 1: Generating 3-scene story...")
            script = self.script_generator.generate_script(self.config.prompt, self.config.duration)
            
            # Step 2: Generate cartoon images with Stable Diffusion
            logger.info("Step 2: Generating cartoon images...")
            image_paths = []
            for i, scene in enumerate(script['scenes']):
                image_path = self.output_dir / f"scene_{i+1}.png"
                self.image_generator.generate_cartoon_image(
                    scene['visual_prompt'], 
                    str(image_path)
                )
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
            
            frame_dirs = self.animation_generator.animate_multiple_images_with_duration(
                image_paths,
                str(self.output_dir),
                frames_per_scene,
                prompts=scene_prompts
            )
            
            # Step 4: Convert frames to MP4 videos
            logger.info("Step 4: Converting frames to videos...")
            video_clips = self.video_processor.frames_to_multiple_videos(
                frame_dirs,
                str(self.output_dir),
                fps=15
            )
            
            # Step 5: Generate narration with ElevenLabs
            logger.info("Step 5: Generating narration...")
            narration_path = self.output_dir / "narration.mp3"
            self.voice_generator.generate_narration_from_script(
                script,
                self.config.voice_id,
                str(narration_path)
            )
            
            # Step 6: Use video clips directly (no lip-sync)
            logger.info("Step 6: Preparing video clips...")
            final_clips = video_clips
            
            # Step 7: Create subtitles
            logger.info("Step 7: Creating subtitles...")
            subtitles_path = self.output_dir / "subtitles.srt"
            self.video_processor.create_subtitles_srt(script, str(subtitles_path))
            
            # Step 8: Select background music
            logger.info("Step 8: Adding background music...")
            background_music = self._get_background_music()
            
            # Step 9: Compile final video
            logger.info("Step 9: Compiling final video...")
            final_output = self.output_dir / "final_short.mp4"
            self.video_processor.compile_final_video(
                final_clips,
                str(narration_path),
                background_music,
                str(subtitles_path),
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

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate cartoon-style YouTube Shorts videos")
    parser.add_argument("--prompt", required=True, help="Story prompt (e.g., 'A baby lion opens a smoothie shop in the jungle')")
    parser.add_argument("--duration", type=int, default=30, help="Video duration in seconds")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--style", default="cartoon", help="Visual style")
    parser.add_argument("--voice", default="pNInz6obpgDQGcFmaJgB", help="ElevenLabs voice ID")
    parser.add_argument("--language", default="en", help="Language for narration")
    
    args = parser.parse_args()
    
    # Validate environment variables
    required_env_vars = ['OPENAI_API_KEY', 'ELEVENLABS_API_KEY']
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
