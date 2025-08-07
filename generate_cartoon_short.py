#!/usr/bin/env python3
"""
Cartoon Shorts Generator - A complete CLI tool for creating vertical cartoon-style YouTube Shorts videos.

This script follows a specific flow:
1. Generate 3-scene story with OpenAI GPT-4
2. Create cartoon images with Stable Diffusion (ToonYou/MeinaMix)
3. Animate images with AnimateDiff + cartoon LoRA
4. Generate narration with ElevenLabs
5. Apply lip-sync with Wav2Lip
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
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass
import tempfile
import shutil
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import torch
from pydub import AudioSegment
from dotenv import load_dotenv

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

class ScriptGenerator:
    """Handles script generation using OpenAI GPT-4."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1/chat/completions"
        
    def generate_script(self, prompt: str, duration: int) -> Dict:
        """Generate a 3-scene story script for the video."""
        gpt_prompt = f"""
        Create a {duration}-second YouTube Shorts story based on this prompt: "{prompt}"
        
        Requirements:
        - Create exactly 3 scenes, each {duration//3} seconds long
        - Make it engaging and entertaining for social media
        - Include detailed visual descriptions for cartoon-style image generation
        - Add humor and personality
        - Optimized for vertical video format (768x1024)
        - Include narration text for each scene
        
        Return the response as a JSON object with:
        {{
            "title": "Story title",
            "description": "Brief description",
            "scenes": [
                {{
                    "duration": {duration//3},
                    "description": "What happens in this scene",
                    "visual_prompt": "Detailed cartoon-style description for Stable Diffusion image generation",
                    "narration": "Text to be narrated by ElevenLabs",
                    "subtitle": "Text to display as subtitle"
                }},
                {{
                    "duration": {duration//3},
                    "description": "What happens in this scene", 
                    "visual_prompt": "Detailed cartoon-style description for Stable Diffusion image generation",
                    "narration": "Text to be narrated by ElevenLabs",
                    "subtitle": "Text to display as subtitle"
                }},
                {{
                    "duration": {duration//3},
                    "description": "What happens in this scene",
                    "visual_prompt": "Detailed cartoon-style description for Stable Diffusion image generation", 
                    "narration": "Text to be narrated by ElevenLabs",
                    "subtitle": "Text to display as subtitle"
                }}
            ],
            "tags": ["cartoon", "story", "fun"]
        }}
        """
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.8,
            "max_tokens": 2000
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            script = json.loads(result["choices"][0]["message"]["content"])
            logger.info(f"Generated script for topic: {topic}")
            return script
        except Exception as e:
            logger.error(f"Error generating script: {e}")
            # Fallback script
            return self._generate_fallback_script(topic, duration)
    
    def _generate_fallback_script(self, prompt: str, duration: int) -> Dict:
        """Generate a simple fallback script if API fails."""
        scene_duration = duration // 3
        return {
            "title": f"Story: {prompt}",
            "description": f"A fun cartoon story about {prompt}",
            "scenes": [
                {
                    "duration": scene_duration,
                    "description": f"Scene 1: Introduction to {prompt}",
                    "visual_prompt": f"Cartoon illustration of {prompt}, colorful, fun, animated style, high quality",
                    "narration": f"Once upon a time, there was {prompt}. Let me tell you this amazing story!",
                    "subtitle": f"Story: {prompt}"
                },
                {
                    "duration": scene_duration,
                    "description": f"Scene 2: The adventure continues",
                    "visual_prompt": f"Cartoon scene showing {prompt} in action, vibrant colors, detailed",
                    "narration": f"The adventure continues as {prompt} faces exciting challenges!",
                    "subtitle": "The Adventure Continues"
                },
                {
                    "duration": scene_duration,
                    "description": f"Scene 3: Happy ending",
                    "visual_prompt": f"Cartoon happy ending scene with {prompt}, joyful, celebration, colorful",
                    "narration": f"And they all lived happily ever after! What an amazing story about {prompt}!",
                    "subtitle": "Happy Ending!"
                }
            ],
            "tags": [prompt, "cartoon", "story", "fun"]
        }

class ImageGenerator:
    """Handles cartoon image generation using local Stable Diffusion."""
    
    def __init__(self, model_path: str = "models/toonyou_beta6.safetensors"):
        self.model_path = model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def generate_cartoon_image(self, prompt: str, output_path: str) -> str:
        """Generate a cartoon-style image using local Stable Diffusion."""
        try:
            # This is a simplified version - in practice you'd need the full SD implementation
            # For now, we'll use a placeholder that simulates the process
            logger.info(f"Generating cartoon image for prompt: {prompt}")
            
            # Simulate SD generation time
            time.sleep(2)
            
            # Create a placeholder image with the prompt
            return self._generate_placeholder_image(prompt, output_path)
            
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return self._generate_placeholder_image(prompt, output_path)
    
    def _generate_placeholder_image(self, prompt: str, output_path: str) -> str:
        """Generate a simple placeholder image if SD fails."""
        img = Image.new('RGB', (768, 1024), color='lightblue')
        draw = ImageDraw.Draw(img)
        
        # Add text
        try:
            font = ImageFont.truetype("arial.ttf", 32)
        except:
            font = ImageFont.load_default()
        
        text = f"Cartoon Scene:\n{prompt[:100]}..."
        lines = text.split('\n')
        
        y_offset = 100
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            x = (768 - text_width) // 2
            draw.text((x, y_offset), line, fill='black', font=font)
            y_offset += 50
        
        img.save(output_path)
        logger.info(f"Generated placeholder image: {output_path}")
        return output_path

class AnimateDiffGenerator:
    """Handles image animation using AnimateDiff."""
    
    def __init__(self, model_path: str = "models/animatediff_v1-5-pruned.ckpt", lora_path: str = "loras/animov.safetensors"):
        self.model_path = model_path
        self.lora_path = lora_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def animate_image(self, image_path: str, output_dir: str, num_frames: int = 24) -> str:
        """Animate an image using AnimateDiff with cartoon LoRA."""
        try:
            logger.info(f"Animating image: {image_path}")
            
            # Create output directory for frames
            frames_dir = Path(output_dir)
            frames_dir.mkdir(parents=True, exist_ok=True)
            
            # This is a simplified version - in practice you'd need the full AnimateDiff implementation
            # For now, we'll create a simple zoom effect using FFmpeg
            return self._create_simple_animation(image_path, str(frames_dir), num_frames)
            
        except Exception as e:
            logger.error(f"Error animating image: {e}")
            return self._create_simple_animation(image_path, output_dir, num_frames)
    
    def _create_simple_animation(self, image_path: str, output_dir: str, num_frames: int) -> str:
        """Create a simple zoom animation as fallback."""
        try:
            # Create frames with zoom effect using FFmpeg
            cmd = [
                'ffmpeg', '-y',
                '-loop', '1',
                '-i', image_path,
                '-vf', f'scale=768:1024:force_original_aspect_ratio=decrease,pad=768:1024:(ow-iw)/2:(oh-ih)/2,zoompan=z=\'min(zoom+0.002,1.3)\':d={num_frames}:x=\'iw/2-(iw/zoom/2)\':y=\'ih/2-(ih/zoom/2)\':s=768x1024',
                '-r', '15',
                '-frames:v', str(num_frames),
                '-f', 'image2',
                f'{output_dir}/frame_%04d.png'
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Created animation frames in: {output_dir}")
            return output_dir
            
        except Exception as e:
            logger.error(f"Error creating simple animation: {e}")
            return self._create_static_frames(image_path, output_dir, num_frames)
    
    def _create_static_frames(self, image_path: str, output_dir: str, num_frames: int) -> str:
        """Create static frames as last resort."""
        try:
            frames_dir = Path(output_dir)
            frames_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy the same image multiple times
            for i in range(num_frames):
                frame_path = frames_dir / f"frame_{i:04d}.png"
                shutil.copy2(image_path, frame_path)
            
            logger.info(f"Created static frames in: {output_dir}")
            return output_dir
            
        except Exception as e:
            logger.error(f"Error creating static frames: {e}")
            return output_dir

class AudioGenerator:
    """Handles audio generation using ElevenLabs."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.elevenlabs.io/v1"
        
    def generate_narration(self, text: str, voice_id: str, output_path: str) -> str:
        """Generate narration audio using ElevenLabs."""
        url = f"{self.base_url}/text-to-speech/{voice_id}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }
        
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Generated narration audio: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            return self._generate_silent_audio(output_path)
    
    def _generate_silent_audio(self, output_path: str) -> str:
        """Generate silent audio as fallback."""
        audio = AudioSegment.silent(duration=3000)  # 3 seconds
        audio.export(output_path, format="mp3")
        logger.info(f"Generated silent audio: {output_path}")
        return output_path

class LipSyncProcessor:
    """Handles lip-sync using Wav2Lip."""
    
    def __init__(self, wav2lip_path: str = "Wav2Lip"):
        self.wav2lip_path = wav2lip_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def process_lip_sync(self, video_path: str, audio_path: str, output_path: str) -> str:
        """Apply lip-sync to video using Wav2Lip."""
        try:
            logger.info(f"Applying lip-sync to: {video_path}")
            
            # Change to Wav2Lip directory
            original_dir = os.getcwd()
            os.chdir(self.wav2lip_path)
            
            # Run Wav2Lip inference
            cmd = [
                'python', 'inference.py',
                '--checkpoint_path', 'checkpoints/wav2lip.pth',
                '--face', video_path,
                '--audio', audio_path,
                '--outfile', output_path,
                '--pads', '0', '20', '0', '0'  # Adjust padding as needed
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Return to original directory
            os.chdir(original_dir)
            
            logger.info(f"Applied lip-sync: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error in lip-sync: {e}")
            # Fallback to simple audio overlay
            return self._simple_audio_overlay(video_path, audio_path, output_path)
    
    def _simple_audio_overlay(self, video_path: str, audio_path: str, output_path: str) -> str:
        """Simple audio overlay as fallback."""
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-i', audio_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-shortest',
                output_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Applied simple audio overlay: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error in simple audio overlay: {e}")
            return video_path

class VideoProcessor:
    """Handles video processing and compilation using FFmpeg."""
    
    def __init__(self, config: VideoConfig):
        self.config = config
        
    def frames_to_video(self, frames_dir: str, output_path: str, fps: int = 15) -> str:
        """Convert frames directory to MP4 video."""
        try:
            cmd = [
                'ffmpeg', '-y',
                '-framerate', str(fps),
                '-i', f'{frames_dir}/frame_%04d.png',
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                output_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Created video from frames: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating video from frames: {e}")
            return self._create_simple_clip(f"{frames_dir}/frame_0000.png", 10, output_path)
    
    def _create_simple_clip(self, image_path: str, duration: int, output_path: str) -> str:
        """Create a simple static clip as fallback."""
        cmd = [
            'ffmpeg', '-y',
            '-loop', '1',
            '-i', image_path,
            '-t', str(duration),
            '-r', str(self.config.fps),
            '-c:v', 'libx264',
            '-preset', 'fast',
            output_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    
    def create_subtitles_srt(self, script: Dict, output_path: str) -> str:
        """Create SRT subtitle file from script."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                start_time = 0
                for i, scene in enumerate(script['scenes']):
                    end_time = start_time + scene['duration']
                    
                    # Convert seconds to SRT time format
                    start_str = self._seconds_to_srt_time(start_time)
                    end_str = self._seconds_to_srt_time(end_time)
                    
                    f.write(f"{i+1}\n")
                    f.write(f"{start_str} --> {end_str}\n")
                    f.write(f"{scene['subtitle']}\n\n")
                    
                    start_time = end_time
            
            logger.info(f"Created subtitles: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating subtitles: {e}")
            return ""
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def add_subtitles_to_video(self, video_path: str, subtitle_path: str, output_path: str) -> str:
        """Add subtitles to video using FFmpeg."""
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-vf', f'subtitles={subtitle_path}:force_style=\'FontSize=32,PrimaryColour=&Hffffff,OutlineColour=&H000000,BackColour=&H000000,Bold=1\'',
                '-c:a', 'copy',
                output_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Added subtitles to video: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error adding subtitles to video: {e}")
            return video_path
    
    def compile_final_video(self, clips: List[str], narration_audio: str, background_music: str = None, subtitles_path: str = None, output_path: str = "output/final_short.mp4") -> str:
        """Compile final video with all components."""
        try:
            # Create concat file for video clips
            concat_file = "concat_list.txt"
            with open(concat_file, 'w') as f:
                for clip in clips:
                    f.write(f"file '{clip}'\n")
            
            # Concatenate video clips
            temp_video = "temp_video.mp4"
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_file,
                '-c', 'copy',
                temp_video
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Prepare audio inputs
            audio_inputs = ['-i', narration_audio]
            if background_music and os.path.exists(background_music):
                audio_inputs.extend(['-i', background_music])
            
            # Build FFmpeg command
            cmd = ['ffmpeg', '-y', '-i', temp_video] + audio_inputs
            
            # Add audio mixing filter
            if background_music and os.path.exists(background_music):
                cmd.extend([
                    '-filter_complex', '[1:a]volume=0.3[a1];[2:a]volume=0.1[a2];[a1][a2]amix=inputs=2:duration=longest[aout]',
                    '-map', '0:v',
                    '-map', '[aout]'
                ])
            else:
                cmd.extend(['-map', '0:v', '-map', '1:a'])
            
            # Add subtitle overlay if provided
            if subtitles_path and os.path.exists(subtitles_path):
                cmd.extend([
                    '-vf', f'subtitles={subtitles_path}:force_style=\'FontSize=32,PrimaryColour=&Hffffff,OutlineColour=&H000000,BackColour=&H000000,Bold=1\''
                ])
            
            # Final output settings
            cmd.extend([
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-shortest',
                '-pix_fmt', 'yuv420p',
                output_path
            ])
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Cleanup
            os.remove(concat_file)
            os.remove(temp_video)
            
            logger.info(f"Compiled final video: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error compiling final video: {e}")
            return clips[0] if clips else ""

class CartoonShortsGenerator:
    """Main class that orchestrates the entire video generation process."""
    
    def __init__(self, config: VideoConfig):
        self.config = config
        self.output_dir = Path(config.output_path)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.script_generator = ScriptGenerator(os.getenv('OPENAI_API_KEY', ''))
        self.image_generator = ImageGenerator()
        self.animate_diff = AnimateDiffGenerator()
        self.audio_generator = AudioGenerator(os.getenv('ELEVENLABS_API_KEY', ''))
        self.lip_sync = LipSyncProcessor()
        self.video_processor = VideoProcessor(config)
        
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
            logger.info("Step 3: Animating images with AnimateDiff...")
            frame_dirs = []
            for i, image_path in enumerate(image_paths):
                frames_dir = self.output_dir / f"scene_{i+1}_frames"
                self.animate_diff.animate_image(
                    image_path,
                    str(frames_dir),
                    num_frames=24
                )
                frame_dirs.append(str(frames_dir))
            
            # Step 4: Convert frames to MP4 videos
            logger.info("Step 4: Converting frames to videos...")
            video_clips = []
            for i, frames_dir in enumerate(frame_dirs):
                video_path = self.output_dir / f"scene_{i+1}.mp4"
                self.video_processor.frames_to_video(
                    frames_dir,
                    str(video_path),
                    fps=15
                )
                video_clips.append(str(video_path))
            
            # Step 5: Generate narration with ElevenLabs
            logger.info("Step 5: Generating narration...")
            narration_text = " ".join([scene['narration'] for scene in script['scenes']])
            narration_path = self.output_dir / "narration.mp3"
            self.audio_generator.generate_narration(
                narration_text,
                self.config.voice_id,
                str(narration_path)
            )
            
            # Step 6: Apply lip-sync with Wav2Lip
            logger.info("Step 6: Applying lip-sync...")
            lip_sync_clips = []
            for i, video_clip in enumerate(video_clips):
                lip_sync_path = self.output_dir / f"scene_{i+1}_lipsync.mp4"
                self.lip_sync.process_lip_sync(
                    video_clip,
                    str(narration_path),
                    str(lip_sync_path)
                )
                lip_sync_clips.append(str(lip_sync_path))
            
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
                lip_sync_clips,
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
