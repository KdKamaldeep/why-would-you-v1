#!/usr/bin/env python3
"""
Video Processor Module - Handles video processing and compilation using FFmpeg
"""

import os
import logging
import subprocess
from typing import List, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class VideoConfig:
    """Configuration for video processing."""
    fps: int = 15
    width: int = 768
    height: int = 1024
    # Encoding options (optimize for smaller files)
    codec: str = "libx265"           # Use HEVC for ~40-60% smaller files
    crf: int = 28                     # Lower = higher quality. 28 is good for social/cartoon
    preset: str = "medium"           # slower = smaller; keep reasonable CPU cost
    tune: str = "animation"          # better compression for cartoons
    audio_bitrate: str = "96k"       # narration-friendly bitrate
    faststart: bool = True            # enable moov atom at front for streaming

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
                '-c:v', self.config.codec,
                '-preset', self.config.preset,
                '-crf', str(self.config.crf),
                '-tune', self.config.tune,
                '-pix_fmt', 'yuv420p'
            ]
            # Improve compatibility for HEVC in MP4 (especially on Safari)
            if self.config.codec == 'libx265':
                cmd.extend(['-tag:v', 'hvc1'])
            if self.config.faststart:
                cmd.extend(['-movflags', '+faststart'])
            cmd.append(output_path)
            
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Created video from frames: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating video from frames: {e}")
            return self._create_simple_clip(f"{frames_dir}/frame_0000.png", 10, output_path)
    
    def frames_to_multiple_videos(self, frame_dirs: List[str], output_dir: str, fps: int = 15) -> List[str]:
        """Convert multiple frame directories to MP4 videos."""
        video_paths = []
        for i, frames_dir in enumerate(frame_dirs):
            output_path = f"{output_dir}/scene_{i+1}.mp4"
            video_path = self.frames_to_video(frames_dir, output_path, fps)
            video_paths.append(video_path)
        return video_paths
    
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
            
            # Probe temp video duration to cap final output length safely
            try:
                probe_cmd = [
                    'ffprobe', '-v', 'error',
                    '-show_entries', 'format=duration',
                    '-of', 'default=nw=1:nk=1',
                    temp_video
                ]
                result = subprocess.run(probe_cmd, check=True, capture_output=True)
                video_duration_str = result.stdout.decode('utf-8', errors='ignore').strip()
                video_duration = float(video_duration_str)
            except Exception:
                video_duration = None

            # Prepare audio inputs
            audio_inputs = ['-i', narration_audio]
            have_music = bool(background_music and os.path.exists(background_music))
            if have_music:
                audio_inputs.extend(['-i', background_music])
            
            # Build FFmpeg command
            cmd = ['ffmpeg', '-y', '-i', temp_video] + audio_inputs
            
            # Add audio mixing filter
            if have_music:
                # Mix narration and bgm to the longest, then pad to ensure audio covers full video duration
                cmd.extend([
                    '-filter_complex', '[1:a]volume=0.85[a1];[2:a]volume=0.15[a2];[a1][a2]amix=inputs=2:duration=longest,apad[aout]',
                    '-map', '0:v',
                    '-map', '[aout]'
                ])
            else:
                # Single narration track: pad with silence to ensure full coverage
                cmd.extend(['-map', '0:v', '-map', '1:a', '-af', 'apad'])
            
            # Add subtitle overlay if provided
            if subtitles_path and os.path.exists(subtitles_path):
                cmd.extend([
                    '-vf', f'subtitles={subtitles_path}:force_style=\'FontSize=32,PrimaryColour=&Hffffff,OutlineColour=&H000000,BackColour=&H000000,Bold=1\''
                ])
            
            # Final output settings (size-focused)
            cmd.extend([
                '-c:v', self.config.codec,
                '-preset', self.config.preset,
                '-crf', str(self.config.crf),
                '-tune', self.config.tune,
                '-c:a', 'aac',
                '-b:a', self.config.audio_bitrate,
                '-pix_fmt', 'yuv420p'
            ])
            # Cap final muxing to video duration to prevent runaway outputs
            if video_duration is not None and video_duration > 0:
                cmd.extend(['-t', f"{video_duration:.3f}"])
            # Do NOT use -shortest; we want full video length regardless of audio length
            if self.config.codec == 'libx265':
                cmd.extend(['-tag:v', 'hvc1'])
            if self.config.faststart:
                cmd.extend(['-movflags', '+faststart'])
            cmd.append(output_path)
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Cleanup
            os.remove(concat_file)
            os.remove(temp_video)
            
            logger.info(f"Compiled final video: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error compiling final video: {e}")
            return clips[0] if clips else ""

