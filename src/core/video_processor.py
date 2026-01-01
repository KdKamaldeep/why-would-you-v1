#!/usr/bin/env python3
"""
Video Processor Module - Handles video processing and compilation using FFmpeg
"""

import os
import logging
import subprocess
from typing import List, Dict, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class VideoConfig:
    """Configuration for video processing."""
    fps: int = 10  # Reduced from 15 to 10 for slower playback
    width: int = 768
    height: int = 1024
    # Encoding options (optimize for smaller files)
    codec: str = "libx265"           # Use HEVC for ~40-60% smaller files
    crf: int = 23                     # Lower = higher quality. 23 is good for social media
    preset: str = "medium"           # slower = smaller; keep reasonable CPU cost
    tune: str = "grain"              # Valid for libx265 (psnr, ssim, grain, zerolatency, fastdecode, animation). For libx264, use "film"
    audio_bitrate: str = "96k"       # narration-friendly bitrate
    faststart: bool = True            # enable moov atom at front for streaming

class VideoProcessor:
    """Handles video processing and compilation using FFmpeg."""
    
    def __init__(self, config: VideoConfig):
        self.config = config
        
    def frames_to_video(self, frames_dir: str, output_path: str, fps: int = 10) -> str:  # Reduced default from 15 to 10
        """Convert frames directory to MP4 video."""
        try:
            cmd = [
                'ffmpeg', '-y',
                '-framerate', str(fps),
                '-i', f'{frames_dir}/frame_%04d.png',
                '-c:v', self.config.codec,
                '-preset', self.config.preset,
                '-crf', str(self.config.crf),
                '-pix_fmt', 'yuv420p'
            ]
            # Add tune parameter only for supported codecs
            if self.config.codec == 'libx264':
                cmd.extend(['-tune', 'film'])  # film is valid for libx264
            elif self.config.codec == 'libx265':
                cmd.extend(['-tune', self.config.tune])  # grain, psnr, ssim, etc. for libx265
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

    def get_video_duration(self, video_file: str) -> float:
        """Get the duration of a video file in seconds using FFmpeg."""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'csv=p=0', video_file
            ]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            duration = float(result.stdout.strip())
            logger.info(f"Video duration for {video_file}: {duration:.2f}s")
            return duration
        except Exception as e:
            logger.error(f"Error getting video duration for {video_file}: {e}")
            # Return a default duration if we can't determine it
            return 8.0

    def extend_video_duration(self, input_video: str, target_duration_sec: float, output_video: str) -> str:
        """Extend video duration by looping or slowing down to match target duration."""
        try:
            # Get current video duration
            current_duration = self.get_video_duration(input_video)
            
            if current_duration >= target_duration_sec:
                # Video is already long enough, just copy
                import shutil
                shutil.copy2(input_video, output_video)
                return output_video
            
            # Calculate how many times we need to loop
            loop_count = int(target_duration_sec / current_duration) + 1
            
            if loop_count <= 2:
                # Just slow down the video to match duration
                speed_factor = current_duration / target_duration_sec
                cmd = [
                    'ffmpeg', '-y',
                    '-i', input_video,
                    '-filter:v', f'setpts={speed_factor}*PTS',
                    '-filter:a', f'atempo={1/speed_factor}' if speed_factor < 0.5 else 'atempo=0.5,atempo=0.5' if speed_factor < 0.25 else 'atempo=0.5,atempo=0.5,atempo=0.5',
                    '-c:v', 'libx264',
                    '-c:a', 'aac',
                    '-shortest',
                    output_video
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                logger.info(f"Extended video by slowing down: {current_duration:.1f}s → {target_duration_sec:.1f}s")
            else:
                # Loop the video multiple times
                # Create a concat file
                concat_file = output_video.replace('.mp4', '_concat.txt')
                with open(concat_file, 'w') as f:
                    for _ in range(loop_count):
                        f.write(f"file '{input_video}'\n")
                
                # Concatenate videos
                cmd = [
                    'ffmpeg', '-y',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', concat_file,
                    '-c', 'copy',
                    output_video
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                
                # Trim to exact duration
                temp_output = output_video.replace('.mp4', '_temp.mp4')
                import shutil
                shutil.move(output_video, temp_output)
                
                cmd = [
                    'ffmpeg', '-y',
                    '-i', temp_output,
                    '-t', str(target_duration_sec),
                    '-c', 'copy',
                    output_video
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                
                # Clean up
                os.remove(concat_file)
                os.remove(temp_output)
                
                logger.info(f"Extended video by looping {loop_count}x: {current_duration:.1f}s → {target_duration_sec:.1f}s")
            
            return output_video
            
        except Exception as e:
            logger.error(f"Error extending video duration: {e}")
            # Fallback: just copy the original
            import shutil
            shutil.copy2(input_video, output_video)
            return output_video

    def adjust_video_duration(self, input_video: str, target_duration_sec: float, output_video: str) -> str:
        """Adjust video duration by speeding up or slowing down to match target duration."""
        try:
            # Get current video duration
            current_duration = self.get_video_duration(input_video)
            
            if abs(current_duration - target_duration_sec) < 0.1:
                # Duration is close enough, just copy
                import shutil
                shutil.copy2(input_video, output_video)
                return output_video
            
            # Calculate speed factor
            speed_factor = current_duration / target_duration_sec
            
            cmd = [
                'ffmpeg', '-y',
                '-i', input_video,
                '-filter:v', f'setpts={1/speed_factor:.6f}*PTS',
                '-filter:a', f'atempo={speed_factor:.6f}',
                '-c:v', self.config.codec,
                '-preset', self.config.preset,
                '-crf', str(self.config.crf),
                '-c:a', 'aac',
                '-b:a', self.config.audio_bitrate
            ]
            # Add tune parameter only for supported codecs
            if self.config.codec == 'libx264':
                cmd.extend(['-tune', 'film'])  # film is valid for libx264
            elif self.config.codec == 'libx265':
                cmd.extend(['-tune', self.config.tune])  # grain, psnr, ssim, etc. for libx265
            cmd.append(output_video)
            
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Adjusted video duration: {current_duration:.2f}s → {target_duration_sec:.2f}s (speed: {speed_factor:.2f}x)")
            return output_video
            
        except Exception as e:
            logger.error(f"Error adjusting video duration: {e}")
            return input_video

    def estimate_narration_duration(self, text: str, words_per_minute: int = 150) -> float:
        """Estimate narration duration based on word count."""
        words = len(text.split())
        duration_minutes = words / words_per_minute
        return duration_minutes * 60  # Convert to seconds

    def get_audio_duration(self, audio_file: str) -> float:
        """Get the duration of an audio file in seconds using FFmpeg."""
        import os
        if not os.path.exists(audio_file):
            logger.warning(f"Audio file does not exist: {audio_file}, returning default duration")
            return 8.0
        
        logger.info(f"Getting audio duration for {audio_file}")
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'csv=p=0', audio_file
            ]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            duration = float(result.stdout.strip())
            logger.info(f"Audio duration for {audio_file}: {duration:.2f}s")
            return duration
        except Exception as e:
            logger.error(f"Error getting audio duration for {audio_file}: {e}")
            # Return a default duration if we can't determine it
            return 8.0

    def adjust_audio_to_duration(self, input_audio: str, target_duration_sec: float, output_audio: str) -> str:
        """Pad with silence or trim audio to exactly target duration."""
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', input_audio,
                '-af', 'apad',
                '-t', f"{target_duration_sec:.3f}",
                '-c:a', 'aac',
                '-b:a', self.config.audio_bitrate,
                output_audio
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            return output_audio
        except Exception as e:
            logger.error(f"Error fitting audio to duration: {e}")
            return input_audio

    def concat_audios(self, audio_files: List[str], output_audio: str) -> str:
        """Concatenate multiple audio files into one AAC file."""
        try:
            concat_list = 'audio_concat_list.txt'
            with open(concat_list, 'w') as f:
                for p in audio_files:
                    f.write(f"file '{p}'\n")
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat', '-safe', '0',
                '-i', concat_list,
                '-c:a', 'aac',
                '-b:a', self.config.audio_bitrate,
                output_audio
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            try:
                os.remove(concat_list)
            except Exception:
                pass
            return output_audio
        except Exception as e:
            logger.error(f"Error concatenating audios: {e}")
            return audio_files[0] if audio_files else ''
    
    def frames_to_multiple_videos(self, frame_dirs: List[str], output_dir: str, fps: int = 10) -> List[str]:  # Reduced default from 15 to 10
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
    
    def create_pause_video(self, duration: float, output_path: str, color: str = "black") -> str:
        """Create a pause video of specified duration."""
        try:
            cmd = [
                'ffmpeg', '-y',
                '-f', 'lavfi',
                '-i', f'color=c={color}:size={self.config.width}x{self.config.height}:duration={duration}',
                '-c:v', self.config.codec,
                '-preset', self.config.preset,
                '-crf', str(self.config.crf),
                '-pix_fmt', 'yuv420p'
            ]
            # Add tune parameter only for supported codecs
            if self.config.codec == 'libx264':
                cmd.extend(['-tune', 'film'])  # film is valid for libx264
            elif self.config.codec == 'libx265':
                cmd.extend(['-tune', self.config.tune])  # grain, psnr, ssim, etc. for libx265
            cmd.append(output_path)
            
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Created pause video: {output_path} ({duration:.2f}s)")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating pause video: {e}")
            return ""

    def create_silent_audio(self, duration: float, output_path: str) -> str:
        """Create silent audio of specified duration."""
        try:
            cmd = [
                'ffmpeg', '-y',
                '-f', 'lavfi',
                '-i', f'anullsrc=channel_layout=stereo:sample_rate=44100:duration={duration}',
                '-c:a', 'aac',
                '-b:a', self.config.audio_bitrate,
                output_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Created silent audio: {output_path} ({duration:.2f}s)")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating silent audio: {e}")
            return ""

    def apply_scene_effects(self, input_video: str, output_video: str, 
                           speed_factor: float = 1.0, 
                           zoom_direction: str = "none",
                           transition: str = "none") -> str:
        """
        Apply speed adjustment, pan/zoom, and swipe transitions to a scene video.
        
        Args:
            input_video: Path to input video
            output_video: Path to output video
            speed_factor: Speed multiplier (1.2 or 1.5 to speed up)
            zoom_direction: "in" or "out" for zoom effects
            transition: "swipe_up", "swipe_down", "swipe_left", "swipe_right", or "none"
        
        Returns:
            Path to processed video
        """
        try:
            # Get video dimensions
            probe_cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height',
                '-of', 'csv=s=x:p=0',
                input_video
            ]
            result = subprocess.run(probe_cmd, check=True, capture_output=True, text=True)
            width, height = map(int, result.stdout.strip().split('x'))
            
            # Get video duration
            duration = self.get_video_duration(input_video)
            
            # Build filter complex for all effects
            filters = []
            
            # 1. Speed adjustment (speed up video)
            if speed_factor > 1.0:
                # Speed up video: setpts reduces PTS (makes it faster)
                # For speed_factor 1.5, we want 1/1.5 = 0.6667
                pts_factor = 1.0 / speed_factor
                filters.append(f"setpts={pts_factor:.6f}*PTS")
            
            # 2. Pan/Zoom effect using zoompan filter
            # Calculate frames for zoompan (use fps estimate)
            # Note: zoompan's 'd' parameter is output frames, so we use original duration
            # since setpts only changes timestamps, not frame count
            fps = 30  # Estimate fps
            # After speed adjustment, output duration = duration / speed_factor
            # But zoompan processes all input frames, so use original frame count
            total_frames = int(duration * fps)
            
            # Build zoompan filter with zoom and pan combined
            if zoom_direction in ["in", "out"]:
                if zoom_direction == "in":
                    # Progressive zoom in: z increases from 1.0 to 1.3
                    zoom_expr = "1.0 + (on/d) * 0.3"
                else:
                    # Progressive zoom out: z decreases from 1.3 to 1.0
                    zoom_expr = "1.3 - (on/d) * 0.3"
                
                # Add panning based on transition direction
                if transition == "swipe_left":
                    # Pan left: x moves from right to left
                    x_expr = "iw - (iw/zoom) - (on/d) * (iw - iw/zoom)"
                    y_expr = "ih/2-(ih/zoom/2)"
                elif transition == "swipe_right":
                    # Pan right: x moves from left to right
                    x_expr = "(on/d) * (iw - iw/zoom)"
                    y_expr = "ih/2-(ih/zoom/2)"
                elif transition == "swipe_up":
                    # Pan up: y moves from bottom to top
                    x_expr = "iw/2-(iw/zoom/2)"
                    y_expr = "ih - (ih/zoom) - (on/d) * (ih - ih/zoom)"
                elif transition == "swipe_down":
                    # Pan down: y moves from top to bottom
                    x_expr = "iw/2-(iw/zoom/2)"
                    y_expr = "(on/d) * (ih - ih/zoom)"
                else:
                    # No panning, just center zoom
                    x_expr = "iw/2-(iw/zoom/2)"
                    y_expr = "ih/2-(ih/zoom/2)"
                
                zoom_filter = f"zoompan=z='{zoom_expr}':d={total_frames}:x='{x_expr}':y='{y_expr}':s={width}x{height}"
                filters.append(zoom_filter)
            elif transition in ["swipe_left", "swipe_right", "swipe_up", "swipe_down"]:
                # Pan only (no zoom) - use crop with moving window for smooth panning
                pan_range = 0.2  # 20% of frame size for panning range
                crop_w = int(width * (1 - pan_range))
                crop_h = int(height * (1 - pan_range))
                
                if transition == "swipe_left":
                    # Pan left: crop window moves from right to left
                    pan_filter = f"crop={crop_w}:{height}:if(gte(t,0), {width - crop_w} - (t/{duration}) * ({width - crop_w}), {width - crop_w}):0"
                elif transition == "swipe_right":
                    # Pan right: crop window moves from left to right
                    pan_filter = f"crop={crop_w}:{height}:if(gte(t,0), (t/{duration}) * ({width - crop_w}), 0):0"
                elif transition == "swipe_up":
                    # Pan up: crop window moves from bottom to top
                    pan_filter = f"crop={width}:{crop_h}:0:if(gte(t,0), {height - crop_h} - (t/{duration}) * ({height - crop_h}), {height - crop_h})"
                elif transition == "swipe_down":
                    # Pan down: crop window moves from top to bottom
                    pan_filter = f"crop={width}:{crop_h}:0:if(gte(t,0), (t/{duration}) * ({height - crop_h}), 0)"
                
                filters.append(pan_filter)
            
            # Build FFmpeg command
            cmd = [
                'ffmpeg', '-y',
                '-i', input_video
            ]
            
            # Add video filter if we have any effects
            if filters:
                filter_chain = ','.join(filters)
                cmd.extend(['-vf', filter_chain])
            
            # Handle audio speed adjustment
            if speed_factor > 1.0:
                # Speed up audio to match video
                # atempo can only go up to 2.0, so chain if needed
                if speed_factor <= 2.0:
                    cmd.extend(['-af', f'atempo={speed_factor:.6f}'])
                else:
                    # Chain atempo filters for speeds > 2.0
                    atempo_chain = []
                    remaining_speed = speed_factor
                    while remaining_speed > 2.0:
                        atempo_chain.append('atempo=2.0')
                        remaining_speed /= 2.0
                    if remaining_speed > 1.0:
                        atempo_chain.append(f'atempo={remaining_speed:.6f}')
                    cmd.extend(['-af', ','.join(atempo_chain)])
            
            # Encoding settings
            cmd.extend([
                '-c:v', self.config.codec,
                '-preset', self.config.preset,
                '-crf', str(self.config.crf),
                '-c:a', 'aac',
                '-b:a', self.config.audio_bitrate,
                '-pix_fmt', 'yuv420p'
            ])
            
            # Add tune parameter
            if self.config.codec == 'libx264':
                cmd.extend(['-tune', 'film'])
            elif self.config.codec == 'libx265':
                cmd.extend(['-tune', self.config.tune])
            
            if self.config.codec == 'libx265':
                cmd.extend(['-tag:v', 'hvc1'])
            
            if self.config.faststart:
                cmd.extend(['-movflags', '+faststart'])
            
            cmd.append(output_video)
            
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Applied effects to scene: speed={speed_factor}x, zoom={zoom_direction}, transition={transition}")
            return output_video
            
        except Exception as e:
            logger.error(f"Error applying scene effects: {e}")
            # Fallback: just copy the original
            import shutil
            shutil.copy2(input_video, output_video)
            return output_video

    def compile_final_video(self, clips: List[str], narration_audio: Union[str, List[str]], background_music: str = None, subtitles_path: str = None, output_path: str = "output/final_short.mp4") -> str:
        """Compile final video with all components."""
        try:
            # Create concat file for video clips
            # Use absolute paths to avoid issues
            concat_file = "concat_list.txt"
            with open(concat_file, 'w') as f:
                for clip in clips:
                    # Convert to absolute path
                    abs_clip = os.path.abspath(clip)
                    # Escape single quotes in path for ffmpeg
                    escaped_path = abs_clip.replace("'", "'\\''")
                    f.write(f"file '{escaped_path}'\n")
            
            logger.info(f"📋 Created concat file with {len(clips)} clips")
            for i, clip in enumerate(clips, 1):
                logger.info(f"   {i}. {os.path.abspath(clip)}")
            
            # Concatenate video clips
            # Use re-encoding instead of copy to handle different codecs from Veo
            temp_video = "temp_video.mp4"
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_file,
                '-c:v', self.config.codec,
                '-preset', self.config.preset,
                '-crf', str(self.config.crf),
                '-c:a', 'aac',
                '-b:a', self.config.audio_bitrate,
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',  # Ensure moov atom is at front
                temp_video
            ]
            
            # Add tune parameter
            if self.config.codec == 'libx264':
                cmd.extend(['-tune', 'film'])
            elif self.config.codec == 'libx265':
                cmd.extend(['-tune', self.config.tune])
            
            if self.config.codec == 'libx265':
                cmd.extend(['-tag:v', 'hvc1'])
            
            logger.info(f"🔗 Concatenating {len(clips)} video clips...")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if not os.path.exists(temp_video) or os.path.getsize(temp_video) == 0:
                error_msg = result.stderr if result.stderr else "Unknown error"
                logger.error(f"❌ FFmpeg concat error: {error_msg}")
                raise RuntimeError(f"Failed to concatenate videos. FFmpeg error: {error_msg}")
            
            # Verify temp video was created successfully
            if not os.path.exists(temp_video) or os.path.getsize(temp_video) == 0:
                raise RuntimeError(f"Failed to create temp video: {temp_video} is missing or empty")
            
            # Probe temp video duration
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

            # If narration_audio is a list, first concatenate into one track
            if isinstance(narration_audio, list):
                merged_narration = 'merged_narration.aac'
                narration_audio = self.concat_audios(narration_audio, merged_narration)
            
            # Probe narration audio duration
            narration_duration = None
            if narration_audio and os.path.exists(narration_audio):
                try:
                    probe_cmd = [
                        'ffprobe', '-v', 'error',
                        '-show_entries', 'format=duration',
                        '-of', 'default=nw=1:nk=1',
                        narration_audio
                    ]
                    result = subprocess.run(probe_cmd, check=True, capture_output=True)
                    narration_duration_str = result.stdout.decode('utf-8', errors='ignore').strip()
                    narration_duration = float(narration_duration_str)
                except Exception:
                    narration_duration = None
            
            # If narration is longer than video, loop the video to match narration length
            if video_duration and narration_duration and narration_duration > video_duration:
                logger.info(f"📹 Video ({video_duration:.2f}s) is shorter than narration ({narration_duration:.2f}s)")
                logger.info(f"🔄 Looping video to match narration length...")
                
                # Calculate how many loops needed
                loops_needed = int(narration_duration / video_duration) + 1
                logger.info(f"   Looping video {loops_needed} times to cover {narration_duration:.2f}s")
                
                # Create a concat file with the video repeated
                loop_concat_file = "loop_concat_list.txt"
                with open(loop_concat_file, 'w') as f:
                    for _ in range(loops_needed):
                        f.write(f"file '{temp_video}'\n")
                
                # Create looped video
                looped_video = "temp_video_looped.mp4"
                loop_cmd = [
                    'ffmpeg', '-y',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', loop_concat_file,
                    '-c', 'copy',
                    '-movflags', '+faststart',
                    looped_video
                ]
                subprocess.run(loop_cmd, check=True, capture_output=True)
                
                # Trim to exact narration duration
                final_looped_video = "temp_video_final.mp4"
                trim_cmd = [
                    'ffmpeg', '-y',
                    '-i', looped_video,
                    '-t', f"{narration_duration:.3f}",
                    '-c', 'copy',
                    final_looped_video
                ]
                subprocess.run(trim_cmd, check=True, capture_output=True)
                
                # Replace temp_video with looped version
                try:
                    os.remove(temp_video)
                    os.rename(final_looped_video, temp_video)
                    os.remove(looped_video)
                    os.remove(loop_concat_file)
                except Exception as e:
                    logger.warning(f"Could not clean up loop files: {e}")
                
                # Update video duration to match narration
                video_duration = narration_duration
                logger.info(f"✅ Video extended to {video_duration:.2f}s to match narration")

            # Prepare audio inputs
            # Veo videos already have SFX, so we need to mix video audio + narration
            # No background music - Veo videos include SFX
            audio_inputs = ['-i', narration_audio]
            
            # Check if video has audio stream
            probe_audio_cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'a:0',
                '-show_entries', 'stream=codec_type',
                '-of', 'default=nw=1:nk=1',
                temp_video
            ]
            has_video_audio = False
            try:
                result = subprocess.run(probe_audio_cmd, capture_output=True, text=True)
                has_video_audio = result.returncode == 0 and 'audio' in result.stdout.lower()
            except Exception:
                pass
            
            # Build FFmpeg command
            cmd = ['ffmpeg', '-y', '-i', temp_video] + audio_inputs
            
            # Mix video audio (SFX from Veo) with narration if video has audio
            if has_video_audio:
                # Video audio is input 0:a, narration is input 1:a
                # Mix them: SFX at lower volume (0.3), narration at higher volume (1.0)
                cmd.extend([
                    '-filter_complex', '[0:a]volume=0.3[sfx];[1:a]volume=1.0[voice];[sfx][voice]amix=inputs=2:duration=longest:dropout_transition=2,apad[aout]',
                    '-map', '0:v',
                    '-map', '[aout]'
                ])
            else:
                # Video has no audio, just use narration
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
                '-c:a', 'aac',
                '-b:a', self.config.audio_bitrate,
                '-pix_fmt', 'yuv420p'
            ])
            # Add tune parameter only for supported codecs
            if self.config.codec == 'libx264':
                cmd.extend(['-tune', 'film'])  # film is valid for libx264
            elif self.config.codec == 'libx265':
                cmd.extend(['-tune', self.config.tune])  # grain, psnr, ssim, etc. for libx265
            # Use narration duration if available, otherwise video duration
            # This ensures video matches narration length (video was already extended if needed)
            target_duration = narration_duration if narration_duration else video_duration
            if target_duration is not None and target_duration > 0:
                cmd.extend(['-t', f"{target_duration:.3f}"])
            # Do NOT use -shortest; we want full narration length
            if self.config.codec == 'libx265':
                cmd.extend(['-tag:v', 'hvc1'])
            if self.config.faststart:
                cmd.extend(['-movflags', '+faststart'])
            cmd.append(output_path)
            
            # Run FFmpeg command and check for errors
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Verify output file was created successfully
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                error_msg = result.stderr if result.stderr else "Unknown FFmpeg error"
                raise RuntimeError(f"FFmpeg failed to create output file: {output_path}. Error: {error_msg}")
            
            # Cleanup
            try:
                os.remove(concat_file)
            except Exception:
                pass
            try:
                os.remove(temp_video)
            except Exception:
                pass
            
            logger.info(f"Compiled final video: {output_path}")
            return output_path
            
        except subprocess.CalledProcessError as e:
            error_output = e.stderr if isinstance(e.stderr, str) else (e.stderr.decode('utf-8', errors='ignore') if e.stderr else str(e))
            logger.error(f"❌ Error compiling final video: FFmpeg command failed with exit code {e.returncode}")
            logger.error(f"FFmpeg stderr: {error_output}")
            if hasattr(e, 'stdout') and e.stdout:
                stdout_str = e.stdout if isinstance(e.stdout, str) else e.stdout.decode('utf-8', errors='ignore')
                logger.error(f"FFmpeg stdout: {stdout_str}")
            # Clean up partial files
            try:
                if os.path.exists("temp_video.mp4"):
                    os.remove("temp_video.mp4")
                if os.path.exists(output_path):
                    os.remove(output_path)
                if os.path.exists("concat_list.txt"):
                    os.remove("concat_list.txt")
            except Exception:
                pass
            raise RuntimeError(f"Failed to compile final video. FFmpeg error: {error_output}")
        except Exception as e:
            logger.error(f"Error compiling final video: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Clean up partial files
            try:
                if os.path.exists("temp_video.mp4"):
                    os.remove("temp_video.mp4")
                if os.path.exists(output_path):
                    os.remove(output_path)
            except Exception:
                pass
            raise

