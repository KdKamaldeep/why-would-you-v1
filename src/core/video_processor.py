#!/usr/bin/env python3
"""
Video Processor Module - Handles video processing and compilation using FFmpeg
"""

import os
import logging
import subprocess
from typing import List, Dict, Union
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class VideoConfig:
    """Configuration for video processing."""
    fps: int = 10  # Reduced from 15 to 10 for slower playback
    width: int = 768
    height: int = 1024
    # Encoding options (optimized for speed - AI shorts)
    # DEV mode: veryfast preset, crf=24 (fastest, slightly lower quality)
    # PROD mode: fast preset, crf=22 (balanced speed/quality)
    # Set VIDEO_EXPORT_MODE=dev for faster exports during development
    _export_mode: str = None  # Internal: will be set based on env var
    
    @property
    def export_mode(self) -> str:
        """Get export mode from environment or default to PROD."""
        if self._export_mode:
            return self._export_mode
        return os.getenv("VIDEO_EXPORT_MODE", "prod").lower()
    
    @property
    def codec(self) -> str:
        """Always use libx264 for speed (HEVC is too slow for shorts)."""
        return "libx264"
    
    @property
    def preset(self) -> str:
        """Get preset based on export mode."""
        if self.export_mode == "dev":
            return "veryfast"  # Fastest encoding
        return "fast"  # Balanced speed/quality
    
    @property
    def crf(self) -> int:
        """Get CRF based on export mode."""
        if self.export_mode == "dev":
            return 24  # Slightly lower quality, faster
        return 22  # High quality, still fast
    
    tune: str = "film"              # Valid for libx264 (film, animation, grain, stillimage, fastdecode, zerolatency)
    audio_bitrate: str = "96k"       # narration-friendly bitrate
    faststart: bool = True            # enable moov atom at front for streaming

class VideoProcessor:
    """Handles video processing and compilation using FFmpeg."""
    
    def __init__(self, config: VideoConfig):
        self.config = config
        # Log encoding configuration
        mode = config.export_mode
        logger.info(f"📹 VideoProcessor initialized: mode={mode.upper()}, codec={config.codec}, preset={config.preset}, crf={config.crf}")
        
    def frames_to_video(self, frames_dir: str, output_path: str, fps: int = 10) -> str:  # Reduced default from 15 to 10
        """Convert frames directory to MP4 video. Tries GPU encoding first, falls back to CPU."""
        try:
            # Try GPU NVENC first (much faster - 5-10x speedup)
            try:
                cmd = [
                    'ffmpeg', '-y',
                    '-framerate', str(fps),
                    '-i', f'{frames_dir}/frame_%04d.png',
                    '-c:v', 'h264_nvenc',
                    '-preset', 'p1',  # p1 = fastest, p7 = slowest (best quality)
                    '-rc', 'vbr',  # Variable bitrate mode
                    '-b:v', '10M',  # Target bitrate
                    '-maxrate', '20M',  # Max bitrate
                    '-pix_fmt', 'yuv420p'
                ]
                if self.config.faststart:
                    cmd.extend(['-movflags', '+faststart'])
                cmd.append(output_path)
                
                logger.info("🚀 Attempting GPU encoding (h264_nvenc)...")
                result = subprocess.run(cmd, check=True, capture_output=True, text=True, stderr=subprocess.PIPE)
                logger.info(f"✅ Video encoded using GPU NVENC (h264_nvenc) - {output_path}")
                return output_path
            except subprocess.CalledProcessError as e:
                error_msg = e.stderr.decode('utf-8', errors='ignore') if isinstance(e.stderr, bytes) else (e.stderr or str(e))
                logger.warning(f"⚠️ GPU NVENC encoding failed: {error_msg[:300]}")
                logger.info("🔄 Falling back to CPU encoding (libx264)...")
            except Exception as e:
                logger.warning(f"⚠️ GPU encoding error: {e}")
                logger.info("🔄 Falling back to CPU encoding (libx264)...")
            
            # CPU fallback (libx264 with optimized preset)
            codec = self.config.codec
            preset = self.config.preset
            crf = self.config.crf
            logger.info(f"💻 Encoding with CPU: codec={codec}, preset={preset}, crf={crf}")
            cmd = [
                'ffmpeg', '-y',
                '-framerate', str(fps),
                '-i', f'{frames_dir}/frame_%04d.png',
                '-c:v', codec,
                '-preset', preset,
                '-crf', str(crf),
                '-tune', self.config.tune,
                '-pix_fmt', 'yuv420p'
            ]
            if self.config.faststart:
                cmd.extend(['-movflags', '+faststart'])
            cmd.append(output_path)
            
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"✅ Video encoded using CPU ({codec}, {preset}, crf={crf})")
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
            # Add tune parameter for libx264
            cmd.extend(['-tune', self.config.tune])  # film is default for libx264
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
            # Add tune parameter for libx264
            cmd.extend(['-tune', self.config.tune])  # film is default for libx264
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

    def compile_final_video(self, clips: List[str], narration_audio: Union[str, List[str]], background_music: str = None, subtitles_path: str = None, output_path: str = "output/final_short.mp4") -> str:
        """Compile final video with all components."""
        try:
            # Check if any clips already have audio (e.g., lipsync videos)
            # Wav2Lip outputs include audio, so we should use that instead of narration
            clips_with_audio = []
            for clip in clips:
                try:
                    # Check if clip has audio stream
                    probe_cmd = [
                        'ffprobe', '-v', 'error',
                        '-select_streams', 'a',
                        '-show_entries', 'stream=codec_type',
                        '-of', 'default=nw=1:nk=1',
                        clip
                    ]
                    result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=2)
                    has_audio = 'audio' in result.stdout.lower() or (result.returncode == 0 and result.stdout.strip())
                    clips_with_audio.append(has_audio)
                    if has_audio:
                        logger.info(f"🎵 Clip {Path(clip).name} already contains audio (likely lipsync video)")
                except Exception:
                    clips_with_audio.append(False)
            
            # If ALL clips have audio (e.g., all are lipsync videos), use their audio instead of narration
            all_clips_have_audio = all(clips_with_audio) and len(clips_with_audio) > 0
            
            if all_clips_have_audio:
                logger.info("🎵 All clips have audio (lipsync videos) - will use video audio instead of separate narration")
                narration_audio_to_use = None  # Don't add narration, use audio from videos
            elif any(clips_with_audio):
                logger.warning(f"⚠️ Some clips have audio, some don't - using narration audio (may cause conflicts)")
                narration_audio_to_use = narration_audio
            else:
                narration_audio_to_use = narration_audio
            
            # Add cross-dissolve transitions between clips (0.25-0.4s, using 0.3s as default)
            transition_duration = 0.3  # 0.3s cross-dissolve (middle of 0.25-0.4s range)
            
            if len(clips) <= 1:
                # Single clip or no clips - no transitions needed
                if len(clips) == 0:
                    raise ValueError("No video clips provided")
                
                # Single clip - just copy it
                temp_video = "temp_video.mp4"
                cmd = [
                    'ffmpeg', '-y',
                    '-i', clips[0],
                    '-c', 'copy',
                    '-movflags', '+faststart',
                    temp_video
                ]
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            else:
                # Multiple clips - apply cross-dissolve transitions using fade filters
                # Apply fade-out to end of clips (except last) and fade-in to start (except first)
                # Then concatenate - creates smooth cross-dissolve effect
                logger.info(f"🎬 Applying cross-dissolve transitions ({transition_duration}s) between {len(clips)} clips...")
                
                # Get durations of all clips
                clip_durations = []
                for clip in clips:
                    duration = self.get_video_duration(clip)
                    clip_durations.append(duration)
                
                # Build complex filter with fade in/out for cross-dissolve effect
                temp_video = "temp_video.mp4"
                input_args = []
                filter_parts = []
                concat_inputs = []
                
                # Add all clips as inputs and apply fade filters
                for i, clip in enumerate(clips):
                    input_args.extend(['-i', clip])
                    clip_dur = clip_durations[i]
                    
                    # Build fade filter for this clip
                    fade_filter = f"[{i}:v]"
                    
                    # Add fade-in to start (except first clip)
                    if i > 0:
                        fade_filter += f"fade=t=in:st=0:d={transition_duration},"
                    
                    # Add fade-out to end (except last clip)
                    if i < len(clips) - 1:
                        fade_start = max(0, clip_dur - transition_duration)
                        fade_filter += f"fade=t=out:st={fade_start}:d={transition_duration},"
                    
                    # Remove trailing comma and set output label
                    fade_filter = fade_filter.rstrip(',')
                    output_label = f"v{i}"
                    fade_filter += f"[{output_label}]"
                    filter_parts.append(fade_filter)
                    concat_inputs.append(output_label)
                
                # Concatenate all faded clips
                # If all clips have audio (lipsync videos), include audio in concatenation
                # Otherwise, only concatenate video (a=0 means no audio)
                concat_inputs_str = "".join([f"[{label}]" for label in concat_inputs])
                if all_clips_have_audio:
                    # Include audio in concatenation: v=1:a=1
                    # Need to also extract audio from each clip
                    audio_inputs = []
                    for i, clip in enumerate(clips):
                        audio_inputs.append(f"[{i}:a]")
                    audio_concat = "".join(audio_inputs)
                    concat_filter = f"{concat_inputs_str}concat=n={len(clips)}:v=1:a=0[vout];{audio_concat}concat=n={len(clips)}:v=0:a=1[aout]"
                    logger.info("🎵 Concatenating clips with audio (lipsync videos)")
                else:
                    # Only concatenate video (no audio): v=1:a=0
                    concat_filter = f"{concat_inputs_str}concat=n={len(clips)}:v=1:a=0[vout]"
                    logger.info("🎵 Concatenating clips without audio (will add narration separately)")

                filter_parts.append(concat_filter)
                filter_complex = ";".join(filter_parts)
                
                # Build ffmpeg command with fade transitions
                # Note: Transitions require re-encoding, but we use fast preset
                codec = self.config.codec
                preset = self.config.preset
                crf = self.config.crf
                logger.info(f"🎬 Applying {len(clips)-1} cross-dissolve transitions with {codec} (preset={preset}, crf={crf})...")
                cmd = [
                    'ffmpeg', '-y'
                ] + input_args + [
                    '-filter_complex', filter_complex,
                ]
                
                # Map video and audio (if present) from filter output
                if all_clips_have_audio:
                    cmd.extend(['-map', '[vout]', '-map', '[aout]'])
                else:
                    cmd.extend(['-map', '[vout]'])
                
                cmd.extend([
                    '-c:v', codec,
                    '-preset', preset,
                    '-crf', str(crf),
                    '-tune', self.config.tune,
                    '-pix_fmt', 'yuv420p',
                ])
                
                # Encode audio if present (from lipsync videos)
                if all_clips_have_audio:
                    cmd.extend(['-c:a', 'aac', '-b:a', self.config.audio_bitrate])
                
                cmd.extend([
                    '-movflags', '+faststart',
                    temp_video
                ])
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                if not os.path.exists(temp_video) or os.path.getsize(temp_video) == 0:
                    error_msg = result.stderr if result.stderr else "Unknown error"
                    logger.error(f"❌ FFmpeg fade transition failed: {error_msg}")
                    # Fallback to simple concat if fade fails
                    logger.warning("⚠️ Falling back to simple concatenation without transitions")
                    concat_file = "concat_list.txt"
                    with open(concat_file, 'w') as f:
                        for clip in clips:
                            f.write(f"file '{clip}'\n")
                    cmd = [
                        'ffmpeg', '-y',
                        '-f', 'concat',
                        '-safe', '0',
                        '-i', concat_file,
                        '-c', 'copy',
                        '-movflags', '+faststart',
                        temp_video
                    ]
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                    try:
                        os.remove(concat_file)
                    except:
                        pass
            
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

            # If narration_audio_to_use is a list, first concatenate into one track
            if narration_audio_to_use and isinstance(narration_audio_to_use, list):
                merged_narration = 'merged_narration.aac'
                narration_audio_to_use = self.concat_audios(narration_audio_to_use, merged_narration)
            
            # Probe narration audio duration (if we're using it)
            narration_duration = None
            if narration_audio_to_use and os.path.exists(narration_audio_to_use):
                try:
                    probe_cmd = [
                        'ffprobe', '-v', 'error',
                        '-show_entries', 'format=duration',
                        '-of', 'default=nw=1:nk=1',
                        narration_audio_to_use
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
            # Use narration_audio_to_use (already processed - string or None, not a list)
            audio_inputs = []
            if narration_audio_to_use:
                audio_inputs.extend(['-i', narration_audio_to_use])
            have_music = bool(background_music and os.path.exists(background_music))
            if have_music:
                audio_inputs.extend(['-i', background_music])
            
            # Build FFmpeg command - try GPU encoding first
            # Try GPU NVENC first (much faster - 5-10x speedup)
            use_gpu = False
            nvenc_codec = None
            try:
                # Check if NVENC is available by trying to list encoders
                check_cmd = ['ffmpeg', '-hide_banner', '-encoders']
                result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=5)
                if 'h264_nvenc' in result.stdout:
                    use_gpu = True
                    nvenc_codec = "h264_nvenc"  # Always use H.264 for speed
                    logger.info(f"🚀 GPU NVENC detected - will use {nvenc_codec} for final encoding")
                else:
                    logger.info("💻 GPU NVENC not available - will use CPU encoding")
            except Exception as e:
                logger.warning(f"⚠️ Could not check for GPU encoders: {e}")
                logger.info("💻 Will use CPU encoding")
            
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
            
            # Final output settings - try GPU first, fall back to CPU
            codec = self.config.codec
            preset = self.config.preset
            crf = self.config.crf
            
            if use_gpu and nvenc_codec:
                # GPU encoding (much faster - 5-10x speedup)
                logger.info(f"🚀 Final encoding: GPU ({nvenc_codec})")
                
                # Use CRF mode for better quality/size ratio (NVENC supports constqp)
                # CRF 23 = good quality, reasonable file size (similar to libx264 CRF 22)
                # Fallback to lower bitrate VBR if CRF not supported
                use_crf = self.config.export_mode == "dev" or os.getenv("NVENC_USE_CRF", "1").lower() in ("1", "true", "yes")
                
                if use_crf:
                    # Use constant quality mode (better quality/size ratio)
                    # NVENC CQ values: 0-51 (lower = better quality, larger files)
                    # CQ 23 ≈ libx264 CRF 22, CQ 24 ≈ libx264 CRF 23
                    target_cq = 24 if self.config.export_mode == "dev" else 23
                    cmd.extend([
                        '-c:v', nvenc_codec,
                        '-preset', 'p1',  # p1 = fastest, p7 = slowest (best quality)
                        '-rc', 'constqp',  # Constant quality mode
                        '-cq', str(target_cq),  # Constant quality (0-51, lower = better)
                        '-c:a', 'aac',
                        '-b:a', self.config.audio_bitrate,
                        '-pix_fmt', 'yuv420p'
                    ])
                    logger.info(f"   Using CQ mode (cq={target_cq}) for better quality/size ratio")
                else:
                    # Use VBR with bitrate based on export mode
                    if self.config.export_mode == "dev":
                        target_bitrate = '3M'  # Lower bitrate for dev mode
                        max_bitrate = '6M'
                    else:
                        target_bitrate = '5M'  # Reduced from 10M for better file sizes
                        max_bitrate = '10M'    # Reduced from 20M
                    
                    cmd.extend([
                        '-c:v', nvenc_codec,
                        '-preset', 'p1',  # p1 = fastest, p7 = slowest (best quality)
                        '-rc', 'vbr',  # Variable bitrate mode
                        '-b:v', target_bitrate,  # Target bitrate
                        '-maxrate', max_bitrate,  # Max bitrate
                        '-c:a', 'aac',
                        '-b:a', self.config.audio_bitrate,
                        '-pix_fmt', 'yuv420p'
                    ])
                    logger.info(f"   Using VBR mode (bitrate={target_bitrate}, max={max_bitrate})")
            else:
                # CPU encoding (libx264 with optimized preset)
                logger.info(f"💻 Final encoding: CPU ({codec}, preset={preset}, crf={crf})")
                cmd.extend([
                    '-c:v', codec,
                    '-preset', preset,
                    '-crf', str(crf),
                    '-tune', self.config.tune,
                    '-c:a', 'aac',
                    '-b:a', self.config.audio_bitrate,
                    '-pix_fmt', 'yuv420p'
                ])
            
            # Don't trim video after stitching - use full video duration
            # Video was already synced/extended to match audio during scene processing
            # Trimming here causes loss of content (~1 second)
            # Do NOT use -t or -shortest; we want full video length
            if self.config.faststart:
                cmd.extend(['-movflags', '+faststart'])
            cmd.append(output_path)
            
            # Run FFmpeg command and check for errors
            # Try GPU first, fall back to CPU if it fails
            try:
                if use_gpu and nvenc_codec:
                    logger.info(f"🚀 Encoding final video with GPU ({nvenc_codec})...")
                else:
                    logger.info(f"💻 Encoding final video with CPU ({self.config.codec})...")
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                if use_gpu and nvenc_codec:
                    logger.info("✅ Final video encoded using GPU NVENC")
                else:
                    logger.info("✅ Final video encoded using CPU")
            except subprocess.CalledProcessError as e:
                # If GPU encoding failed, try CPU fallback
                if use_gpu and nvenc_codec:
                    error_output = e.stderr if e.stderr else "Unknown error"
                    logger.warning(f"⚠️ GPU encoding failed: {error_output[:300]}")
                    logger.info("🔄 Falling back to CPU encoding...")
                    # Rebuild command with CPU codec
                    cmd_cpu = ['ffmpeg', '-y', '-i', temp_video] + audio_inputs
                    if have_music:
                        cmd_cpu.extend([
                            '-filter_complex', '[1:a]volume=0.85[a1];[2:a]volume=0.15[a2];[a1][a2]amix=inputs=2:duration=longest,apad[aout]',
                            '-map', '0:v',
                            '-map', '[aout]'
                        ])
                    else:
                        cmd_cpu.extend(['-map', '0:v', '-map', '1:a', '-af', 'apad'])
                    if subtitles_path and os.path.exists(subtitles_path):
                        cmd_cpu.extend([
                            '-vf', f'subtitles={subtitles_path}:force_style=\'FontSize=32,PrimaryColour=&Hffffff,OutlineColour=&H000000,BackColour=&H000000,Bold=1\''
                        ])
                    cmd_cpu.extend([
                        '-c:v', self.config.codec,
                        '-preset', self.config.preset,
                        '-crf', str(self.config.crf),
                        '-c:a', 'aac',
                        '-b:a', self.config.audio_bitrate,
                        '-pix_fmt', 'yuv420p'
                    ])
                    cmd_cpu.extend(['-tune', self.config.tune])
                    # Don't trim video - use full video duration
                    if self.config.faststart:
                        cmd_cpu.extend(['-movflags', '+faststart'])
                    cmd_cpu.append(output_path)
                    result = subprocess.run(cmd_cpu, check=True, capture_output=True, text=True)
                    logger.info("✅ Final video encoded using CPU (fallback)")
                else:
                    raise  # Re-raise if already using CPU
            
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

