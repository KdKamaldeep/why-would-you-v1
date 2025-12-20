#!/usr/bin/env python3
"""
Reel Renderer - Creates platform-ready vertical Reels/Shorts videos
Outputs 1080x1920, 30fps, H.264/AAC format for YouTube/Instagram
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def run_ffmpeg(cmd: list, verbose: bool = False) -> None:
    """
    Run FFmpeg command with proper error handling.
    
    Args:
        cmd: FFmpeg command as list of arguments
        verbose: If True, print the command before executing
        
    Raises:
        RuntimeError: If FFmpeg command fails
    """
    if verbose:
        logger.info(f"🔧 Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        if verbose and result.stdout:
            logger.debug(f"FFmpeg stdout: {result.stdout}")
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode('utf-8', errors='ignore') if isinstance(e.stderr, bytes) else str(e.stderr)
        logger.error(f"❌ FFmpeg command failed: {' '.join(cmd)}")
        logger.error(f"Error output: {error_msg}")
        raise RuntimeError(f"FFmpeg failed: {error_msg}") from e
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Please install FFmpeg and ensure it's in your PATH.")


def make_vertical(
    input_mp4: str,
    output_mp4: str,
    mode: str = "pad",
    out_w: int = 1080,
    out_h: int = 1920,
    fps: int = 30,
    verbose: bool = False
) -> str:
    """
    Convert video to vertical format (1080x1920) with pad or crop mode.
    
    Args:
        input_mp4: Input video path
        output_mp4: Output video path
        mode: "pad" (safe, no cropping) or "crop" (fills frame)
        out_w: Output width (default: 1080)
        out_h: Output height (default: 1920)
        fps: Output FPS (default: 30)
        verbose: Print FFmpeg commands
        
    Returns:
        Path to output video
    """
    input_path = Path(input_mp4)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_mp4}")
    
    output_path = Path(output_mp4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📐 Converting to vertical format: {out_w}x{out_h} @ {fps}fps")
    logger.info(f"🎬 Mode: {mode}")
    
    # Build video filter based on mode
    if mode == "pad":
        # Scale to fit width, pad to height (safe for faces)
        vf = f"scale={out_w}:-2,pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2"
    elif mode == "crop":
        # Scale to fit height, crop to width (fills frame)
        vf = f"scale=-2:{out_h},crop={out_w}:{out_h}"
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'pad' or 'crop'")
    
    cmd = [
        'ffmpeg', '-y',
        '-i', str(input_path),
        '-vf', vf,
        '-r', str(fps),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'medium',
        '-crf', '23',  # Good quality for social media
        '-movflags', '+faststart',  # Web optimization
        str(output_path)
    ]
    
    run_ffmpeg(cmd, verbose=verbose)
    logger.info(f"✅ Vertical video created: {output_path}")
    return str(output_path)


def mix_audio(
    video_mp4: str,
    voice_wav: Optional[str],
    music_path: Optional[str],
    output_mp4: str,
    voice_vol: float = 1.0,
    music_vol: float = 0.12,
    fps: int = 30,
    out_w: int = 1080,
    out_h: int = 1920,
    mode: str = "pad",
    verbose: bool = False
) -> str:
    """
    Mix voiceover and optional background music with video.
    Creates final platform-ready reel.
    
    Args:
        video_mp4: Input video path (already vertical format)
        voice_wav: Voiceover audio path (optional)
        music_path: Background music path (optional)
        output_mp4: Output video path
        voice_vol: Voice volume multiplier (default: 1.0)
        music_vol: Music volume multiplier (default: 0.12)
        fps: Output FPS (default: 30)
        out_w: Output width (default: 1080)
        out_h: Output height (default: 1920)
        mode: Vertical mode (pad/crop) - used if video needs conversion
        verbose: Print FFmpeg commands
        
    Returns:
        Path to output video
    """
    video_path = Path(video_mp4)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_mp4}")
    
    output_path = Path(output_mp4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if video needs vertical conversion
    needs_vertical = False
    try:
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'csv=p=0',
            str(video_path)
        ]
        result = subprocess.run(probe_cmd, check=True, capture_output=True, text=True)
        info = result.stdout.strip().split(',')
        if len(info) >= 2 and info[0] and info[1]:
            width = int(info[0])
            height = int(info[1])
            if width != out_w or height != out_h:
                needs_vertical = True
            else:
                logger.info(f"✅ Video already in target format: {width}x{height}")
        else:
            needs_vertical = True
    except Exception as e:
        logger.warning(f"Could not probe video dimensions: {e}, assuming needs conversion")
        needs_vertical = True
    
    # Convert to vertical if needed
    if needs_vertical:
        temp_vertical = str(output_path.parent / "_temp_vertical.mp4")
        logger.info("📐 Converting video to vertical format first...")
        make_vertical(str(video_path), temp_vertical, mode, out_w, out_h, fps, verbose)
        video_path = Path(temp_vertical)
    else:
        temp_vertical = None
    
    logger.info(f"🎵 Mixing audio: voice={voice_wav is not None}, music={music_path is not None}")
    logger.info(f"🔊 Volume levels: voice={voice_vol}, music={music_vol}")
    
    # Build FFmpeg command
    cmd = ['ffmpeg', '-y', '-i', str(video_path)]
    input_count = 1  # Video is input 0
    
    # Add voice audio if provided
    has_voice = voice_wav and Path(voice_wav).exists()
    if has_voice:
        cmd.extend(['-i', voice_wav])
        input_count += 1
        voice_input_idx = input_count - 1
    else:
        voice_input_idx = None
    
    # Add music if provided
    has_music = music_path and Path(music_path).exists()
    if has_music:
        cmd.extend(['-i', music_path])
        input_count += 1
        music_input_idx = input_count - 1
    else:
        music_input_idx = None
    
    # Build filter_complex for audio mixing
    if has_voice and has_music:
        # Mix both voice and music
        filter_complex = (
            f"[{voice_input_idx}:a]volume={voice_vol}[v];"
            f"[{music_input_idx}:a]volume={music_vol}[m];"
            f"[v][m]amix=inputs=2:dropout_transition=2[a]"
        )
        cmd.extend(['-filter_complex', filter_complex])
        cmd.extend(['-map', '0:v', '-map', '[a]'])
    elif has_voice:
        # Voice only
        filter_complex = f"[{voice_input_idx}:a]volume={voice_vol}[a]"
        cmd.extend(['-filter_complex', filter_complex])
        cmd.extend(['-map', '0:v', '-map', '[a]'])
    elif has_music:
        # Music only
        filter_complex = f"[{music_input_idx}:a]volume={music_vol}[a]"
        cmd.extend(['-filter_complex', filter_complex])
        cmd.extend(['-map', '0:v', '-map', '[a]'])
    else:
        # No audio - create silent audio track
        logger.info("ℹ️ No audio provided, creating silent audio track")
        # Generate silent audio using lavfi
        cmd.extend(['-f', 'lavfi', '-i', 'anullsrc=channel_layout=stereo:sample_rate=48000'])
        cmd.extend(['-map', '0:v', '-map', '1:a'])
    
    # Output settings
    cmd.extend([
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'medium',
        '-crf', '23',
        '-r', str(fps),
        '-c:a', 'aac',
        '-b:a', '192k',
        '-ar', '48000',  # 48kHz for platform compatibility
        '-shortest',  # Match shortest stream
        '-movflags', '+faststart',
        str(output_path)
    ])
    
    try:
        run_ffmpeg(cmd, verbose=verbose)
        logger.info(f"✅ Final reel created: {output_path}")
    finally:
        # Clean up temp file if created
        if temp_vertical and Path(temp_vertical).exists():
            try:
                os.remove(temp_vertical)
            except Exception as e:
                logger.warning(f"Could not remove temp file {temp_vertical}: {e}")
    
    return str(output_path)


def create_reel(
    stitched_video: str,
    voice_audio: Optional[str],
    music_path: Optional[str],
    output_path: str,
    vertical_mode: str = "pad",
    out_width: int = 1080,
    out_height: int = 1920,
    out_fps: int = 30,
    voice_volume: float = 1.0,
    music_volume: float = 0.12,
    verbose: bool = False
) -> str:
    """
    Complete reel creation pipeline: vertical conversion + audio mixing.
    
    Args:
        stitched_video: Path to stitched scene video
        voice_audio: Path to voiceover audio (optional)
        music_path: Path to background music (optional)
        output_path: Final output path
        vertical_mode: "pad" or "crop" (default: "pad")
        out_width: Output width (default: 1080)
        out_height: Output height (default: 1920)
        out_fps: Output FPS (default: 30)
        voice_volume: Voice volume (default: 1.0)
        music_volume: Music volume (default: 0.12)
        verbose: Print FFmpeg commands
        
    Returns:
        Path to final reel
    """
    logger.info("🎬 Creating platform-ready reel...")
    logger.info(f"📐 Format: {out_width}x{out_height} @ {out_fps}fps")
    logger.info(f"🎬 Mode: {vertical_mode}")
    
    # Create vertical video first (if needed, mix_audio will handle it)
    # Then mix audio
    return mix_audio(
        video_mp4=stitched_video,
        voice_wav=voice_audio,
        music_path=music_path,
        output_mp4=output_path,
        voice_vol=voice_volume,
        music_vol=music_volume,
        fps=out_fps,
        out_w=out_width,
        out_h=out_height,
        mode=vertical_mode,
        verbose=verbose
    )
