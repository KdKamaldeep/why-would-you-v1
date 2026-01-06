#!/usr/bin/env python3
"""
Reel Renderer - Creates platform-ready vertical Reels/Shorts videos
Outputs 1080x1920, 30fps, H.264/AAC format for YouTube/Instagram
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Optional, List, Tuple

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
    verbose: bool = False,
    add_hooks: bool = False,
    top_hook_text: Optional[str] = None,
    bottom_hook_text: Optional[str] = None,
    scene_hooks: Optional[List[Tuple[float, float, str]]] = None
) -> str:
    # Default scene_hooks to empty list if None
    if scene_hooks is None:
        scene_hooks = []
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
    # Skip conversion if video is already at perfect shorts dimensions (768x1344)
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
            # Skip conversion if video is already perfect for shorts (768x1344) or matches target
            if (width == 768 and height == 1344) or (width == out_w and height == out_h):
                needs_vertical = False
                logger.info(f"✅ Video already perfect for shorts: {width}x{height} - skipping conversion")
            elif width != out_w or height != out_h:
                needs_vertical = True
            else:
                logger.info(f"✅ Video already in target format: {width}x{height}")
        else:
            needs_vertical = True
    except Exception as e:
        logger.warning(f"Could not probe video dimensions: {e}, assuming needs conversion")
        needs_vertical = True
    
    # Convert to vertical if needed (with hook text support)
    # Skip if video is already 768x1344 (perfect for shorts)
    if needs_vertical:
        temp_vertical = str(output_path.parent / "_temp_vertical.mp4")
        logger.info("📐 Converting video to vertical format first...")
        # If we need hook texts on black areas, we'll add them in mix_audio filter_complex
        # Otherwise just do the normal vertical conversion
        make_vertical(str(video_path), temp_vertical, mode, out_w, out_h, fps, verbose)
        video_path = Path(temp_vertical)
    else:
        temp_vertical = None
    
    logger.info(f"🎵 Mixing audio: voice={voice_wav is not None}, music={music_path is not None}")
    logger.info(f"🔊 Volume levels: voice={voice_vol}, music={music_vol}")
    
    # Get durations to determine if video needs looping
    video_duration = None
    narration_duration = None
    
    try:
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=nw=1:nk=1',
            str(video_path)
        ]
        result = subprocess.run(probe_cmd, check=True, capture_output=True, text=True)
        video_duration = float(result.stdout.strip())
    except Exception:
        pass
    
    has_voice = voice_wav and Path(voice_wav).exists()
    if has_voice:
        try:
            probe_cmd = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=nw=1:nk=1',
                voice_wav
            ]
            result = subprocess.run(probe_cmd, check=True, capture_output=True, text=True)
            narration_duration = float(result.stdout.strip())
        except Exception:
            pass
    
    # If narration is longer than video, loop the video to match
    if video_duration and narration_duration and narration_duration > video_duration:
        logger.info(f"📹 Video ({video_duration:.2f}s) is shorter than narration ({narration_duration:.2f}s)")
        logger.info(f"🔄 Looping video to match narration length...")
        
        # Calculate how many loops needed
        loops_needed = int(narration_duration / video_duration) + 1
        logger.info(f"   Looping video {loops_needed} times to cover {narration_duration:.2f}s")
        
        # Create a concat file with the video repeated
        loop_concat_file = str(output_path.parent / "_loop_concat.txt")
        with open(loop_concat_file, 'w') as f:
            for _ in range(loops_needed):
                f.write(f"file '{video_path}'\n")
        
        # Create looped video
        looped_video = str(output_path.parent / "_temp_looped.mp4")
        loop_cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', loop_concat_file,
            '-c', 'copy',
            '-movflags', '+faststart',
            looped_video
        ]
        run_ffmpeg(loop_cmd, verbose=verbose)
        
        # Trim to exact narration duration
        final_looped_video = str(output_path.parent / "_temp_final_looped.mp4")
        trim_cmd = [
            'ffmpeg', '-y',
            '-i', looped_video,
            '-t', f"{narration_duration:.3f}",
            '-c', 'copy',
            final_looped_video
        ]
        run_ffmpeg(trim_cmd, verbose=verbose)
        
        # Replace video_path with looped version
        video_path = Path(final_looped_video)
        logger.info(f"✅ Video extended to {narration_duration:.2f}s to match narration")
        
        # Clean up intermediate files
        try:
            if Path(looped_video).exists():
                os.remove(looped_video)
            if Path(loop_concat_file).exists():
                os.remove(loop_concat_file)
        except Exception as e:
            logger.warning(f"Could not clean up loop files: {e}")
    
    # Build FFmpeg command
    cmd = ['ffmpeg', '-y', '-i', str(video_path)]
    input_count = 1  # Video is input 0
    
    # Add voice audio if provided
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
    
    # Build video filter for text overlays
    # Determine if we need to add hook texts
    has_top_bottom_hooks = add_hooks and (top_hook_text or bottom_hook_text)
    has_scene_hooks = len(scene_hooks) > 0
    
    # Build filter_complex for audio mixing and video
    filter_complex_parts = []
    
    # Build video filter with drawtext if needed
    if has_top_bottom_hooks or has_scene_hooks:
        # Build drawtext filters (chain them with commas)
        drawtext_filters = []
        
        # Helper function to escape text for drawtext
        def escape_text(text):
            # Escape single quotes, colons, and backslashes
            return text.replace('\\', '\\\\').replace("'", "\\'").replace(':', '\\:')
        
        # Top and bottom hook texts (only if add_hooks is True)
        if has_top_bottom_hooks:
            if top_hook_text:
                # Position at top center of black area (in pad mode, black area is at top)
                # Font size: 48px, white text, black outline
                drawtext_filters.append(
                    f"drawtext=text='{escape_text(top_hook_text)}':"
                    f"fontsize=48:fontcolor=white:"
                    f"x=(w-text_w)/2:y=50:"
                    f"box=1:boxcolor=black@0.7:boxborderw=5:"
                    f"borderw=2:bordercolor=black"
                )
            if bottom_hook_text:
                # Position at bottom center of black area
                drawtext_filters.append(
                    f"drawtext=text='{escape_text(bottom_hook_text)}':"
                    f"fontsize=48:fontcolor=white:"
                    f"x=(w-text_w)/2:y=h-th-50:"
                    f"box=1:boxcolor=black@0.7:boxborderw=5:"
                    f"borderw=2:bordercolor=black"
                )
        
        # Scene hook texts (always show if present, bold style at top with white background)
        if has_scene_hooks:
            for start_time, end_time, hook_text in scene_hooks:
                # Bold, catchy style: large font, top of video, white background, black text
                # Large font size + thick border creates bold appearance
                drawtext_filters.append(
                    f"drawtext=text='{escape_text(hook_text)}':"
                    f"fontsize=56:fontcolor=black:"
                    f"x=(w-text_w)/2:y=100:"
                    f"enable='between(t,{start_time},{end_time})':"
                    f"box=1:boxcolor=white:boxborderw=12:"
                    f"borderw=4:bordercolor=black"
                )
        
        if drawtext_filters:
            # Chain drawtext filters with commas
            video_filter = f"[0:v]{','.join(drawtext_filters)}[vout]"
            filter_complex_parts.append(video_filter)
            video_output = "[vout]"
        else:
            video_output = "0:v"
    else:
        video_output = "0:v"
    
    # Build audio filter
    if has_voice and has_music:
        # Mix both voice and music (use [voice] label instead of [v] to avoid conflict)
        filter_complex_parts.append(f"[{voice_input_idx}:a]volume={voice_vol}[voice]")
        filter_complex_parts.append(f"[{music_input_idx}:a]volume={music_vol}[music]")
        filter_complex_parts.append(f"[voice][music]amix=inputs=2:dropout_transition=2:duration=longest[a]")
        audio_output = "[a]"
    elif has_voice:
        # Voice only
        filter_complex_parts.append(f"[{voice_input_idx}:a]volume={voice_vol}[a]")
        audio_output = "[a]"
    elif has_music:
        # Music only
        filter_complex_parts.append(f"[{music_input_idx}:a]volume={music_vol}[a]")
        audio_output = "[a]"
    else:
        # No audio - create silent audio track
        logger.info("ℹ️ No audio provided, creating silent audio track")
        # Generate silent audio using lavfi
        cmd.extend(['-f', 'lavfi', '-i', 'anullsrc=channel_layout=stereo:sample_rate=48000'])
        audio_output = "1:a"
    
    # Apply filter_complex if we have any filters
    if filter_complex_parts:
        # Join all filter parts with semicolons (each part is a separate filter statement)
        filter_complex = ';'.join(filter_complex_parts)
        cmd.extend(['-filter_complex', filter_complex])
        cmd.extend(['-map', video_output, '-map', audio_output])
    else:
        # No filters at all
        cmd.extend(['-map', '0:v', '-map', audio_output])
    
    # Use narration duration if available, otherwise video duration
    target_duration = narration_duration if narration_duration else video_duration
    duration_args = []
    if target_duration:
        duration_args = ['-t', f"{target_duration:.3f}"]
    
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
    ] + duration_args + [
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
    verbose: bool = False,
    add_hooks: bool = False,
    top_hook_text: Optional[str] = None,
    bottom_hook_text: Optional[str] = None,
    scene_hooks: Optional[List[Tuple[float, float, str]]] = None  # List of (start_time, end_time, hook_text) tuples
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
        verbose=verbose,
        add_hooks=add_hooks,
        top_hook_text=top_hook_text,
        bottom_hook_text=bottom_hook_text,
        scene_hooks=(scene_hooks or []) if scene_hooks is not None else []
    )
