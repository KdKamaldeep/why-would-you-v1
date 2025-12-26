#!/usr/bin/env python3
"""
Script to remove subtitles from final_reel.mp4 files in output folder.
Regenerates final_reel.mp4 from existing scene videos and audio files without subtitles.
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_ffmpeg():
    """Check if FFmpeg is available."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def remove_subtitles(input_path: Path, output_path: Path = None, overwrite: bool = False) -> bool:
    """
    Remove subtitles from a video file using FFmpeg.
    
    Args:
        input_path: Path to input video file
        output_path: Path to output video file (if None, overwrites input if overwrite=True)
        overwrite: If True and output_path is None, overwrites input file
        
    Returns:
        True if successful, False otherwise
    """
    if output_path is None:
        if overwrite:
            # Create temporary output file
            output_path = input_path.parent / f"{input_path.stem}_no_subs{input_path.suffix}"
        else:
            logger.error(f"No output path specified and overwrite=False")
            return False
    
    # FFmpeg command to remove all subtitle streams
    # -map 0 copies all streams except subtitles
    # -map -0:s excludes all subtitle streams
    cmd = [
        'ffmpeg',
        '-i', str(input_path),
        '-map', '0',           # Map all streams from input
        '-map', '-0:s',        # Exclude all subtitle streams
        '-c', 'copy',          # Copy streams without re-encoding (fast)
        '-y',                  # Overwrite output file if exists
        str(output_path)
    ]
    
    try:
        logger.info(f"Removing subtitles from: {input_path}")
        logger.info(f"Output: {output_path}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info(f"✅ Successfully removed subtitles: {output_path}")
        
        # If overwriting, replace original with new file
        if overwrite and output_path != input_path:
            input_path.unlink()
            output_path.rename(input_path)
            logger.info(f"✅ Replaced original file: {input_path}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ FFmpeg error: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"❌ Error processing {input_path}: {e}")
        return False


def find_final_reels(output_dir: Path) -> list:
    """
    Find all directories containing final_reel.mp4 files (recursively).
    Returns directories that have final_reel.mp4 and can be regenerated.
    
    Args:
        output_dir: Root output directory to search
        
    Returns:
        List of Path objects to directories containing final_reel.mp4
    """
    reel_dirs = []
    
    if not output_dir.exists():
        logger.error(f"Output directory does not exist: {output_dir}")
        return reel_dirs
    
    # Search recursively for final_reel.mp4 files
    for file_path in output_dir.rglob("final_reel.mp4"):
        reel_dir = file_path.parent
        if reel_dir not in reel_dirs:
            reel_dirs.append(reel_dir)
            logger.info(f"Found final_reel in: {reel_dir}")
    
    return reel_dirs


def find_scene_files(reel_dir: Path) -> tuple:
    """
    Find scene video files and audio files in a reel directory.
    Checks both the reel directory and a 'scenes' subdirectory.
    
    Args:
        reel_dir: Directory containing the reel files
        
    Returns:
        Tuple of (list of scene video paths, list of audio paths) sorted by scene number
    """
    scene_videos = []
    audio_files = []
    
    # Check scenes subdirectory first, then root directory
    search_dirs = [reel_dir / "scenes", reel_dir]
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
            
        # Find scene video files (scene_1.mp4, scene_2.mp4, etc.)
        for scene_file in sorted(search_dir.glob("scene_*.mp4")):
            if scene_file not in scene_videos:
                scene_videos.append(scene_file)
    
    # Find audio files in root directory (audio_scene_1.wav, audio_scene_1.aac, etc.)
    for audio_file in sorted(reel_dir.glob("audio_scene_*.*")):
        if audio_file.suffix.lower() in ['.wav', '.aac', '.mp3', '.m4a']:
            audio_files.append(audio_file)
    
    # Sort by scene number
    def get_scene_number(path: Path) -> int:
        try:
            # Extract number from filename like "scene_1.mp4" or "audio_scene_1.wav"
            name = path.stem  # This gives us "scene_1" or "audio_scene_1" (no extension)
            if 'scene_' in name:
                # For "scene_1" -> split('scene_')[1] = "1"
                # For "audio_scene_1" -> split('scene_')[1] = "1"
                num_str = name.split('scene_')[1]
                # Remove any remaining non-numeric characters (shouldn't be needed, but safe)
                num_str = ''.join(filter(str.isdigit, num_str))
                return int(num_str) if num_str else 0
            return 0
        except Exception as e:
            logger.warning(f"Could not extract scene number from {path}: {e}")
            return 0
    
    scene_videos.sort(key=get_scene_number)
    audio_files.sort(key=get_scene_number)
    
    # Debug: Log found files
    logger.info(f"Found scene videos: {[f.name for f in scene_videos]}")
    logger.info(f"Found audio files: {[f.name for f in audio_files]}")
    
    return scene_videos, audio_files


def get_video_format_from_script(reel_dir: Path) -> dict:
    """
    Try to read script.json to get video format settings.
    
    Args:
        reel_dir: Directory containing the reel
        
    Returns:
        Dict with format settings or defaults
    """
    script_path = reel_dir / "script.json"
    defaults = {
        "out_width": 1080,
        "out_height": 1920,
        "out_fps": 30,
        "vertical_mode": "pad"
    }
    
    if script_path.exists():
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                script_data = json.load(f)
                # Check if video_format is specified in script metadata
                # For now, default to shorts format (9:16)
                return defaults
        except Exception as e:
            logger.warning(f"Could not read script.json: {e}")
    
    return defaults


def regenerate_final_reel(reel_dir: Path, scene_videos: list, audio_files: list, output_suffix: str = "_clean") -> bool:
    """
    Regenerate final_reel.mp4 from scene videos and audio files without subtitles.
    
    Args:
        reel_dir: Directory containing the reel
        scene_videos: List of scene video file paths
        audio_files: List of audio file paths
        output_suffix: Suffix for output file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Import required modules
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))
        sys.path.insert(0, str(project_root / "src"))
        
        from src.core.video_processor import VideoProcessor, VideoConfig as VPConfig
        from src.render.reel_renderer import create_reel
        
        if not scene_videos:
            logger.error(f"No scene video files found in {reel_dir}")
            return False
        
        logger.info(f"Found {len(scene_videos)} scene video(s) and {len(audio_files)} audio file(s)")
        
        # Get video format settings (default to shorts format)
        format_settings = get_video_format_from_script(reel_dir)
        
        # Create video processor config
        vp_config = VPConfig()
        video_processor = VideoProcessor(vp_config)
        
        # Step 1: Create stitched video from scenes (without subtitles)
        stitched_output = reel_dir / f"stitched{output_suffix}.mp4"
        logger.info(f"Creating stitched video from {len(scene_videos)} scenes...")
        
        # Prepare audio paths (match scene count)
        # Match audio files to scenes by scene number, not by index
        def get_scene_num_from_path(path: Path) -> int:
            try:
                name = path.stem
                if 'scene_' in name:
                    num_str = name.split('scene_')[1]
                    num_str = ''.join(filter(str.isdigit, num_str))
                    return int(num_str) if num_str else 0
                return 0
            except:
                return 0
        
        # Create a mapping of scene number to audio file
        audio_by_scene = {get_scene_num_from_path(audio): audio for audio in audio_files}
        logger.info(f"Audio mapping: {[(k, v.name) for k, v in sorted(audio_by_scene.items())]}")
        
        narration_audio = []
        for scene_video in scene_videos:
            scene_num = get_scene_num_from_path(scene_video)
            logger.info(f"Processing scene_{scene_num}: {scene_video.name}")
            if scene_num in audio_by_scene:
                audio_path = str(audio_by_scene[scene_num])
                narration_audio.append(audio_path)
                logger.info(f"  ✓ Matched with audio: {audio_by_scene[scene_num].name}")
            else:
                narration_audio.append("")  # No audio for this scene
                logger.warning(f"  ⚠ No audio found for scene_{scene_num}.mp4")
        
        # Filter out empty audio paths and prepare narration_audio
        # If all scenes have audio, pass as list; otherwise filter empty strings
        filtered_narration_audio = [a for a in narration_audio if a]
        
        if not filtered_narration_audio:
            # No audio files found
            logger.warning("⚠️ No audio files found, creating video without audio")
            final_narration_audio = None
        elif len(filtered_narration_audio) == 1:
            # Single audio file
            final_narration_audio = filtered_narration_audio[0]
        else:
            # Multiple audio files - pass as list to be concatenated
            final_narration_audio = filtered_narration_audio
        
        logger.info(f"Using {len(filtered_narration_audio)} audio file(s) for {len(scene_videos)} scene(s)")
        
        # Compile stitched video WITHOUT subtitles (pass None for subtitles_path)
        video_processor.compile_final_video(
            clips=[str(v) for v in scene_videos],
            narration_audio=final_narration_audio,
            background_music=None,
            subtitles_path=None,  # No subtitles!
            output_path=str(stitched_output)
        )
        
        logger.info(f"✅ Stitched video created: {stitched_output}")
        
        # Step 2: Create final reel from stitched video
        final_reel_output = reel_dir / f"final_reel{output_suffix}.mp4"
        logger.info(f"Creating final reel...")
        
        # Get merged audio if multiple audio files
        voice_audio = None
        if len(audio_files) > 1:
            merged_audio = reel_dir / f"merged_voice{output_suffix}.aac"
            video_processor.concat_audios(
                [str(a) for a in audio_files],
                str(merged_audio)
            )
            voice_audio = str(merged_audio)
        elif len(audio_files) == 1:
            voice_audio = str(audio_files[0])
        
        # Create reel (vertical format, no subtitles)
        create_reel(
            stitched_video=str(stitched_output),
            voice_audio=voice_audio,
            music_path=None,
            output_path=str(final_reel_output),
            vertical_mode=format_settings["vertical_mode"],
            out_width=format_settings["out_width"],
            out_height=format_settings["out_height"],
            out_fps=format_settings["out_fps"],
            voice_volume=1.0,
            music_volume=0.12,
            verbose=False
        )
        
        logger.info(f"✅ Final reel regenerated without subtitles: {final_reel_output}")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Failed to import required modules: {e}")
        logger.error("Make sure you're running from the project root directory")
        import traceback
        logger.error(traceback.format_exc())
        return False
    except Exception as e:
        logger.error(f"❌ Error regenerating final reel: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def find_subtitle_files(output_dir: Path) -> list:
    """
    Find all subtitle files (.srt) in output directory (recursively).
    
    Args:
        output_dir: Root output directory to search
        
    Returns:
        List of Path objects to .srt files
    """
    subtitle_files = []
    
    if not output_dir.exists():
        return subtitle_files
    
    # Search recursively for .srt files
    for file_path in output_dir.rglob("*.srt"):
        subtitle_files.append(file_path)
        logger.info(f"Found subtitle file: {file_path}")
    
    return subtitle_files


def remove_subtitle_file(subtitle_path: Path) -> bool:
    """
    Remove a subtitle file.
    
    Args:
        subtitle_path: Path to subtitle file to remove
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if subtitle_path.exists():
            subtitle_path.unlink()
            logger.info(f"✅ Removed subtitle file: {subtitle_path}")
            return True
        else:
            logger.warning(f"Subtitle file does not exist: {subtitle_path}")
            return False
    except Exception as e:
        logger.error(f"❌ Error removing subtitle file {subtitle_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Remove subtitles from final_reel.mp4 files in output folder"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory to search for final_reel.mp4 files (default: output)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite original files (default: create new files with _no_subs suffix)"
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_no_subs",
        help="Suffix for output files when not overwriting (default: _no_subs)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually processing"
    )
    parser.add_argument(
        "--remove-srt-files",
        action="store_true",
        help="Also remove .srt subtitle files from the output directory (default: only remove subtitle streams from video)"
    )
    
    args = parser.parse_args()
    
    # Check FFmpeg
    if not check_ffmpeg():
        logger.error("❌ FFmpeg not found. Please install FFmpeg first.")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    
    # Find all final_reel.mp4 files
    logger.info(f"Searching for final_reel.mp4 files in: {output_dir}")
    final_reels = find_final_reels(output_dir)
    
    if not final_reels:
        logger.warning(f"No final_reel.mp4 files found in {output_dir}")
        sys.exit(0)
    
    logger.info(f"Found {len(final_reels)} final_reel.mp4 file(s)")
    
    # Find subtitle files if requested
    subtitle_files = []
    if args.remove_srt_files:
        logger.info(f"Searching for .srt subtitle files in: {output_dir}")
        subtitle_files = find_subtitle_files(output_dir)
        if subtitle_files:
            logger.info(f"Found {len(subtitle_files)} subtitle file(s)")
    
    if args.dry_run:
        logger.info("DRY RUN - Reels that would be regenerated:")
        for reel_dir in final_reels:
            scene_videos, audio_files = find_scene_files(reel_dir)
            logger.info(f"  Would regenerate: {reel_dir}")
            logger.info(f"    Scenes: {len(scene_videos)}")
            logger.info(f"    Audio files: {len(audio_files)}")
            output_path = reel_dir / f"final_reel{args.output_suffix}.mp4"
            logger.info(f"    Output: {output_path}")
        
        if args.remove_srt_files and subtitle_files:
            logger.info("\nSubtitle files that would be removed:")
            for srt_file in subtitle_files:
                logger.info(f"  Would remove: {srt_file}")
        sys.exit(0)
    
    # Process each reel directory
    successful = []
    failed = []
    
    for reel_dir in final_reels:
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing reel directory: {reel_dir}")
        logger.info(f"{'='*70}")
        
        # Find scene videos and audio files
        scene_videos, audio_files = find_scene_files(reel_dir)
        
        if not scene_videos:
            logger.warning(f"⚠️ No scene video files found in {reel_dir}, skipping")
            failed.append((reel_dir, "No scene videos found"))
            continue
        
        # Regenerate final_reel without subtitles
        if regenerate_final_reel(reel_dir, scene_videos, audio_files, args.output_suffix):
            successful.append(reel_dir)
        else:
            failed.append((reel_dir, "Regeneration failed"))
    
    # Remove subtitle files if requested
    removed_srt_files = []
    failed_srt_removals = []
    
    if args.remove_srt_files and subtitle_files:
        logger.info(f"\n{'='*70}")
        logger.info("REMOVING SUBTITLE FILES")
        logger.info(f"{'='*70}")
        
        for srt_file in subtitle_files:
            if remove_subtitle_file(srt_file):
                removed_srt_files.append(srt_file)
            else:
                failed_srt_removals.append(srt_file)
    
    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"✅ Reels regenerated: {len(successful)}/{len(final_reels)}")
    for reel_dir in successful:
        output_file = reel_dir / f"final_reel{args.output_suffix}.mp4"
        logger.info(f"   ✓ {reel_dir} -> {output_file}")
    
    if failed:
        logger.info(f"\n❌ Reels failed: {len(failed)}/{len(final_reels)}")
        for reel_dir, error in failed:
            logger.info(f"   ✗ {reel_dir}: {error}")
    
    if args.remove_srt_files:
        logger.info(f"\n📝 Subtitle files removed: {len(removed_srt_files)}/{len(subtitle_files)}")
        for srt_file in removed_srt_files:
            logger.info(f"   ✓ Removed: {srt_file}")
        
        if failed_srt_removals:
            logger.info(f"\n❌ Subtitle files failed: {len(failed_srt_removals)}/{len(subtitle_files)}")
            for srt_file in failed_srt_removals:
                logger.info(f"   ✗ {srt_file}")
    
    logger.info(f"{'='*70}")
    
    sys.exit(0 if not failed and not failed_srt_removals else 1)


if __name__ == "__main__":
    main()
