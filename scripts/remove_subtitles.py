#!/usr/bin/env python3
"""
Script to remove subtitles from final_reel.mp4 files in output folder.
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse
import logging

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
    Find all final_reel.mp4 files in output directory (recursively).
    
    Args:
        output_dir: Root output directory to search
        
    Returns:
        List of Path objects to final_reel.mp4 files
    """
    final_reels = []
    
    if not output_dir.exists():
        logger.error(f"Output directory does not exist: {output_dir}")
        return final_reels
    
    # Search recursively for final_reel.mp4 files
    for file_path in output_dir.rglob("final_reel.mp4"):
        final_reels.append(file_path)
        logger.info(f"Found: {file_path}")
    
    return final_reels


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
        logger.info("DRY RUN - Files that would be processed:")
        for file_path in final_reels:
            if args.overwrite:
                logger.info(f"  Would process: {file_path} (overwrite)")
            else:
                output_path = file_path.parent / f"{file_path.stem}{args.output_suffix}{file_path.suffix}"
                logger.info(f"  Would process: {file_path} -> {output_path}")
        
        if args.remove_srt_files and subtitle_files:
            logger.info("\nSubtitle files that would be removed:")
            for srt_file in subtitle_files:
                logger.info(f"  Would remove: {srt_file}")
        sys.exit(0)
    
    # Process each file
    successful = []
    failed = []
    
    for file_path in final_reels:
        if args.overwrite:
            output_path = None
        else:
            output_path = file_path.parent / f"{file_path.stem}{args.output_suffix}{file_path.suffix}"
        
        if remove_subtitles(file_path, output_path, args.overwrite):
            successful.append(file_path)
        else:
            failed.append(file_path)
    
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
    logger.info(f"✅ Videos processed: {len(successful)}/{len(final_reels)}")
    for file_path in successful:
        logger.info(f"   ✓ {file_path}")
    
    if failed:
        logger.info(f"\n❌ Videos failed: {len(failed)}/{len(final_reels)}")
        for file_path in failed:
            logger.info(f"   ✗ {file_path}")
    
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
