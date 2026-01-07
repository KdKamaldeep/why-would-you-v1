#!/usr/bin/env python3
"""
Test script to verify the video/audio compilation fix.
Tests compile_final_video with existing videos and audios in output folders.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Optional

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.video_processor import VideoProcessor, VideoConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_output_folders(base_path: str = "output") -> List[Path]:
    """Find all output folders containing videos."""
    base = Path(base_path)
    if not base.exists():
        logger.warning(f"Base output path does not exist: {base_path}")
        return []
    
    output_folders = []
    
    # Check base folder itself
    if (base / "scenes").exists():
        output_folders.append(base)
    
    # Check subdirectories
    for item in base.iterdir():
        if item.is_dir():
            scenes_dir = item / "scenes"
            if scenes_dir.exists():
                output_folders.append(item)
    
    return output_folders


def find_video_clips(output_folder: Path) -> List[str]:
    """Find all video clips in the scenes directory.
    Prioritizes lip_sync files, then synced files, then base files.
    """
    scenes_dir = output_folder / "scenes"
    if not scenes_dir.exists():
        return []
    
    # Collect all scene files and prioritize
    scene_files = {}
    for video_file in scenes_dir.glob("scene_*.mp4"):
        scene_num = None
        file_type = None
        
        # Extract scene number and file type
        if "_lipsync" in video_file.name:
            # scene_1_lipsync.mp4
            parts = video_file.stem.split("_")
            if len(parts) >= 2:
                try:
                    scene_num = int(parts[1])
                    file_type = "lipsync"
                except ValueError:
                    continue
        elif "_synced" in video_file.name:
            # scene_1_synced.mp4
            parts = video_file.stem.split("_")
            if len(parts) >= 2:
                try:
                    scene_num = int(parts[1])
                    file_type = "synced"
                except ValueError:
                    continue
        else:
            # scene_1.mp4 (base file)
            parts = video_file.stem.split("_")
            if len(parts) >= 2:
                try:
                    scene_num = int(parts[1])
                    file_type = "base"
                except ValueError:
                    continue
        
        if scene_num is not None:
            # Priority: lipsync > synced > base
            if scene_num not in scene_files:
                scene_files[scene_num] = {}
            scene_files[scene_num][file_type] = str(video_file)
    
    # Select best file for each scene (lipsync > synced > base)
    video_clips = []
    for scene_num in sorted(scene_files.keys()):
        files = scene_files[scene_num]
        if "lipsync" in files:
            video_clips.append(files["lipsync"])
            logger.debug(f"Scene {scene_num}: Using lipsync file")
        elif "synced" in files:
            video_clips.append(files["synced"])
            logger.debug(f"Scene {scene_num}: Using synced file")
        elif "base" in files:
            video_clips.append(files["base"])
            logger.debug(f"Scene {scene_num}: Using base file")
    
    return video_clips


def find_audio_clips(output_folder: Path) -> List[str]:
    """Find all audio clips in the output folder. Only returns _pp (post-processed) versions."""
    audio_clips = []
    seen_scenes = set()
    
    # Look for audio_scene_*_pp.wav files (post-processed versions only)
    for pp_audio_file in output_folder.glob("audio_scene_*_pp.wav"):
        audio_clips.append(str(pp_audio_file))
        # Track which scene this is to avoid duplicates
        scene_num = pp_audio_file.stem.replace("audio_scene_", "").replace("_pp", "")
        seen_scenes.add(scene_num)
    
    # Also check for pause audio files
    for pause_audio in output_folder.glob("pause_audio_*.aac"):
        audio_clips.append(str(pause_audio))
    
    return sorted(audio_clips)


def test_compile_video(
    video_processor: VideoProcessor,
    video_clips: List[str],
    audio_clips: List[str],
    output_folder: Path,
    test_name: str
) -> bool:
    """Test compile_final_video with given clips and audios."""
    logger.info(f"\n{'='*70}")
    logger.info(f"Testing: {test_name}")
    logger.info(f"{'='*70}")
    logger.info(f"📁 Output folder: {output_folder}")
    logger.info(f"🎬 Video clips: {len(video_clips)}")
    logger.info(f"🎵 Audio clips: {len(audio_clips)}")
    
    if not video_clips:
        logger.warning("⚠️ No video clips found - skipping test")
        return False
    
    if not audio_clips:
        logger.warning("⚠️ No audio clips found - will test without audio")
    
    # Show what we're testing
    logger.info("\n📋 Video clips:")
    for i, clip in enumerate(video_clips, 1):
        logger.info(f"   {i}. {Path(clip).name}")
    
    if audio_clips:
        logger.info("\n📋 Audio clips:")
        for i, audio in enumerate(audio_clips, 1):
            logger.info(f"   {i}. {Path(audio).name}")
    
    # Create test output path
    test_output = output_folder / f"test_compiled_{test_name}.mp4"
    
    try:
        logger.info(f"\n🔄 Testing compile_final_video...")
        logger.info(f"   Input: {len(video_clips)} videos, {len(audio_clips)} audios")
        logger.info(f"   Output: {test_output}")
        
        # Test with audio_clips as a list (the fix we're testing)
        result = video_processor.compile_final_video(
            clips=video_clips,
            narration_audio=audio_clips if audio_clips else None,  # Pass as list to test the fix
            background_music=None,
            subtitles_path=None,
            output_path=str(test_output)
        )
        
        if Path(result).exists():
            file_size = Path(result).stat().st_size / (1024 * 1024)  # MB
            logger.info(f"✅ Test PASSED: Video compiled successfully")
            logger.info(f"   Output: {result}")
            logger.info(f"   Size: {file_size:.2f} MB")
            return True
        else:
            logger.error(f"❌ Test FAILED: Output file not created")
            return False
            
    except TypeError as e:
        if "expected str, bytes or os.PathLike object, not list" in str(e):
            logger.error(f"❌ Test FAILED: The fix didn't work - still getting list error")
            logger.error(f"   Error: {e}")
            return False
        else:
            logger.error(f"❌ Test FAILED: TypeError: {e}")
            return False
    except Exception as e:
        logger.error(f"❌ Test FAILED: {type(e).__name__}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Main test function."""
    logger.info("🧪 Testing video/audio compilation fix")
    logger.info("=" * 70)
    
    # Initialize video processor with DEV mode for smaller test files
    try:
        config = VideoConfig(fps=24, width=1280, height=720)
        # Force dev mode for testing (smaller files, faster encoding)
        #config._export_mode = "dev"
        video_processor = VideoProcessor(config)
        logger.info("✅ VideoProcessor initialized (DEV mode - smaller files)")
    except Exception as e:
        logger.error(f"❌ Failed to initialize VideoProcessor: {e}")
        return 1
    
    # Find all output folders
    output_folders = find_output_folders()
    
    if not output_folders:
        logger.warning("⚠️ No output folders found")
        logger.info("💡 Looking in current directory...")
        output_folders = [Path(".")]
    
    logger.info(f"\n📁 Found {len(output_folders)} output folder(s)")
    
    # Test each output folder
    passed_tests = 0
    failed_tests = 0
    
    for output_folder in output_folders:
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing folder: {output_folder}")
        logger.info(f"{'='*70}")
        
        # Find videos and audios
        video_clips = find_video_clips(output_folder)
        audio_clips = find_audio_clips(output_folder)
        
        if not video_clips and not audio_clips:
            logger.info("ℹ️ No videos or audios found in this folder - skipping")
            continue
        
        # Test with videos and audios
        test_name = output_folder.name if output_folder.name else "root"
        success = test_compile_video(
            video_processor,
            video_clips,
            audio_clips,
            output_folder,
            test_name
        )
        
        if success:
            passed_tests += 1
        else:
            failed_tests += 1
    
    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"✅ Passed: {passed_tests}")
    logger.info(f"❌ Failed: {failed_tests}")
    logger.info(f"📊 Total: {passed_tests + failed_tests}")
    
    if failed_tests == 0:
        logger.info("\n🎉 All tests passed! The fix is working correctly.")
        return 0
    else:
        logger.error(f"\n⚠️ {failed_tests} test(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

