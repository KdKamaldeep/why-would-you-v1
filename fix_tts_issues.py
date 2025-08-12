#!/usr/bin/env python3
"""
Script to fix TTS and torchaudio compatibility issues.
"""

import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def fix_tts_issues():
    """Fix TTS and torchaudio compatibility issues."""
    try:
        logger.info("🔧 Fixing TTS and torchaudio compatibility issues...")
        
        # Update TTS to a compatible version
        logger.info("📦 Updating TTS to compatible version...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "--upgrade", "TTS>=0.25.0"
            ], check=True, capture_output=True, text=True)
            logger.info("✅ Successfully updated TTS")
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️ Failed to update TTS: {e}")
            logger.info(f"   Error output: {e.stderr}")
        
        # Update torchaudio to latest version
        logger.info("📦 Updating torchaudio...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "--upgrade", "torchaudio"
            ], check=True, capture_output=True, text=True)
            logger.info("✅ Successfully updated torchaudio")
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️ Failed to update torchaudio: {e}")
            logger.info(f"   Error output: {e.stderr}")
        
        # Install torchcodec if not present (for newer torchaudio)
        logger.info("📦 Installing torchcodec...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "torchcodec"
            ], check=True, capture_output=True, text=True)
            logger.info("✅ Successfully installed torchcodec")
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️ Failed to install torchcodec: {e}")
            logger.info(f"   Error output: {e.stderr}")
        
        logger.info("🎉 TTS compatibility fixes completed!")
        logger.info("💡 You may need to restart your Python environment for changes to take effect.")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to fix TTS issues: {e}")
        return False

if __name__ == "__main__":
    success = fix_tts_issues()
    if success:
        logger.info("✅ All TTS compatibility fixes applied successfully!")
        sys.exit(0)
    else:
        logger.error("💥 TTS compatibility fix failed!")
        sys.exit(1)
