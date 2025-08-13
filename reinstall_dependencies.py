#!/usr/bin/env python3
"""
Script to reinstall dependencies with compatible versions.
"""

import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def reinstall_dependencies():
    """Reinstall dependencies with compatible versions."""
    try:
        logger.info("🔄 Reinstalling dependencies with compatible versions...")
        
        # First, uninstall problematic packages
        packages_to_remove = [
            "transformers",
            "TTS",
            "diffusers"
        ]
        
        for package in packages_to_remove:
            logger.info(f"🗑️ Uninstalling {package}...")
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "uninstall", "-y", package
                ], check=True, capture_output=True, text=True)
                logger.info(f"✅ Successfully uninstalled {package}")
            except subprocess.CalledProcessError as e:
                logger.info(f"ℹ️ {package} not installed or already removed")
        
        # Install from requirements.txt
        logger.info("📦 Installing from requirements.txt...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ], check=True, capture_output=True, text=True)
            logger.info("✅ Successfully installed all dependencies")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install dependencies: {e}")
            logger.info(f"   Error output: {e.stderr}")
            return False
        
        logger.info("🎉 Dependency reinstallation completed!")
        logger.info("💡 You may need to restart your Python environment for changes to take effect.")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to reinstall dependencies: {e}")
        return False

if __name__ == "__main__":
    success = reinstall_dependencies()
    if success:
        logger.info("✅ All dependencies reinstalled successfully!")
        sys.exit(0)
    else:
        logger.error("💥 Dependency reinstallation failed!")
        sys.exit(1)
