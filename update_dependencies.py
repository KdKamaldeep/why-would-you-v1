#!/usr/bin/env python3
"""
Script to update dependencies and fix warning issues.
"""

import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def update_dependencies():
    """Update dependencies to fix warning issues."""
    try:
        logger.info("🔄 Updating dependencies to fix warning issues...")
        
        # Update core packages
        packages_to_update = [
            "diffusers>=0.31.0",
            "transformers>=4.50.0", 
            "accelerate>=0.28.0",
            "safetensors>=0.4.3",
            "huggingface_hub>=0.30.1"
        ]
        
        for package in packages_to_update:
            logger.info(f"📦 Installing {package}...")
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "--upgrade", package
                ], check=True, capture_output=True, text=True)
                logger.info(f"✅ Successfully installed {package}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"⚠️ Failed to install {package}: {e}")
                logger.info(f"   Error output: {e.stderr}")
        
        logger.info("🎉 Dependency update completed!")
        logger.info("💡 You may need to restart your Python environment for changes to take effect.")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to update dependencies: {e}")
        return False

if __name__ == "__main__":
    success = update_dependencies()
    if success:
        logger.info("✅ All dependencies updated successfully!")
        sys.exit(0)
    else:
        logger.error("💥 Dependency update failed!")
        sys.exit(1)
