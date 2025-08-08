#!/usr/bin/env python3
"""
Install AI Dependencies - Install required packages for image generation
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a Python package using pip."""
    print(f"📦 Installing {package}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def main():
    """Install all required AI dependencies."""
    print("🤖 Installing AI Dependencies for Cartoon Generation")
    print("=" * 60)
    
    # Required packages for AI image generation
    packages = [
        "diffusers",
        "transformers", 
        "accelerate",
        "safetensors",
        "xformers",  # For memory efficiency (optional but recommended)
    ]
    
    successful = 0
    failed = 0
    
    for package in packages:
        if install_package(package):
            successful += 1
        else:
            failed += 1
    
    print("\n" + "=" * 60)
    print("📊 Installation Summary:")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All AI dependencies installed successfully!")
        print("You can now generate real AI cartoon images instead of placeholders.")
        print("\nNext steps:")
        print("1. Make sure you've downloaded the models: download_models.bat")
        print("2. Run your cartoon generator: python simple_cartoon_generator.py")
    else:
        print("\n⚠️ Some packages failed to install.")
        print("The system will still work but may use placeholder images.")
        
        # Try alternative installation for failed packages
        print("\nTrying alternative installation methods...")
        if "xformers" in [packages[i] for i, pkg in enumerate(packages) if not install_package(pkg)]:
            print("Note: xformers is optional and may require specific GPU drivers.")

if __name__ == "__main__":
    main()
