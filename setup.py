#!/usr/bin/env python3
"""
Setup script for Cartoon Shorts Generator
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_ffmpeg():
    """Check if FFmpeg is installed."""
    try:
        subprocess.run(['ffmpeg', '-version'], check=True, capture_output=True)
        print("✅ FFmpeg is already installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_ffmpeg():
    """Install FFmpeg based on the operating system."""
    system = platform.system().lower()
    
    if system == "windows":
        print("📋 Installing FFmpeg on Windows...")
        print("Please install FFmpeg manually:")
        print("1. Download from: https://ffmpeg.org/download.html")
        print("2. Extract to C:\\ffmpeg")
        print("3. Add C:\\ffmpeg\\bin to your PATH environment variable")
        return False
    
    elif system == "darwin":  # macOS
        return run_command("brew install ffmpeg", "Installing FFmpeg via Homebrew")
    
    elif system == "linux":
        # Try different package managers
        if run_command("which apt-get", "Checking for apt-get"):
            return run_command(" apt update &&  apt install -y ffmpeg", "Installing FFmpeg via apt")
        elif run_command("which yum", "Checking for yum"):
            return run_command(" yum install -y ffmpeg", "Installing FFmpeg via yum")
        elif run_command("which dnf", "Checking for dnf"):
            return run_command(" dnf install -y ffmpeg", "Installing FFmpeg via dnf")
        else:
            print("❌ Could not determine package manager. Please install FFmpeg manually.")
            return False
    
    else:
        print(f"❌ Unsupported operating system: {system}")
        return False

def install_python_dependencies():
    """Install Python dependencies."""
    print("📋 Installing Python dependencies...")
    print("Note: dlib and face_alignment are commented out due to CMake requirements.")
    print("If you need face processing, install CMake first, then uncomment them in requirements.txt")
    return run_command("pip install -r requirements.txt", "Installing Python dependencies")

def install_cmake():
    """Install CMake for dlib compilation."""
    system = platform.system().lower()
    
    if system == "windows":
        print("📋 Installing CMake on Windows...")
        print("Please install CMake manually:")
        print("1. Download from: https://cmake.org/download/")
        print("2. Run the installer and make sure to add CMake to PATH")
        print("3. Restart your terminal after installation")
        return False
    
    elif system == "darwin":  # macOS
        return run_command("brew install cmake", "Installing CMake via Homebrew")
    
    elif system == "linux":
        # Try different package managers
        if run_command("which apt-get", "Checking for apt-get"):
            return run_command(" apt update &&  apt install -y cmake", "Installing CMake via apt")
        elif run_command("which yum", "Checking for yum"):
            return run_command(" yum install -y cmake", "Installing CMake via yum")
        elif run_command("which dnf", "Checking for dnf"):
            return run_command(" dnf install -y cmake", "Installing CMake via dnf")
        else:
            print("❌ Could not determine package manager. Please install CMake manually.")
            return False
    
    else:
        print(f"❌ Unsupported operating system: {system}")
        return False



def download_models():
    """Download AI models."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print("📋 Downloading AI models...")
    print("Please download the following models manually:")
    print("1. ToonYou model: https://huggingface.co/ckpt/ToonYou/resolve/main/ToonYou_beta6.safetensors")
    print("2. MeinaMix model: https://huggingface.co/Meina/MeinaMix/resolve/main/MeinaMix.safetensors")
    print("3. AnimateDiff models: https://huggingface.co/guoyww/animatediff")
    print("4. Save them in the 'models/' directory")
    return True

def create_directories():
    """Create necessary directories."""
    directories = ["output", "music", "models", "loras"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("✅ Created necessary directories")
    return True

def setup_env_file():
    """Set up environment file."""
    if Path(".env").exists():
        print("✅ .env file already exists")
        return True
    
    if Path("config.env").exists():
        print("📋 Copying config.env to .env...")
        import shutil
        shutil.copy("config.env", ".env")
        print("✅ Created .env file from config.env")
        print("⚠️  Please edit .env and add your API keys")
        return True
    
    print("❌ No config.env file found. Please create .env file manually.")
    return False

def main():
    """Main setup function."""
    print("🎬 Cartoon Shorts Generator Setup")
    print("=" * 40)
    
    success = True
    
    # Check and install FFmpeg
    if not check_ffmpeg():
        success &= install_ffmpeg()
    
    # Install Python dependencies
    success &= install_python_dependencies()
    
    # Install CMake (optional, for dlib)
    print("\n📋 CMake Installation (Optional)")
    print("CMake is required if you want to use dlib for face processing.")
    print("The current setup uses mediapipe as an alternative.")
    cmake_choice = input("Do you want to install CMake for dlib support? (y/N): ").lower().strip()
    if cmake_choice in ['y', 'yes']:
        success &= install_cmake()
        print("After installing CMake, you can uncomment dlib and face_alignment in requirements.txt")
        print("Then run: pip install dlib face_alignment")
    

    
    # Download models
    success &= download_models()
    
    # Create directories
    success &= create_directories()
    
    # Set up environment file
    success &= setup_env_file()
    
    print("\n" + "=" * 40)
    if success:
        print("✅ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Edit .env file and add your API keys")
        print("2. Download the required AI models")
        print("3. Add background music to the 'music/' directory")
        print("5. Run: python generate_cartoon_short.py --prompt 'Your story prompt'")
    else:
        print("⚠️  Setup completed with some issues.")
        print("Please check the output above and complete manual steps.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
