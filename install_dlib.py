#!/usr/bin/env python3
"""
Helper script to install dlib and face_alignment with CMake support
"""

import os
import sys
import subprocess
import platform

def check_cmake():
    """Check if CMake is installed."""
    try:
        result = subprocess.run(['cmake', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ CMake is installed: {result.stdout.strip()}")
            return True
        else:
            return False
    except FileNotFoundError:
        return False

def install_cmake():
    """Install CMake based on the operating system."""
    system = platform.system().lower()
    
    if system == "windows":
        print("📋 Installing CMake on Windows...")
        print("Please install CMake manually:")
        print("1. Download from: https://cmake.org/download/")
        print("2. Run the installer and make sure to add CMake to PATH")
        print("3. Restart your terminal after installation")
        print("4. Run this script again")
        return False
    
    elif system == "darwin":  # macOS
        print("📋 Installing CMake via Homebrew...")
        try:
            subprocess.run(['brew', 'install', 'cmake'], check=True)
            print("✅ CMake installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install CMake via Homebrew")
            return False
    
    elif system == "linux":
        print("📋 Installing CMake via package manager...")
        try:
            # Try apt-get first
            subprocess.run(['sudo', 'apt', 'update'], check=True)
            subprocess.run(['sudo', 'apt', 'install', '-y', 'cmake'], check=True)
            print("✅ CMake installed successfully")
            return True
        except subprocess.CalledProcessError:
            try:
                # Try yum
                subprocess.run(['sudo', 'yum', 'install', '-y', 'cmake'], check=True)
                print("✅ CMake installed successfully")
                return True
            except subprocess.CalledProcessError:
                try:
                    # Try dnf
                    subprocess.run(['sudo', 'dnf', 'install', '-y', 'cmake'], check=True)
                    print("✅ CMake installed successfully")
                    return True
                except subprocess.CalledProcessError:
                    print("❌ Failed to install CMake. Please install manually.")
                    return False
    
    else:
        print(f"❌ Unsupported operating system: {system}")
        return False

def install_dlib():
    """Install dlib."""
    print("📋 Installing dlib...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'dlib'], check=True)
        print("✅ dlib installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dlib: {e}")
        return False

def install_face_alignment():
    """Install face_alignment."""
    print("📋 Installing face_alignment...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'face_alignment'], check=True)
        print("✅ face_alignment installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install face_alignment: {e}")
        return False

def main():
    """Main installation function."""
    print("🔧 dlib and face_alignment Installation Helper")
    print("=" * 50)
    
    # Check if CMake is installed
    if not check_cmake():
        print("❌ CMake is not installed. Installing CMake first...")
        if not install_cmake():
            print("❌ Failed to install CMake. Please install it manually and try again.")
            return 1
        
        # Verify CMake installation
        if not check_cmake():
            print("❌ CMake installation verification failed. Please restart your terminal and try again.")
            return 1
    
    # Install dlib
    if not install_dlib():
        print("❌ Failed to install dlib. Check the error messages above.")
        return 1
    
    # Install face_alignment
    if not install_face_alignment():
        print("❌ Failed to install face_alignment. Check the error messages above.")
        return 1
    
    print("\n🎉 Installation completed successfully!")
    print("You can now uncomment dlib and face_alignment in requirements.txt")
    print("and use them in your cartoon shorts generator.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
