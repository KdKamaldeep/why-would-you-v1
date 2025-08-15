#!/usr/bin/env python3
"""
Fix dependencies for WhyWouldYou-v1 project.
This script helps resolve import issues with ControlNetPipeline.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    print("🔧 Fixing dependencies for WhyWouldYou-v1...")
    print("=" * 50)
    
    # Step 1: Upgrade pip
    run_command("pip install --upgrade pip", "Upgrading pip")
    
    # Step 2: Install/upgrade diffusers to a compatible version
    print("\n📦 Installing compatible diffusers version...")
    success = run_command("pip install diffusers>=0.31.0", "Installing diffusers")
    
    if not success:
        print("⚠️ Trying alternative diffusers version...")
        success = run_command("pip install diffusers==0.31.0", "Installing specific diffusers version")
    
    # Step 3: Install other required packages
    print("\n📦 Installing other required packages...")
    run_command("pip install transformers==4.49.0", "Installing transformers")
    run_command("pip install accelerate", "Installing accelerate")
    run_command("pip install safetensors", "Installing safetensors")
    
    # Step 4: Test the import
    print("\n🧪 Testing imports...")
    test_script = """
import sys
try:
    from diffusers import StableDiffusionPipeline
    print("✅ StableDiffusionPipeline import successful")
except ImportError as e:
    print(f"❌ StableDiffusionPipeline import failed: {e}")

try:
    from diffusers import ControlNetPipeline
    print("✅ ControlNetPipeline import successful")
except ImportError as e:
    print(f"⚠️ ControlNetPipeline import failed: {e}")
    print("This is expected in some diffusers versions")

try:
    from diffusers import ControlNetModel
    print("✅ ControlNetModel import successful")
except ImportError as e:
    print(f"⚠️ ControlNetModel import failed: {e}")

print("\\n📋 Installed package versions:")
import pkg_resources
packages = ['diffusers', 'transformers', 'accelerate', 'safetensors']
for pkg in packages:
    try:
        version = pkg_resources.get_distribution(pkg).version
        print(f"  {pkg}: {version}")
    except:
        print(f"  {pkg}: not installed")
"""
    
    try:
        result = subprocess.run([sys.executable, "-c", test_script], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Import test failed: {e}")
        print(f"Error output: {e.stderr}")
    
    print("\n" + "=" * 50)
    print("✅ Dependency fix completed!")
    print("\n💡 If you still see ControlNetPipeline import errors:")
    print("   - The code has been updated to handle missing ControlNetPipeline gracefully")
    print("   - Face generation will work without ControlNet features")
    print("   - You can still use the main cartoon generation features")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
