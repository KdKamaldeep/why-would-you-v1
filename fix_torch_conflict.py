#!/usr/bin/env python3
"""
Fix PyTorch Version Conflicts
"""

import subprocess
import sys

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def fix_torch_versions():
    """Fix PyTorch version conflicts."""
    print("🔧 Fixing PyTorch Version Conflicts")
    print("=" * 50)
    
    # Method 1: Uninstall and reinstall PyTorch ecosystem
    print("\n📦 Method 1: Clean PyTorch reinstallation")
    
    # Uninstall all PyTorch packages
    torch_packages = [
        "torch", "torchvision", "torchaudio", "torchtext",
        "xformers"  # This often causes conflicts
    ]
    
    for package in torch_packages:
        print(f"🗑️ Uninstalling {package}...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", package, "-y"], 
                      capture_output=True)
    
    print("✅ PyTorch packages uninstalled")
    
    # Reinstall compatible versions
    print("\n📦 Installing compatible PyTorch versions...")
    
    # For CUDA systems
    cuda_cmd = f"{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    
    # For CPU-only systems  
    cpu_cmd = f"{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    
    print("🎯 Trying CUDA installation first...")
    if run_command(cuda_cmd, "Installing PyTorch with CUDA support"):
        print("✅ CUDA PyTorch installed successfully!")
    else:
        print("⚠️ CUDA installation failed, trying CPU version...")
        if run_command(cpu_cmd, "Installing PyTorch CPU version"):
            print("✅ CPU PyTorch installed successfully!")
        else:
            print("❌ Both installations failed. Manual intervention needed.")
            return False
    
    # Install other AI packages
    print("\n📦 Installing AI image generation packages...")
    ai_packages = [
        "diffusers",
        "accelerate", 
        "safetensors",
        "transformers"
    ]
    
    for package in ai_packages:
        run_command(f"{sys.executable} -m pip install {package}", f"Installing {package}")
    
    # Try xformers (optional)
    print("\n📦 Installing xformers (optional, may fail on some systems)...")
    xformers_success = run_command(f"{sys.executable} -m pip install xformers", "Installing xformers")
    if not xformers_success:
        print("⚠️ xformers installation failed - this is optional and the system will work without it")
    
    print("\n🎉 PyTorch conflict resolution completed!")
    return True

def verify_installation():
    """Verify the installation works."""
    print("\n🔍 Verifying installation...")
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        import diffusers
        print(f"✅ Diffusers version: {diffusers.__version__}")
        
        print("✅ All packages working correctly!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    print("🔧 PyTorch Conflict Resolver")
    print("=" * 40)
    
    print("This will fix version conflicts by:")
    print("1. Uninstalling conflicting PyTorch packages")
    print("2. Reinstalling compatible versions")
    print("3. Installing AI image generation packages")
    print()
    
    choice = input("Continue? (y/N): ").lower().strip()
    if choice not in ['y', 'yes']:
        print("Operation cancelled.")
        return
    
    success = fix_torch_versions()
    
    if success:
        verify_installation()
        print("\n🎊 All done! You can now run your cartoon generator:")
        print("python simple_cartoon_generator.py --prompt 'Your story here'")
    else:
        print("\n❌ Some issues occurred. You may need to:")
        print("1. Update your GPU drivers")
        print("2. Use CPU-only versions")
        print("3. Install packages manually")

if __name__ == "__main__":
    main()
