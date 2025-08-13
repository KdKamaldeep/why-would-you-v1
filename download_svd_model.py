#!/usr/bin/env python3
"""
Simple script to download SVD model
"""

import sys
from pathlib import Path

def download_svd_model():
    """Download SVD model using huggingface_hub"""
    try:
        from huggingface_hub import snapshot_download
        print("✅ huggingface_hub is available")
    except ImportError:
        print("❌ huggingface_hub not installed")
        print("💡 Install with: pip install huggingface_hub")
        return False
    
    target_dir = Path("models/svd/stable-video-diffusion-img2vid")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Target directory: {target_dir}")
    print("⬇️ Downloading stabilityai/stable-video-diffusion-img2vid...")
    
    try:
        snapshot_download(
            repo_id="stabilityai/stable-video-diffusion-img2vid",
            repo_type="model",
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            ignore_patterns=["*.md", "*.png", "*.jpg", "*.jpeg"],
            resume_download=True,
        )
        print(f"✅ SVD model downloaded to {target_dir}")
        
        # Check for key files
        key_files = ["config.json", "model.safetensors", "pytorch_model.bin"]
        for file in key_files:
            file_path = target_dir / file
            if file_path.exists():
                print(f"✅ Found: {file}")
            else:
                print(f"⚠️ Missing: {file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to download SVD model: {e}")
        return False

if __name__ == "__main__":
    print("🎬 SVD Model Download")
    print("=" * 40)
    
    success = download_svd_model()
    
    if success:
        print("\n🎉 SVD model download completed!")
        print("💡 You can now use SVD animations:")
        print("   python src/interfaces/simple_cartoon_generator.py --prompt 'Walking cat' --animation-type svd")
    else:
        print("\n❌ SVD model download failed!")
        print("💡 Try installing huggingface_hub:")
        print("   pip install huggingface_hub")
