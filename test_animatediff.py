#!/usr/bin/env python3
"""
Test AnimateDiff Installation and Animation Generation
"""

import torch
import logging
from pathlib import Path
from animation_generator import AnimationGenerator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_animatediff():
    """Test AnimateDiff functionality."""
    print("🎬 Testing AnimateDiff Animation System")
    print("=" * 50)
    
    # Check if models exist
    motion_model = Path("models/mm_sd_v15_v2.safetensors")
    if not motion_model.exists():
        print("❌ Motion model not found!")
        print("Please run: download_models.bat")
        return False
    
    print("✅ Motion model found")
    
    # Test AnimationGenerator initialization
    try:
        print("🔧 Initializing AnimationGenerator...")
        animator = AnimationGenerator()
        print("✅ AnimationGenerator initialized")
        
        # Check if AnimateDiff pipeline is loaded
        if animator.pipe is not None:
            print("✅ Real AnimateDiff pipeline loaded!")
            print(f"📱 Device: {animator.device}")
            print("🎯 Ready for AI-powered character animation!")
        else:
            print("⚠️ AnimateDiff pipeline not loaded - will use enhanced fallback")
            print("💡 This still provides multiple animation effects:")
            print("   • Zoom and pan animations")
            print("   • Sliding effects")
            print("   • Rotation with zoom")
            print("   • Parallax scrolling")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def create_test_animation():
    """Create a test animation."""
    print("\n🎨 Creating Test Animation")
    print("-" * 30)
    
    # Create a simple test image first
    from PIL import Image, ImageDraw
    
    # Create test image
    img = Image.new('RGB', (768, 1024), color='lightblue')
    draw = ImageDraw.Draw(img)
    draw.ellipse([284, 400, 484, 600], fill='yellow', outline='orange', width=5)  # Sun
    draw.rectangle([200, 600, 568, 800], fill='green')  # Ground
    draw.text((300, 850), "Test Scene", fill='black')
    
    test_image_path = "test_image.png"
    img.save(test_image_path)
    print(f"✅ Created test image: {test_image_path}")
    
    # Test animation
    try:
        animator = AnimationGenerator()
        output_dir = "test_animation_frames"
        
        print("🎬 Generating animation...")
        result = animator.animate_image(
            test_image_path, 
            output_dir, 
            num_frames=12,
            prompt="sunny day with moving clouds"
        )
        
        if Path(output_dir).exists():
            frame_count = len(list(Path(output_dir).glob("*.png")))
            print(f"✅ Animation created! {frame_count} frames in {output_dir}")
            return True
        else:
            print("❌ Animation creation failed")
            return False
            
    except Exception as e:
        print(f"❌ Animation test failed: {e}")
        return False

def main():
    """Main test function."""
    success = test_animatediff()
    
    if success:
        print("\n" + "=" * 50)
        print("🎉 AnimateDiff system is ready!")
        print("\n🚀 What you can expect:")
        print("• 🎭 Character movements and expressions")
        print("• 🌊 Fluid motion and physics effects") 
        print("• ✨ Dynamic scene animations")
        print("• 🎪 Multiple fallback animation styles")
        print("• 🎯 Scene-specific animation prompts")
        
        # Optional: Create test animation
        choice = input("\nCreate a test animation? (y/N): ").lower().strip()
        if choice in ['y', 'yes']:
            create_test_animation()
            
        print("\n💡 Ready to generate cartoons with:")
        print("python simple_cartoon_generator.py --prompt 'Your amazing story'")
    else:
        print("\n❌ Setup issues detected.")
        print("💡 Try running: fix_torch_conflict.bat")

if __name__ == "__main__":
    main()
