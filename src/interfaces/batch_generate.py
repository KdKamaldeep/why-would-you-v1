#!/usr/bin/env python3
"""
Batch Cartoon Generator - Generate multiple cartoons from a list
"""

import os
import time
from pathlib import Path
from datetime import datetime

# Sample prompts for batch generation
SAMPLE_PROMPTS = [
    "A baby elephant learns to paint with its trunk",
    "A friendly dragon opens a bakery in a village",
    "A robot discovers the joy of gardening",
    "A magical cat helps children find lost toys",
    "A young wizard's first day at magic school goes wrong"
]

def batch_generate(prompts_file=None, use_samples=False):
    """Generate multiple cartoons from a list of prompts."""
    
    print("🎬 Batch Cartoon Generator")
    print("=" * 50)
    
    # Get prompts
    if use_samples:
        prompts = SAMPLE_PROMPTS
        print(f"📝 Using {len(prompts)} sample prompts")
    elif prompts_file and Path(prompts_file).exists():
        with open(prompts_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"📝 Loaded {len(prompts)} prompts from {prompts_file}")
    else:
        print("❌ No prompts provided!")
        print("Usage:")
        print("  python batch_generate.py --samples    (use built-in samples)")
        print("  python batch_generate.py prompts.txt  (use file with one prompt per line)")
        return
    
    if not prompts:
        print("❌ No valid prompts found!")
        return
    
    try:
        from generate_cartoon_short import CartoonShortsGenerator, VideoConfig
        
        # Create batch output directory
        batch_dir = Path("output") / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        successful = 0
        failed = 0
        
        print(f"📁 Output directory: {batch_dir}")
        print("🚀 Starting batch generation...")
        print("-" * 50)
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n[{i}/{len(prompts)}] Generating: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
            
            try:
                # Create config for this prompt
                config = VideoConfig(
                    prompt=prompt,
                    duration=30,
                    style="cartoon",
                    output_path=str(batch_dir / f"video_{i:02d}")
                )
                
                # Generate
                generator = CartoonShortsGenerator(config)
                output_path = generator.generate()
                
                print(f"✅ Success: {output_path}")
                successful += 1
                
                # Brief pause between generations
                time.sleep(2)
                
            except Exception as e:
                print(f"❌ Failed: {e}")
                failed += 1
                continue
        
        print("\n" + "=" * 50)
        print("🎊 Batch Generation Complete!")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"📁 All videos saved in: {batch_dir}")
        
    except ImportError:
        print("❌ Missing dependencies. Run: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Batch generation error: {e}")

def main():
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--samples":
            batch_generate(use_samples=True)
        else:
            batch_generate(prompts_file=sys.argv[1])
    else:
        print("🎬 Batch Cartoon Generator")
        print("\nUsage:")
        print("  python batch_generate.py --samples")
        print("  python batch_generate.py prompts.txt")
        print("\nCreate prompts.txt with one story idea per line:")
        print("  A cat becomes a superhero")
        print("  A tree that grows candy")
        print("  etc...")

if __name__ == "__main__":
    main()
