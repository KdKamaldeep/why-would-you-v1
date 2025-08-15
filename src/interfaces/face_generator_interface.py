#!/usr/bin/env python3
"""
Face Generator Interface - User-friendly interface for face-based image generation
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# Add the parent directory to the path to import core modules
sys.path.append(str(Path(__file__).parent.parent))

from core.face_image_generator import FaceImageGenerator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceGeneratorInterface:
    """User interface for face-based image generation."""
    
    def __init__(self):
        self.generator = None
        self.output_dir = Path("output/face_generated")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def initialize_generator(self, 
                           model_path: Optional[str] = None,
                           controlnet_path: Optional[str] = None,
                           ip_adapter_path: Optional[str] = None):
        """Initialize the face image generator."""
        logger.info("🚀 Initializing Face Image Generator...")
        
        self.generator = FaceImageGenerator(
            model_path=model_path,
            controlnet_path=controlnet_path,
            ip_adapter_path=ip_adapter_path
        )
        
        if self.generator.pipe is None:
            logger.error("❌ Failed to initialize diffusion pipeline")
            return False
        
        logger.info("✅ Face Image Generator initialized successfully!")
        return True
    
    def generate_single_image(self, 
                            prompt: str,
                            face_image_path: str,
                            negative_prompt: str = "",
                            output_name: Optional[str] = None) -> bool:
        """Generate a single image with face from reference."""
        if self.generator is None:
            logger.error("❌ Generator not initialized")
            return False
        
        if not Path(face_image_path).exists():
            logger.error(f"❌ Face image not found: {face_image_path}")
            return False
        
        # Generate output filename
        if output_name is None:
            output_name = f"face_generated_{Path(face_image_path).stem}.png"
        
        output_path = self.output_dir / output_name
        
        logger.info(f"🎨 Generating image with face from: {face_image_path}")
        logger.info(f"📝 Prompt: {prompt}")
        
        result = self.generator.generate_with_face(
            prompt=prompt,
            face_image_path=face_image_path,
            negative_prompt=negative_prompt,
            output_path=str(output_path)
        )
        
        if result:
            logger.info(f"✅ Image generated successfully: {output_path}")
            return True
        else:
            logger.error("❌ Failed to generate image")
            return False
    
    def generate_video(self,
                      prompt: str,
                      face_image_path: str,
                      negative_prompt: str = "",
                      num_frames: int = 30,
                      output_name: Optional[str] = None) -> bool:
        """Generate a video with face swapping."""
        if self.generator is None:
            logger.error("❌ Generator not initialized")
            return False
        
        if not Path(face_image_path).exists():
            logger.error(f"❌ Face image not found: {face_image_path}")
            return False
        
        # Generate output filename
        if output_name is None:
            output_name = f"face_video_{Path(face_image_path).stem}.mp4"
        
        output_path = self.output_dir / output_name
        
        logger.info(f"🎬 Generating video with face from: {face_image_path}")
        logger.info(f"📝 Prompt: {prompt}")
        logger.info(f"🎞️ Frames: {num_frames}")
        
        result = self.generator.generate_face_swap_video(
            prompt=prompt,
            face_image_path=face_image_path,
            background_prompt=negative_prompt,
            num_frames=num_frames,
            output_path=str(output_path)
        )
        
        if result:
            logger.info(f"✅ Video generated successfully: {output_path}")
            return True
        else:
            logger.error("❌ Failed to generate video")
            return False
    
    def interactive_mode(self):
        """Run in interactive mode."""
        print("🎭 Face-Based Image Generator")
        print("=" * 40)
        
        # Initialize generator
        if not self.initialize_generator():
            print("❌ Failed to initialize generator. Please check your setup.")
            return
        
        while True:
            print("\nOptions:")
            print("1. Generate single image")
            print("2. Generate video")
            print("3. Exit")
            
            choice = input("\nSelect option (1-3): ").strip()
            
            if choice == "1":
                self._interactive_single_image()
            elif choice == "2":
                self._interactive_video()
            elif choice == "3":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid option. Please try again.")
    
    def _interactive_single_image(self):
        """Interactive single image generation."""
        print("\n🎨 Single Image Generation")
        print("-" * 30)
        
        # Get inputs
        face_path = input("Enter path to face reference image: ").strip()
        if not face_path:
            print("❌ Face image path is required")
            return
        
        prompt = input("Enter generation prompt: ").strip()
        if not prompt:
            print("❌ Prompt is required")
            return
        
        negative_prompt = input("Enter negative prompt (optional): ").strip()
        output_name = input("Enter output filename (optional): ").strip()
        
        # Generate image
        success = self.generate_single_image(
            prompt=prompt,
            face_image_path=face_path,
            negative_prompt=negative_prompt,
            output_name=output_name if output_name else None
        )
        
        if success:
            print("✅ Image generated successfully!")
        else:
            print("❌ Image generation failed!")
    
    def _interactive_video(self):
        """Interactive video generation."""
        print("\n🎬 Video Generation")
        print("-" * 20)
        
        # Get inputs
        face_path = input("Enter path to face reference image: ").strip()
        if not face_path:
            print("❌ Face image path is required")
            return
        
        prompt = input("Enter generation prompt: ").strip()
        if not prompt:
            print("❌ Prompt is required")
            return
        
        negative_prompt = input("Enter negative prompt (optional): ").strip()
        
        try:
            num_frames = int(input("Enter number of frames (default 30): ").strip() or "30")
        except ValueError:
            num_frames = 30
        
        output_name = input("Enter output filename (optional): ").strip()
        
        # Generate video
        success = self.generate_video(
            prompt=prompt,
            face_image_path=face_path,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            output_name=output_name if output_name else None
        )
        
        if success:
            print("✅ Video generated successfully!")
        else:
            print("❌ Video generation failed!")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Face-Based Image Generator")
    parser.add_argument("--mode", choices=["single", "video", "interactive"], 
                       default="interactive", help="Generation mode")
    parser.add_argument("--prompt", type=str, help="Generation prompt")
    parser.add_argument("--face-image", type=str, help="Path to face reference image")
    parser.add_argument("--negative-prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--output", type=str, help="Output filename")
    parser.add_argument("--frames", type=int, default=30, help="Number of frames for video")
    parser.add_argument("--model-path", type=str, help="Path to diffusion model")
    parser.add_argument("--controlnet-path", type=str, help="Path to ControlNet model")
    parser.add_argument("--ip-adapter-path", type=str, help="Path to IP-Adapter model")
    
    args = parser.parse_args()
    
    # Create interface
    interface = FaceGeneratorInterface()
    
    # Initialize generator
    if not interface.initialize_generator(
        model_path=args.model_path,
        controlnet_path=args.controlnet_path,
        ip_adapter_path=args.ip_adapter_path
    ):
        sys.exit(1)
    
    # Run based on mode
    if args.mode == "interactive":
        interface.interactive_mode()
    elif args.mode == "single":
        if not args.prompt or not args.face_image:
            print("❌ --prompt and --face-image are required for single mode")
            sys.exit(1)
        
        success = interface.generate_single_image(
            prompt=args.prompt,
            face_image_path=args.face_image,
            negative_prompt=args.negative_prompt,
            output_name=args.output
        )
        sys.exit(0 if success else 1)
    
    elif args.mode == "video":
        if not args.prompt or not args.face_image:
            print("❌ --prompt and --face-image are required for video mode")
            sys.exit(1)
        
        success = interface.generate_video(
            prompt=args.prompt,
            face_image_path=args.face_image,
            negative_prompt=args.negative_prompt,
            num_frames=args.frames,
            output_name=args.output
        )
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
