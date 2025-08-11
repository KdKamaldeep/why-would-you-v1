#!/usr/bin/env python3
"""
Test script to verify GPT-2 enhancement of visual_prompt from script with 77-token limit
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.prompt_enhancer import PromptEnhancer
from core.generate_cartoon_short import VideoConfig, CartoonShortsGenerator

def test_visual_prompt_enhancement():
    """Test that GPT-2 enhancement specifically targets visual_prompt from script."""
    
    print("🎯 Testing Visual Prompt Enhancement from Script")
    print("=" * 60)
    
    # Initialize the prompt enhancer
    enhancer = PromptEnhancer()
    
    if not enhancer.is_available():
        print("❌ GPT-2 model not available. Please install transformers:")
        print("   pip install transformers")
        return
    
    print("✅ GPT-2 model loaded successfully!")
    print()
    
    # Test visual prompts from typical script scenes
    test_scenes = [
        {
            'visual_prompt': 'two children looking at an old map by warm lantern light',
            'description': 'Scene description here'
        },
        {
            'visual_prompt': 'a cozy village festival with colorful decorations and happy people',
            'description': 'Another scene description'
        },
        {
            'visual_prompt': 'a magical cat discovers a hidden garden with glowing flowers',
            'description': 'Magical scene description'
        },
        {
            'visual_prompt': 'a wise old owl teaching young animals in a forest classroom',
            'description': 'Educational scene description'
        }
    ]
    
    print("🎨 Testing visual_prompt enhancement with 77-token limit:")
    print("-" * 50)
    
    for i, scene in enumerate(test_scenes, 1):
        visual_prompt = scene['visual_prompt']
        print(f"\n{i}. Original visual_prompt: {visual_prompt}")
        
        # Test enhancement with 77-token limit
        enhanced = enhancer.enhance_prompt(
            visual_prompt, 
            enhancement_type="cartoon",
            max_tokens=77
        )
        
        # Count tokens
        token_count = len(enhancer.tokenizer.encode(enhanced))
        
        print(f"   Enhanced: {enhanced}")
        print(f"   Token count: {token_count}/77")
        print(f"   {'✅' if token_count <= 77 else '❌'} Within token limit")
        print()

def test_cartoon_generator_integration():
    """Test the integration with CartoonShortsGenerator."""
    
    print("\n🎬 Testing CartoonShortsGenerator Integration")
    print("=" * 60)
    
    # Create a test configuration
    config = VideoConfig(
        prompt="A magical cat discovers a hidden garden",
        duration=30,
        output_path="test_output",
        enable_prompt_enhancement=True
    )
    
    print(f"Configuration:")
    print(f"  Prompt: {config.prompt}")
    print(f"  Duration: {config.duration}s")
    print(f"  Output: {config.output_path}")
    print(f"  Prompt Enhancement: {'Enabled' if config.enable_prompt_enhancement else 'Disabled'}")
    print()
    
    try:
        # Initialize the generator
        generator = CartoonShortsGenerator(config)
        
        print("✅ Generator initialized successfully!")
        
        # Test the _compose_image_prompt method directly
        test_scene = {
            'visual_prompt': 'two children looking at an old map by warm lantern light',
            'description': 'Scene description here'
        }
        
        print("\n🎯 Testing _compose_image_prompt method:")
        print(f"Input scene: {test_scene}")
        
        composed_prompt = generator._compose_image_prompt(test_scene)
        print(f"Composed prompt: {composed_prompt}")
        
        # Check token count
        if hasattr(generator, 'image_generator') and generator.image_generator.prompt_enhancer:
            token_count = len(generator.image_generator.prompt_enhancer.tokenizer.encode(composed_prompt))
            print(f"Token count: {token_count}/77")
            print(f"{'✅' if token_count <= 77 else '❌'} Within token limit")
        
        print("\n🚀 Ready to generate cartoon with enhanced visual_prompt!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def test_token_limiting():
    """Test the token limiting functionality."""
    
    print("\n🔢 Testing Token Limiting")
    print("=" * 60)
    
    enhancer = PromptEnhancer()
    
    if not enhancer.is_available():
        print("❌ GPT-2 model not available")
        return
    
    # Test with a very long prompt
    long_prompt = "a very detailed scene with many elements including a magical forest with glowing mushrooms, colorful butterflies, ancient trees with faces, a crystal clear stream, rainbow flowers, fairy lights, a wooden bridge, stone statues, and a mysterious castle in the background with towers and flags"
    
    print(f"Original long prompt: {long_prompt}")
    original_tokens = len(enhancer.tokenizer.encode(long_prompt))
    print(f"Original token count: {original_tokens}")
    
    # Test different token limits
    for max_tokens in [50, 77, 100]:
        limited_prompt = enhancer._limit_tokens(long_prompt, max_tokens)
        limited_tokens = len(enhancer.tokenizer.encode(limited_prompt))
        
        print(f"\nMax tokens: {max_tokens}")
        print(f"Limited prompt: {limited_prompt}")
        print(f"Limited token count: {limited_tokens}")
        print(f"{'✅' if limited_tokens <= max_tokens else '❌'} Within limit")

if __name__ == "__main__":
    test_visual_prompt_enhancement()
    test_cartoon_generator_integration()
    test_token_limiting()
