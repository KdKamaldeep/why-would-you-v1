#!/usr/bin/env python3
"""
Test Script for Professional Prompt Optimization System
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.prompt_enhancer import ProfessionalPromptEnhancer
from src.core.prompt_validator import PromptValidator
from src.interfaces.prompt_optimizer import PromptOptimizer

def test_prompt_enhancement():
    """Test the professional prompt enhancer."""
    print("🧪 Testing Professional Prompt Enhancement")
    print("=" * 60)
    
    enhancer = ProfessionalPromptEnhancer()
    
    # Test prompts
    test_prompts = [
        "a cat",
        "person in room",
        "Sardar Patel sitting at wooden desk, maps of India spread out, white dhoti kurta, round spectacles, 1940s study room, warm sepia lighting, cinematic shadows, historical drama style",
        "cute cartoon cat, sitting on wooden table, wearing red hat, bright eyes, soft lighting, clean lines, vibrant colors, professional illustration"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📝 Test {i}: {prompt}")
        print("-" * 40)
        
        try:
            analysis = enhancer.analyze_prompt(prompt, style="cartoon")
            
            print(f"🎯 Clarity Score: {analysis.clarity_score:.2f}")
            print(f"🏗️ Structure Score: {analysis.structure_score:.2f}")
            print(f"🔍 Specificity Score: {analysis.specificity_score:.2f}")
            
            if analysis.issues:
                print(f"⚠️ Issues: {', '.join(analysis.issues)}")
            
            if analysis.suggestions:
                print(f"💡 Suggestions: {', '.join(analysis.suggestions)}")
            
            print(f"🚀 Enhanced: {analysis.enhanced_prompt}")
            print(f"🚫 Negative: {analysis.optimized_negative_prompt}")
            
        except Exception as e:
            print(f"❌ Error: {e}")

def test_prompt_optimizer():
    """Test the prompt optimizer interface."""
    print("\n🧪 Testing Prompt Optimizer Interface")
    print("=" * 60)
    
    optimizer = PromptOptimizer()
    
    # Test optimization
    test_prompt = "a cat playing in garden"
    print(f"📝 Original: {test_prompt}")
    
    try:
        optimized = optimizer.optimize_prompt(test_prompt, style="cartoon")
        print(f"🚀 Optimized: {optimized}")
        
        # Show examples
        print("\n📚 Examples for cartoon style:")
        optimizer.show_examples("cartoon")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def test_style_comparison():
    """Test optimization across different styles."""
    print("\n🧪 Testing Style Comparison")
    print("=" * 60)
    
    enhancer = ProfessionalPromptEnhancer()
    test_prompt = "a person with a camera"
    
    styles = ["cartoon", "anime", "realistic"]
    
    for style in styles:
        print(f"\n🎨 Style: {style}")
        print("-" * 30)
        
        try:
            analysis = enhancer.analyze_prompt(test_prompt, style=style)
            print(f"Enhanced: {analysis.enhanced_prompt}")
            print(f"Negative: {analysis.optimized_negative_prompt}")
            
        except Exception as e:
            print(f"❌ Error: {e}")

def test_batch_optimization():
    """Test batch optimization."""
    print("\n🧪 Testing Batch Optimization")
    print("=" * 60)
    
    optimizer = PromptOptimizer()
    
    prompts = [
        "a cat",
        "a dog",
        "a bird",
        "a person reading a book",
        "a car on the street"
    ]
    
    print("📝 Original prompts:")
    for i, prompt in enumerate(prompts, 1):
        print(f"  {i}. {prompt}")
    
    try:
        optimized_prompts = optimizer.batch_optimize(prompts, style="cartoon")
        
        print("\n🚀 Optimized prompts:")
        for i, (original, optimized) in enumerate(zip(prompts, optimized_prompts), 1):
            print(f"  {i}. Original: {original}")
            print(f"     Optimized: {optimized}")
            print()
        
    except Exception as e:
        print(f"❌ Error: {e}")

def test_validation_simulation():
    """Test validation system (simulated)."""
    print("\n🧪 Testing Validation System (Simulated)")
    print("=" * 60)
    
    validator = PromptValidator()
    
    # Simulate validation results
    test_prompts = [
        "a cat",
        "cute cartoon cat, sitting on wooden table, wearing red hat, bright eyes, soft lighting, clean lines, vibrant colors, professional illustration"
    ]
    
    test_negative_prompts = [
        "",
        "blurry, low quality, distorted, deformed, ugly, bad anatomy"
    ]
    
    # Note: This will fail without real images, but shows the interface
    print("📝 Testing validation interface:")
    for i, (prompt, negative) in enumerate(zip(test_prompts, test_negative_prompts), 1):
        print(f"  {i}. Prompt: {prompt}")
        print(f"     Negative: {negative}")
        print(f"     (Validation would analyze generated image quality)")
        print()

def demonstrate_best_practices():
    """Demonstrate best practices."""
    print("\n📚 Best Practices Demonstration")
    print("=" * 60)
    
    examples = {
        "Bad": "a cat",
        "Good": "cute cartoon cat, sitting on wooden table, wearing red hat, bright eyes, soft lighting, clean lines, vibrant colors, professional illustration",
        "Improved": "adorable cartoon cat character, sitting confidently on rustic wooden table, wearing bright red hat, large expressive eyes, warm soft lighting, clean bold lines, vibrant saturated colors, professional digital illustration"
    }
    
    for quality, prompt in examples.items():
        print(f"\n{quality}:")
        print(f"  {prompt}")
        
        # Analyze each
        try:
            enhancer = ProfessionalPromptEnhancer()
            analysis = enhancer.analyze_prompt(prompt, style="cartoon")
            print(f"  Clarity: {analysis.clarity_score:.2f}, Structure: {analysis.structure_score:.2f}, Specificity: {analysis.specificity_score:.2f}")
        except Exception as e:
            print(f"  Analysis error: {e}")

def main():
    """Run all tests."""
    print("🎯 Professional Prompt Optimization System - Test Suite")
    print("=" * 80)
    
    try:
        test_prompt_enhancement()
        test_prompt_optimizer()
        test_style_comparison()
        test_batch_optimization()
        test_validation_simulation()
        demonstrate_best_practices()
        
        print("\n✅ All tests completed!")
        print("\n💡 To use the system:")
        print("   python -m src.interfaces.prompt_optimizer --mode interactive")
        print("   python -m src.interfaces.prompt_optimizer --mode analyze --prompt 'your prompt'")
        print("   python -m src.interfaces.prompt_optimizer --mode optimize --prompt 'your prompt'")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install transformers torch pillow numpy scipy")

if __name__ == "__main__":
    main()
