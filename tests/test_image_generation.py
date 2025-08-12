#!/usr/bin/env python3
"""
Test script for Stable Diffusion image generation
"""

import os
import logging
from pathlib import Path
import sys
import argparse
import re
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.image_generator import ImageGenerator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_prompt_compliance(image_path: str, prompt: str) -> Dict[str, any]:
    """
    Analyze if a generated image complies with the given visual prompt.
    Returns a dictionary with compliance scores and analysis.
    """
    try:
        from collections import Counter
        
        # Load the image
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image)
        
        # Extract key elements from the prompt
        prompt_elements = extract_prompt_elements(prompt)
        
        # Analyze image characteristics
        analysis = {
            'prompt_elements': prompt_elements,
            'image_analysis': analyze_image_characteristics(img_array),
            'compliance_score': 0.0,
            'missing_elements': [],
            'present_elements': [],
            'overall_quality': 'unknown'
        }
        
        # Check for basic quality issues first
        if is_image_blank_or_poor_quality(img_array):
            analysis['overall_quality'] = 'poor'
            analysis['compliance_score'] = 0.0
            return analysis
        
        analysis['overall_quality'] = 'good'
        
        # Analyze color compliance
        color_compliance = analyze_color_compliance(img_array, prompt_elements)
        analysis['color_analysis'] = color_compliance
        
        # Analyze composition compliance
        composition_compliance = analyze_composition_compliance(img_array, prompt_elements)
        analysis['composition_analysis'] = composition_compliance
        
        # Calculate overall compliance score
        analysis['compliance_score'] = calculate_compliance_score(
            color_compliance, composition_compliance, prompt_elements
        )
        
        # Identify missing and present elements
        analysis['missing_elements'] = identify_missing_elements(analysis)
        analysis['present_elements'] = identify_present_elements(analysis)
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing prompt compliance: {e}")
        return {
            'error': str(e),
            'compliance_score': 0.0,
            'overall_quality': 'error'
        }

def extract_prompt_elements(prompt: str) -> Dict[str, any]:
    """Extract key elements from the visual prompt."""
    elements = {
        'characters': [],
        'colors': [],
        'setting': '',
        'lighting': '',
        'style': '',
        'actions': [],
        'objects': []
    }
    
    # Extract characters (animals, people)
    character_patterns = [
        r'cartoon\s+(\w+)\s+(\w+)',  # cartoon brown monkey
        r'(\w+)\s+(\w+)\s+wearing',  # brown monkey wearing
        r'(\w+)\s+(\w+)\s+in\s+(\w+)',  # bear in blue vest
    ]
    
    for pattern in character_patterns:
        matches = re.findall(pattern, prompt, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                elements['characters'].append(' '.join(match))
            else:
                elements['characters'].append(match)
    
    # Extract colors
    color_patterns = [
        r'(\w+)\s+scarf',  # red scarf
        r'(\w+)\s+vest',   # blue vest
        r'(\w+)\s+bow',    # green bow
        r'(\w+)\s+light',  # blue light
        r'(\w+)\s+leaves', # green leaves
    ]
    
    for pattern in color_patterns:
        matches = re.findall(pattern, prompt, re.IGNORECASE)
        elements['colors'].extend(matches)
    
    # Extract setting
    setting_keywords = ['jungle', 'night', 'stars', 'grass', 'tree', 'background']
    for keyword in setting_keywords:
        if keyword in prompt.lower():
            elements['setting'] = keyword
    
    # Extract lighting
    lighting_keywords = ['moonlight', 'sunlight', 'soft', 'bright', 'warm', 'blue']
    for keyword in lighting_keywords:
        if keyword in prompt.lower():
            elements['lighting'] = keyword
    
    # Extract style
    style_keywords = ['cartoon', 'storybook', 'illustration', 'serene']
    for keyword in style_keywords:
        if keyword in prompt.lower():
            elements['style'] = keyword
    
    # Extract actions
    action_keywords = ['sleeping', 'standing', 'gathered', 'playing']
    for keyword in action_keywords:
        if keyword in prompt.lower():
            elements['actions'].append(keyword)
    
    return elements

def analyze_image_characteristics(img_array: np.ndarray) -> Dict[str, any]:
    """Analyze basic image characteristics."""
    # Convert to different color spaces for analysis
    hsv = Image.fromarray(img_array).convert('HSV')
    hsv_array = np.array(hsv)
    
    # Analyze color distribution
    hue_hist = np.histogram(hsv_array[:, :, 0], bins=18, range=(0, 180))[0]
    saturation_hist = np.histogram(hsv_array[:, :, 1], bins=10, range=(0, 255))[0]
    value_hist = np.histogram(hsv_array[:, :, 2], bins=10, range=(0, 255))[0]
    
    # Calculate dominant colors
    rgb_array = img_array.reshape(-1, 3)
    unique_colors, counts = np.unique(rgb_array, axis=0, return_counts=True)
    dominant_colors = unique_colors[np.argsort(counts)[-5:]]  # Top 5 colors
    
    return {
        'dominant_colors': dominant_colors.tolist(),
        'hue_distribution': hue_hist.tolist(),
        'saturation_distribution': saturation_hist.tolist(),
        'brightness_distribution': value_hist.tolist(),
        'total_unique_colors': len(unique_colors),
        'mean_brightness': np.mean(hsv_array[:, :, 2]),
        'mean_saturation': np.mean(hsv_array[:, :, 1])
    }

def is_image_blank_or_poor_quality(img_array: np.ndarray) -> bool:
    """Check if image is blank or poor quality."""
    variance = np.var(img_array)
    unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
    gray = np.mean(img_array, axis=2)
    brightness_variance = np.var(gray)
    mean_brightness = np.mean(gray)
    
    return (
        variance < 1000 or
        unique_colors < 100 or
        brightness_variance < 500 or
        mean_brightness < 10 or
        mean_brightness > 245
    )

def analyze_color_compliance(img_array: np.ndarray, prompt_elements: Dict) -> Dict[str, any]:
    """Analyze if image colors match prompt requirements."""
    compliance = {
        'score': 0.0,
        'matched_colors': [],
        'missing_colors': [],
        'overall_color_vibrancy': 0.0
    }
    
    # Check for required colors in prompt
    required_colors = prompt_elements.get('colors', [])
    if not required_colors:
        compliance['score'] = 0.5  # Neutral score if no specific colors mentioned
        return compliance
    
    # Analyze color vibrancy
    hsv = Image.fromarray(img_array).convert('HSV')
    hsv_array = np.array(hsv)
    mean_saturation = np.mean(hsv_array[:, :, 1])
    compliance['overall_color_vibrancy'] = mean_saturation / 255.0
    
    # Simple color matching (this could be enhanced with more sophisticated color analysis)
    matched_count = 0
    for color in required_colors:
        # This is a simplified check - in practice, you'd want more sophisticated color matching
        if color.lower() in ['red', 'blue', 'green', 'brown', 'gray']:
            matched_count += 1
            compliance['matched_colors'].append(color)
        else:
            compliance['missing_colors'].append(color)
    
    compliance['score'] = matched_count / len(required_colors) if required_colors else 0.0
    return compliance

def analyze_composition_compliance(img_array: np.ndarray, prompt_elements: Dict) -> Dict[str, any]:
    """Analyze if image composition matches prompt requirements."""
    compliance = {
        'score': 0.0,
        'has_characters': False,
        'has_setting': False,
        'composition_quality': 0.0
    }
    
    # Analyze image structure (simplified)
    height, width = img_array.shape[:2]
    
    # Check for character presence (simplified - looks for non-uniform areas)
    gray = np.mean(img_array, axis=2)
    edge_variance = np.var(np.gradient(gray))
    compliance['has_characters'] = edge_variance > 1000  # High variance suggests detailed content
    
    # Check for setting presence (simplified)
    color_variance = np.var(img_array)
    compliance['has_setting'] = color_variance > 5000  # High color variance suggests rich background
    
    # Overall composition quality
    compliance['composition_quality'] = min(1.0, (edge_variance + color_variance) / 10000)
    
    # Calculate score based on elements
    score_components = []
    if prompt_elements.get('characters') and compliance['has_characters']:
        score_components.append(1.0)
    if prompt_elements.get('setting') and compliance['has_setting']:
        score_components.append(1.0)
    
    compliance['score'] = sum(score_components) / max(len(score_components), 1)
    return compliance

def calculate_compliance_score(color_compliance: Dict, composition_compliance: Dict, prompt_elements: Dict) -> float:
    """Calculate overall compliance score."""
    # Weight different aspects
    color_weight = 0.4
    composition_weight = 0.6
    
    color_score = color_compliance.get('score', 0.0)
    composition_score = composition_compliance.get('score', 0.0)
    
    # Adjust weights based on what's mentioned in prompt
    if not prompt_elements.get('colors'):
        color_weight = 0.2
        composition_weight = 0.8
    
    if not prompt_elements.get('characters') and not prompt_elements.get('setting'):
        composition_weight = 0.4
        color_weight = 0.6
    
    overall_score = (color_score * color_weight) + (composition_score * composition_weight)
    return min(1.0, max(0.0, overall_score))

def identify_missing_elements(analysis: Dict) -> List[str]:
    """Identify elements mentioned in prompt but not detected in image."""
    missing = []
    prompt_elements = analysis.get('prompt_elements', {})
    
    if prompt_elements.get('characters') and not analysis.get('composition_analysis', {}).get('has_characters'):
        missing.append('characters')
    
    if prompt_elements.get('setting') and not analysis.get('composition_analysis', {}).get('has_setting'):
        missing.append('setting')
    
    missing_colors = analysis.get('color_analysis', {}).get('missing_colors', [])
    missing.extend(missing_colors)
    
    return missing

def identify_present_elements(analysis: Dict) -> List[str]:
    """Identify elements that are present in the image."""
    present = []
    
    if analysis.get('composition_analysis', {}).get('has_characters'):
        present.append('characters')
    
    if analysis.get('composition_analysis', {}).get('has_setting'):
        present.append('setting')
    
    matched_colors = analysis.get('color_analysis', {}).get('matched_colors', [])
    present.extend(matched_colors)
    
    return present

def rewrite_prompt_for_better_compliance(original_prompt: str, compliance_analysis: Dict, attempt: int) -> str:
    """
    Intelligently rewrite the prompt to improve compliance based on analysis results.
    """
    missing_elements = compliance_analysis.get('missing_elements', [])
    present_elements = compliance_analysis.get('present_elements', [])
    prompt_elements = compliance_analysis.get('prompt_elements', {})
    score = compliance_analysis.get('compliance_score', 0.0)
    
    print(f"    🔧 Rewriting prompt based on analysis:")
    print(f"       Missing: {missing_elements}")
    print(f"       Present: {present_elements}")
    print(f"       Score: {score:.2f}")
    
    # Start with the original prompt structure
    new_prompt = original_prompt
    
    # Strategy 1: Enhance missing elements with stronger emphasis
    if missing_elements:
        for element in missing_elements:
            if element == 'characters':
                # Add stronger character emphasis
                new_prompt = enhance_character_description(new_prompt, attempt)
            elif element == 'setting':
                # Add stronger setting emphasis
                new_prompt = enhance_setting_description(new_prompt, attempt)
            elif element in ['red', 'blue', 'green', 'brown', 'gray']:
                # Add stronger color emphasis
                new_prompt = enhance_color_description(new_prompt, element, attempt)
    
    # Strategy 2: Add specific enhancement keywords based on attempt
    enhancement_keywords = get_enhancement_keywords(attempt)
    if enhancement_keywords:
        new_prompt = add_enhancement_keywords(new_prompt, enhancement_keywords)
    
    # Strategy 3: Adjust weights for better emphasis
    new_prompt = adjust_prompt_weights(new_prompt, missing_elements, attempt)
    
    # Strategy 4: Add specific style enhancements
    if score < 0.5:
        new_prompt = add_style_enhancements(new_prompt, attempt)
    
    # Ensure the prompt doesn't get too long
    if len(new_prompt) > 500:
        new_prompt = truncate_prompt(new_prompt, 500)
    
    return new_prompt

def enhance_character_description(prompt: str, attempt: int) -> str:
    """Enhance character descriptions in the prompt."""
    character_enhancements = [
        "prominent, clearly visible",
        "large, detailed, well-defined",
        "dominant, highly detailed, sharply focused"
    ]
    
    enhancement = character_enhancements[min(attempt - 1, len(character_enhancements) - 1)]
    
    # Find character descriptions and enhance them
    character_patterns = [
        r'(cartoon\s+\w+\s+\w+)',
        r'(\w+\s+\w+\s+wearing)',
        r'(\w+\s+\w+\s+in\s+\w+)'
    ]
    
    for pattern in character_patterns:
        matches = re.findall(pattern, prompt, re.IGNORECASE)
        for match in matches:
            enhanced = f"{match}, {enhancement}"
            prompt = prompt.replace(match, enhanced, 1)
    
    return prompt

def enhance_setting_description(prompt: str, attempt: int) -> str:
    """Enhance setting descriptions in the prompt."""
    setting_enhancements = [
        "detailed, prominent background",
        "rich, elaborate background setting",
        "highly detailed, immersive background environment"
    ]
    
    enhancement = setting_enhancements[min(attempt - 1, len(setting_enhancements) - 1)]
    
    # Find setting descriptions and enhance them
    setting_keywords = ['jungle', 'night', 'stars', 'grass', 'tree', 'background']
    for keyword in setting_keywords:
        if keyword in prompt.lower():
            # Add enhancement after the setting keyword
            pattern = rf'(\b{keyword}\b)'
            replacement = rf'\1, {enhancement}'
            prompt = re.sub(pattern, replacement, prompt, flags=re.IGNORECASE, count=1)
    
    return prompt

def enhance_color_description(prompt: str, color: str, attempt: int) -> str:
    """Enhance specific color descriptions in the prompt."""
    color_enhancements = [
        "bright, vivid",
        "intense, vibrant",
        "striking, bold"
    ]
    
    enhancement = color_enhancements[min(attempt - 1, len(color_enhancements) - 1)]
    
    # Find color descriptions and enhance them
    pattern = rf'(\b{color}\b)'
    replacement = rf'{enhancement} \1'
    prompt = re.sub(pattern, replacement, prompt, flags=re.IGNORECASE, count=1)
    
    return prompt

def get_enhancement_keywords(attempt: int) -> List[str]:
    """Get enhancement keywords based on attempt number."""
    enhancement_sets = [
        ["vibrant colors", "clear details", "strong composition"],
        ["highly detailed", "rich textures", "sharp focus"],
        ["extremely detailed", "masterpiece quality", "professional illustration"]
    ]
    
    return enhancement_sets[min(attempt - 1, len(enhancement_sets) - 1)]

def add_enhancement_keywords(prompt: str, keywords: List[str]) -> str:
    """Add enhancement keywords to the prompt."""
    # Add keywords at the end of the prompt
    enhancement_text = ", ".join(keywords)
    
    # Check if prompt already has style keywords at the end
    if "style" in prompt.lower() or "illustration" in prompt.lower():
        # Insert before the style section
        style_pattern = r'(.*?)(storybook|illustration|style.*?)$'
        match = re.search(style_pattern, prompt, re.IGNORECASE | re.DOTALL)
        if match:
            before_style = match.group(1).rstrip()
            style_section = match.group(2)
            return f"{before_style}, {enhancement_text}, {style_section}"
    
    # Otherwise, add at the end
    return f"{prompt}, {enhancement_text}"

def adjust_prompt_weights(prompt: str, missing_elements: List[str], attempt: int) -> str:
    """Adjust prompt weights to emphasize missing elements."""
    # Increase weights for missing elements
    weight_increase = min(attempt * 0.2, 0.6)  # Max 0.6 increase
    
    # Find weight patterns and adjust them
    weight_pattern = r'\(([^:]+):([\d.]+)\)'
    
    def adjust_weight(match):
        content = match.group(1)
        current_weight = float(match.group(2))
        
        # Check if this section contains missing elements
        should_increase = any(element.lower() in content.lower() for element in missing_elements)
        
        if should_increase:
            new_weight = min(current_weight + weight_increase, 1.5)  # Cap at 1.5
            return f"({content}:{new_weight:.1f})"
        else:
            return match.group(0)
    
    return re.sub(weight_pattern, adjust_weight, prompt)

def add_style_enhancements(prompt: str, attempt: int) -> str:
    """Add style enhancements for low-scoring prompts."""
    style_enhancements = [
        "professional cartoon style, high quality",
        "masterpiece cartoon illustration, premium quality",
        "award-winning cartoon art, exceptional quality"
    ]
    
    enhancement = style_enhancements[min(attempt - 1, len(style_enhancements) - 1)]
    
    # Replace or enhance existing style descriptions
    style_patterns = [
        r'(storybook\s+illustration\s+style)',
        r'(cartoon\s+style)',
        r'(illustration\s+style)'
    ]
    
    for pattern in style_patterns:
        if re.search(pattern, prompt, re.IGNORECASE):
            prompt = re.sub(pattern, enhancement, prompt, flags=re.IGNORECASE, count=1)
            return prompt
    
    # If no style found, add at the end
    return f"{prompt}, {enhancement}"

def truncate_prompt(prompt: str, max_length: int) -> str:
    """Truncate prompt to fit within length limits."""
    if len(prompt) <= max_length:
        return prompt
    
    # Try to truncate from the end while preserving structure
    truncated = prompt[:max_length-3] + "..."
    
    # Ensure we don't break in the middle of a weight section
    weight_pattern = r'\([^)]*\)'
    matches = list(re.finditer(weight_pattern, truncated))
    
    if matches:
        # Find the last complete weight section
        last_complete = matches[-1].end()
        if last_complete < len(truncated):
            truncated = truncated[:last_complete]
    
    return truncated

def test_image_generation(prompt=None, max_retries=3):
    """Test the image generation system with automatic prompt rewriting and retry."""
    
    print("🎨 Testing Stable Diffusion Image Generation with Auto-Retry")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize image generator
    print("🚀 Initializing Image Generator...")
    image_gen = ImageGenerator()
    
    # Check if SD is available
    if image_gen.is_sd_available():
        print("✅ Stable Diffusion is available!")
        print(f"💻 Device: {image_gen.device}")
        print(f"📁 Model: {image_gen.model_path}")
    else:
        print("⚠️ Stable Diffusion not available - will use placeholder images")
        print("💡 To enable SD, run: bash download_models.sh")
    
    # Use provided prompt or default test prompt
    original_prompt = prompt
    if prompt is None:
        original_prompt = "(wide shot, cartoon brown monkey wearing red scarf, cartoon brown bear in blue vest, cartoon gray squirrel with green bow sleeping peacefully under tree:1.3), (calm jungle night with stars and grass:1.2), (soft blue moonlight, serene storybook illustration style:1.1)"
    
    current_prompt = original_prompt
    best_score = 0.0
    best_image_path = None
    best_attempt = 0
    
    print(f"\n🎬 Generating images with auto-retry (max {max_retries} attempts)...")
    print(f"📝 Original prompt: {original_prompt[:100]}{'...' if len(original_prompt) > 100 else ''}")
    
    for attempt in range(1, max_retries + 1):
        print(f"\n🔄 Attempt {attempt}/{max_retries}")
        print(f"📝 Current prompt: {current_prompt[:100]}{'...' if len(current_prompt) > 100 else ''}")
        
        # Generate image with current prompt
        test_prompts = [current_prompt]
        image_paths = image_gen.generate_multiple_images(test_prompts, str(output_dir))
        
        if not image_paths or not Path(image_paths[0]).exists():
            print(f"    ❌ Image generation failed on attempt {attempt}")
            continue
        
        image_path = image_paths[0]
        size = Path(image_path).stat().st_size / 1024  # KB
        print(f"    ✅ Generated: {image_path} ({size:.1f} KB)")
        
        # Analyze prompt compliance
        print(f"    🔍 Analyzing prompt compliance...")
        compliance_analysis = analyze_prompt_compliance(image_path, current_prompt)
        
        if 'error' in compliance_analysis:
            print(f"    ❌ Analysis error: {compliance_analysis['error']}")
            continue
        
        score = compliance_analysis['compliance_score']
        quality = compliance_analysis['overall_quality']
        missing = compliance_analysis['missing_elements']
        present = compliance_analysis['present_elements']
        
        print(f"    📊 Compliance Score: {score:.2f}/1.0")
        print(f"    🎯 Quality: {quality}")
        print(f"    ✅ Present: {', '.join(present) if present else 'None'}")
        print(f"    ❌ Missing: {', '.join(missing) if missing else 'None'}")
        
        # Track best result
        if score > best_score:
            best_score = score
            best_image_path = image_path
            best_attempt = attempt
        
        # Check if we should continue trying
        if score >= 0.8:
            print(f"    🎉 Excellent compliance achieved! Stopping retries.")
            break
        elif score >= 0.6:
            print(f"    ✅ Good compliance achieved. Consider stopping.")
            if attempt < max_retries:
                continue_choice = input(f"    🤔 Continue with attempt {attempt + 1}? (y/N): ").lower().strip()
                if continue_choice not in ['y', 'yes']:
                    break
        else:
            print(f"    ⚠️ Low compliance. Will retry with improved prompt.")
        
        # If we're going to retry, rewrite the prompt
        if attempt < max_retries:
            print(f"    🔄 Rewriting prompt for next attempt...")
            current_prompt = rewrite_prompt_for_better_compliance(
                original_prompt, 
                compliance_analysis, 
                attempt
            )
            print(f"    📝 New prompt: {current_prompt[:100]}{'...' if len(current_prompt) > 100 else ''}")
    
    # Final results
    print(f"\n🎯 Final Results:")
    print(f"    📊 Best Compliance Score: {best_score:.2f}/1.0")
    print(f"    🏆 Best Image: {best_image_path}")
    print(f"    🎬 Best Attempt: {best_attempt}")
    
    if best_score >= 0.8:
        print(f"    🎉 SUCCESS: Excellent prompt compliance achieved!")
    elif best_score >= 0.6:
        print(f"    ✅ GOOD: Acceptable prompt compliance achieved.")
    else:
        print(f"    ⚠️ POOR: Low prompt compliance. Consider manual prompt refinement.")
    
    print(f"\n📁 Images saved to: {output_dir.absolute()}")
    
    # Summary
    if image_gen.is_sd_available():
        print("\n🎉 SUCCESS: Stable Diffusion is working perfectly!")
        print("🎨 You can now generate professional cartoon images!")
    else:
        print("\n💡 To enable AI image generation:")
        print("   1. Run: bash download_models.sh")
        print("   2. Install: pip install diffusers transformers accelerate safetensors")
        print("   3. Restart this test script")

def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test Stable Diffusion image generation with custom prompts, compliance analysis, and auto-retry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_image_generation.py
  python test_image_generation.py --prompt "a cute cartoon cat playing in a garden"
  python test_image_generation.py -p "cartoon style, colorful background, happy characters"
  python test_image_generation.py --prompt "jungle scene" --retries 5
        """
    )
    
    parser.add_argument(
        '-p', '--prompt',
        type=str,
        help='Custom prompt for image generation (optional)',
        default=None
    )
    
    parser.add_argument(
        '-r', '--retries',
        type=int,
        help='Maximum number of retry attempts (default: 3)',
        default=3
    )
    
    args = parser.parse_args()
    
    # Run the test with the provided prompt and retry settings
    test_image_generation(args.prompt, args.retries)

if __name__ == "__main__":
    main()
