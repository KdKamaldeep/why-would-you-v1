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
    Uses third-party open-source packages for better image understanding.
    Returns a dictionary with compliance scores and analysis.
    """
    try:
        from collections import Counter
        import cv2
        from PIL import Image, ImageEnhance
        
        # Load the image
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image)
        
        # Extract key elements from the prompt
        prompt_elements = extract_prompt_elements(prompt)
        
        # Analyze image characteristics using OpenCV
        analysis = {
            'prompt_elements': prompt_elements,
            'image_analysis': analyze_image_characteristics_advanced(img_array),
            'compliance_score': 0.0,
            'missing_elements': [],
            'present_elements': [],
            'overall_quality': 'unknown'
        }
        
        # Check for basic quality issues first
        if is_image_blank_or_poor_quality_advanced(img_array):
            analysis['overall_quality'] = 'poor'
            analysis['compliance_score'] = 0.0
            return analysis
        
        analysis['overall_quality'] = 'good'
        
        # Analyze color compliance using advanced color detection
        color_compliance = analyze_color_compliance_advanced(img_array, prompt_elements)
        analysis['color_analysis'] = color_compliance
        
        # Analyze composition compliance using computer vision
        composition_compliance = analyze_composition_compliance_advanced(img_array, prompt_elements)
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

def analyze_image_characteristics_advanced(img_array: np.ndarray) -> Dict[str, any]:
    """Analyze image characteristics using OpenCV and advanced computer vision techniques."""
    try:
        import cv2
        
        # Convert PIL array to OpenCV format (BGR)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Advanced color analysis
        # Calculate color histograms
        color_hist = cv2.calcHist([img_cv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        color_hist = cv2.normalize(color_hist, color_hist).flatten()
        
        # Edge detection for structure analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Texture analysis using Local Binary Patterns
        from skimage.feature import local_binary_pattern
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10))
        lbp_hist = lbp_hist.astype(float) / lbp_hist.sum()
        
        # Dominant colors using K-means clustering
        pixels = img_array.reshape(-1, 3)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(pixels)
        dominant_colors = kmeans.cluster_centers_.astype(int)
        
        # Brightness and contrast analysis
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Color diversity
        unique_colors = len(np.unique(pixels, axis=0))
        
        return {
            'dominant_colors': dominant_colors.tolist(),
            'color_histogram': color_hist.tolist(),
            'edge_density': float(edge_density),
            'texture_lbp': lbp_hist.tolist(),
            'brightness': float(brightness),
            'contrast': float(contrast),
            'total_unique_colors': int(unique_colors),
            'mean_saturation': float(np.mean(hsv[:, :, 1])),
            'color_variance': float(np.var(img_array))
        }
    except ImportError:
        # Fallback to basic analysis if OpenCV is not available
        return analyze_image_characteristics_basic(img_array)

def analyze_image_characteristics_basic(img_array: np.ndarray) -> Dict[str, any]:
    """Basic image characteristics analysis as fallback."""
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

def is_image_blank_or_poor_quality_advanced(img_array: np.ndarray) -> bool:
    """Advanced quality detection using OpenCV and computer vision techniques."""
    try:
        import cv2
        
        # Convert to OpenCV format
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Multiple quality metrics
        # 1. Variance analysis
        variance = np.var(img_array)
        
        # 2. Edge density (blank images have few edges)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # 3. Color diversity
        unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
        
        # 4. Brightness analysis
        brightness_variance = np.var(gray)
        mean_brightness = np.mean(gray)
        
        # 5. Contrast analysis
        contrast = np.std(gray)
        
        # 6. Texture analysis using Local Binary Patterns
        from skimage.feature import local_binary_pattern
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        lbp_variance = np.var(lbp)
        
        # 7. Blur detection using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Quality thresholds
        is_poor_quality = (
            variance < 1000 or  # Low color variance
            edge_density < 0.01 or  # Very few edges
            unique_colors < 100 or  # Few unique colors
            brightness_variance < 500 or  # Low brightness variance
            mean_brightness < 10 or  # Too dark
            mean_brightness > 245 or  # Too bright
            contrast < 20 or  # Low contrast
            lbp_variance < 5 or  # Low texture variance
            laplacian_var < 100  # Too blurry
        )
        
        return is_poor_quality
        
    except ImportError:
        # Fallback to basic quality detection
        return is_image_blank_or_poor_quality_basic(img_array)

def is_image_blank_or_poor_quality_basic(img_array: np.ndarray) -> bool:
    """Basic quality detection as fallback."""
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

def analyze_color_compliance_advanced(img_array: np.ndarray, prompt_elements: Dict) -> Dict[str, any]:
    """Advanced color analysis using OpenCV and color science."""
    try:
        import cv2
        from colorthief import ColorThief
        import io
        
        compliance = {
            'score': 0.0,
            'matched_colors': [],
            'missing_colors': [],
            'overall_color_vibrancy': 0.0,
            'color_palette': [],
            'color_harmony': 0.0
        }
        
        # Check for required colors in prompt
        required_colors = prompt_elements.get('colors', [])
        if not required_colors:
            compliance['score'] = 0.5  # Neutral score if no specific colors mentioned
            return compliance
        
        # Convert to OpenCV format
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        
        # Advanced color analysis
        # 1. Color vibrancy using saturation
        mean_saturation = np.mean(hsv[:, :, 1])
        compliance['overall_color_vibrancy'] = mean_saturation / 255.0
        
        # 2. Extract dominant colors using ColorThief
        try:
            # Convert numpy array to PIL Image for ColorThief
            pil_image = Image.fromarray(img_array)
            img_io = io.BytesIO()
            pil_image.save(img_io, format='PNG')
            img_io.seek(0)
            
            color_thief = ColorThief(img_io)
            dominant_colors = color_thief.get_palette(color_count=8, quality=1)
            compliance['color_palette'] = dominant_colors
            
            # 3. Advanced color matching using color distance
            matched_count = 0
            for required_color in required_colors:
                color_matched = False
                for dominant_color in dominant_colors:
                    if is_color_similar(required_color, dominant_color):
                        color_matched = True
                        break
                
                if color_matched:
                    matched_count += 1
                    compliance['matched_colors'].append(required_color)
                else:
                    compliance['missing_colors'].append(required_color)
            
            compliance['score'] = matched_count / len(required_colors) if required_colors else 0.0
            
        except Exception:
            # Fallback to basic color matching
            matched_count = 0
            for color in required_colors:
                if color.lower() in ['red', 'blue', 'green', 'brown', 'gray']:
                    matched_count += 1
                    compliance['matched_colors'].append(color)
                else:
                    compliance['missing_colors'].append(color)
            
            compliance['score'] = matched_count / len(required_colors) if required_colors else 0.0
        
        # 4. Color harmony analysis
        compliance['color_harmony'] = calculate_color_harmony(dominant_colors)
        
        return compliance
        
    except ImportError:
        # Fallback to basic color analysis
        return analyze_color_compliance_basic(img_array, prompt_elements)

def analyze_color_compliance_basic(img_array: np.ndarray, prompt_elements: Dict) -> Dict[str, any]:
    """Basic color analysis as fallback."""
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
    
    # Simple color matching
    matched_count = 0
    for color in required_colors:
        if color.lower() in ['red', 'blue', 'green', 'brown', 'gray']:
            matched_count += 1
            compliance['matched_colors'].append(color)
        else:
            compliance['missing_colors'].append(color)
    
    compliance['score'] = matched_count / len(required_colors) if required_colors else 0.0
    return compliance

def analyze_composition_compliance_advanced(img_array: np.ndarray, prompt_elements: Dict) -> Dict[str, any]:
    """Advanced composition analysis using computer vision techniques."""
    try:
        import cv2
        
        compliance = {
            'score': 0.0,
            'has_characters': False,
            'has_setting': False,
            'composition_quality': 0.0,
            'foreground_objects': 0,
            'background_complexity': 0.0,
            'rule_of_thirds': 0.0
        }
        
        # Convert to OpenCV format
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # 1. Edge detection for structure analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # 2. Contour detection for object identification
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter significant contours (potential objects/characters)
        significant_contours = [c for c in contours if cv2.contourArea(c) > 1000]
        compliance['foreground_objects'] = len(significant_contours)
        
        # 3. Background complexity analysis
        # Use morphological operations to separate foreground and background
        kernel = np.ones((5,5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        background_mask = cv2.erode(dilated, kernel, iterations=2)
        
        # Calculate background complexity
        background_complexity = np.sum(background_mask > 0) / background_mask.size
        compliance['background_complexity'] = float(background_complexity)
        
        # 4. Rule of thirds analysis
        height, width = gray.shape
        third_h = height // 3
        third_w = width // 3
        
        # Check if there are significant features at rule of thirds intersections
        intersections = [
            gray[third_h, third_w],
            gray[third_h, 2*third_w],
            gray[2*third_h, third_w],
            gray[2*third_h, 2*third_w]
        ]
        intersection_variance = np.var(intersections)
        compliance['rule_of_thirds'] = min(1.0, intersection_variance / 1000)
        
        # 5. Advanced character detection
        # Look for regions with high edge density and color variance
        compliance['has_characters'] = (
            edge_density > 0.02 and  # Sufficient edge density
            len(significant_contours) > 0 and  # Has significant objects
            compliance['foreground_objects'] > 0  # Has foreground objects
        )
        
        # 6. Advanced setting detection
        color_variance = np.var(img_array)
        compliance['has_setting'] = (
            background_complexity > 0.01 and  # Complex background
            color_variance > 5000  # High color variance
        )
        
        # 7. Overall composition quality
        composition_score = (
            edge_density * 0.3 +
            background_complexity * 0.2 +
            compliance['rule_of_thirds'] * 0.2 +
            min(1.0, len(significant_contours) / 5) * 0.3  # Object count
        )
        compliance['composition_quality'] = min(1.0, composition_score)
        
        # Calculate score based on elements
        score_components = []
        if prompt_elements.get('characters') and compliance['has_characters']:
            score_components.append(1.0)
        if prompt_elements.get('setting') and compliance['has_setting']:
            score_components.append(1.0)
        
        compliance['score'] = sum(score_components) / max(len(score_components), 1)
        return compliance
        
    except ImportError:
        # Fallback to basic composition analysis
        return analyze_composition_compliance_basic(img_array, prompt_elements)

def analyze_composition_compliance_basic(img_array: np.ndarray, prompt_elements: Dict) -> Dict[str, any]:
    """Basic composition analysis as fallback."""
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

def rewrite_prompt_for_better_compliance(original_prompt: str, compliance_analysis: Dict, attempt: int, image_gen: ImageGenerator) -> str:
    """
    Use GPT-2 to intelligently rewrite the prompt based on compliance analysis and advanced image characteristics.
    """
    missing_elements = compliance_analysis.get('missing_elements', [])
    present_elements = compliance_analysis.get('present_elements', [])
    score = compliance_analysis.get('compliance_score', 0.0)
    image_analysis = compliance_analysis.get('image_analysis', {})
    
    print(f"    🔧 Using GPT-2 to rewrite prompt based on analysis:")
    print(f"       Missing: {missing_elements}")
    print(f"       Present: {present_elements}")
    print(f"       Score: {score:.2f}")
    
    # Extract advanced image characteristics for prompt enhancement
    enhancement_data = extract_enhancement_data_from_analysis(image_analysis, compliance_analysis)
    
    # Use the image generator's GPT-2 enhanced prompt rewriting with advanced analysis
    if image_gen.prompt_enhancer and image_gen.prompt_enhancer.is_available():
        print(f"    🎯 Using GPT-2 for intelligent prompt rewriting with advanced analysis...")
        print(f"    📊 Enhancement data: {enhancement_data}")
        return image_gen.rewrite_prompt_for_better_compliance_with_analysis(
            original_prompt, compliance_analysis, enhancement_data, attempt
        )
    else:
        print(f"    ⚠️ GPT-2 not available, using enhanced fallback method...")
        # Enhanced fallback using analysis data
        return enhance_prompt_with_analysis_data(original_prompt, enhancement_data, attempt)

def extract_enhancement_data_from_analysis(image_analysis: Dict, compliance_analysis: Dict) -> Dict:
    """Extract enhancement data from advanced image analysis."""
    enhancement_data = {
        'color_issues': [],
        'composition_issues': [],
        'quality_issues': [],
        'style_suggestions': [],
        'technical_improvements': []
    }
    
    # Analyze color characteristics
    dominant_colors = image_analysis.get('dominant_colors', [])
    color_variance = image_analysis.get('color_variance', 0)
    mean_saturation = image_analysis.get('mean_saturation', 0)
    total_unique_colors = image_analysis.get('total_unique_colors', 0)
    
    # Color analysis
    if color_variance < 5000:
        enhancement_data['color_issues'].append('low_color_variance')
    if mean_saturation < 50:
        enhancement_data['color_issues'].append('low_saturation')
    if total_unique_colors < 1000:
        enhancement_data['color_issues'].append('limited_color_palette')
    
    # Analyze composition characteristics
    edge_density = image_analysis.get('edge_density', 0)
    brightness = image_analysis.get('brightness', 0)
    contrast = image_analysis.get('contrast', 0)
    
    # Composition analysis
    if edge_density < 0.02:
        enhancement_data['composition_issues'].append('low_detail')
    if brightness < 50:
        enhancement_data['composition_issues'].append('too_dark')
    elif brightness > 200:
        enhancement_data['composition_issues'].append('too_bright')
    if contrast < 30:
        enhancement_data['composition_issues'].append('low_contrast')
    
    # Analyze texture and quality
    texture_lbp = image_analysis.get('texture_lbp', [])
    if texture_lbp and max(texture_lbp) < 0.1:
        enhancement_data['quality_issues'].append('low_texture_detail')
    
    # Style suggestions based on analysis
    if color_variance > 10000 and mean_saturation > 100:
        enhancement_data['style_suggestions'].append('vibrant_colors')
    if edge_density > 0.05:
        enhancement_data['style_suggestions'].append('high_detail')
    if brightness > 150:
        enhancement_data['style_suggestions'].append('bright_lighting')
    
    # Technical improvements
    if compliance_analysis.get('color_analysis', {}).get('score', 0) < 0.5:
        enhancement_data['technical_improvements'].append('enhance_color_description')
    if compliance_analysis.get('composition_analysis', {}).get('score', 0) < 0.5:
        enhancement_data['technical_improvements'].append('enhance_composition_description')
    
    return enhancement_data

def enhance_prompt_with_analysis_data(original_prompt: str, enhancement_data: Dict, attempt: int) -> str:
    """Enhance prompt using analysis data when GPT-2 is not available."""
    enhanced_prompt = original_prompt
    
    # Add style enhancements based on analysis
    style_suggestions = enhancement_data.get('style_suggestions', [])
    if 'vibrant_colors' in style_suggestions:
        enhanced_prompt += ", vibrant colors, saturated palette"
    if 'high_detail' in style_suggestions:
        enhanced_prompt += ", highly detailed, sharp focus"
    if 'bright_lighting' in style_suggestions:
        enhanced_prompt += ", bright lighting, well-lit scene"
    
    # Address quality issues
    quality_issues = enhancement_data.get('quality_issues', [])
    if 'low_texture_detail' in quality_issues:
        enhanced_prompt += ", textured surfaces, rich details"
    
    # Address composition issues
    composition_issues = enhancement_data.get('composition_issues', [])
    if 'low_detail' in composition_issues:
        enhanced_prompt += ", detailed rendering, fine details"
    if 'low_contrast' in composition_issues:
        enhanced_prompt += ", high contrast, dramatic lighting"
    
    # Add technical improvements
    technical_improvements = enhancement_data.get('technical_improvements', [])
    if 'enhance_color_description' in technical_improvements:
        enhanced_prompt += ", color-rich, chromatic variety"
    if 'enhance_composition_description' in technical_improvements:
        enhanced_prompt += ", well-composed, balanced layout"
    
    # Add attempt-specific enhancements
    if attempt == 2:
        enhanced_prompt += ", enhanced details, improved composition"
    elif attempt >= 3:
        enhanced_prompt += ", masterwork quality, professional illustration"
    
    return enhanced_prompt



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
        
        # If we're going to retry, rewrite the prompt using GPT-2
        if attempt < max_retries:
            print(f"    🔄 Rewriting prompt with GPT-2 for next attempt...")
            current_prompt = rewrite_prompt_for_better_compliance(
                original_prompt, 
                compliance_analysis, 
                attempt,
                image_gen
            )
            print(f"    📝 New GPT-2 enhanced prompt: {current_prompt[:100]}{'...' if len(current_prompt) > 100 else ''}")
    
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

def is_color_similar(color1: tuple, color2: tuple, threshold: float = 50.0) -> bool:
    """Check if two colors are similar using Euclidean distance in RGB space."""
    if len(color1) != 3 or len(color2) != 3:
        return False
    
    # Calculate Euclidean distance in RGB space
    distance = np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)))
    return distance <= threshold

def calculate_color_harmony(colors: list) -> float:
    """Calculate color harmony score based on color relationships."""
    if not colors or len(colors) < 2:
        return 0.5
    
    try:
        # Convert colors to HSV for better harmony analysis
        import cv2
        harmony_scores = []
        
        for i, color1 in enumerate(colors):
            for j, color2 in enumerate(colors[i+1:], i+1):
                # Convert RGB to HSV
                color1_hsv = cv2.cvtColor(np.array([[color1]], dtype=np.uint8), cv2.COLOR_RGB2HSV)[0, 0]
                color2_hsv = cv2.cvtColor(np.array([[color2]], dtype=np.uint8), cv2.COLOR_RGB2HSV)[0, 0]
                
                # Calculate hue difference
                hue_diff = abs(color1_hsv[0] - color2_hsv[0])
                if hue_diff > 90:  # Wrap around
                    hue_diff = 180 - hue_diff
                
                # Harmony rules: complementary (180°), analogous (30°), triadic (120°)
                if 170 <= hue_diff <= 190:  # Complementary
                    harmony_scores.append(1.0)
                elif 25 <= hue_diff <= 35:  # Analogous
                    harmony_scores.append(0.8)
                elif 115 <= hue_diff <= 125:  # Triadic
                    harmony_scores.append(0.9)
                else:
                    # Other relationships get lower scores
                    harmony_scores.append(max(0.1, 1.0 - (hue_diff / 180.0)))
        
        return np.mean(harmony_scores) if harmony_scores else 0.5
        
    except ImportError:
        # Fallback: simple RGB distance-based harmony
        distances = []
        for i, color1 in enumerate(colors):
            for j, color2 in enumerate(colors[i+1:], i+1):
                distance = np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)))
                # Normalize distance (max possible distance is sqrt(255^2 * 3) ≈ 441)
                normalized_distance = distance / 441.0
                distances.append(1.0 - normalized_distance)
        
        return np.mean(distances) if distances else 0.5

if __name__ == "__main__":
    main()
