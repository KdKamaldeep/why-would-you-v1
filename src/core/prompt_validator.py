#!/usr/bin/env python3
"""
Prompt Validator Module - Analyzes generated images and provides feedback for prompt improvement
"""

import logging
import os
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import time

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of prompt validation."""
    success: bool
    quality_score: float
    issues: List[str]
    suggestions: List[str]
    prompt_improvements: List[str]
    should_regenerate: bool

@dataclass
class ImageAnalysis:
    """Analysis of a generated image."""
    file_path: str
    prompt_used: str
    negative_prompt: str
    generation_time: float
    file_size: int
    dimensions: Tuple[int, int]
    quality_indicators: Dict[str, float]

class PromptValidator:
    """Validates generated images and provides prompt improvement feedback."""
    
    def __init__(self, validation_threshold: float = 0.7):
        self.validation_threshold = validation_threshold
        self.quality_indicators = {
            "clarity": 0.0,
            "composition": 0.0,
            "style_consistency": 0.0,
            "detail_quality": 0.0,
            "color_balance": 0.0
        }
        
        # Common generation issues and their solutions
        self.issue_solutions = {
            "blurry": [
                "Add 'sharp focus' to positive prompt",
                "Add 'high resolution' to positive prompt",
                "Remove 'blurry' from negative prompt if present"
            ],
            "distorted": [
                "Add 'proper proportions' to positive prompt",
                "Add 'well-formed' to positive prompt",
                "Add 'distorted, deformed' to negative prompt"
            ],
            "poor_composition": [
                "Add specific camera angles (e.g., 'medium shot', 'wide shot')",
                "Add composition descriptors (e.g., 'rule of thirds', 'balanced composition')",
                "Specify positioning (e.g., 'centered', 'off-center')"
            ],
            "style_inconsistency": [
                "Add specific style descriptors (e.g., 'realistic', 'photorealistic', 'anime style')",
                "Add 'consistent style' to positive prompt",
                "Add conflicting styles to negative prompt"
            ],
            "low_detail": [
                "Add 'detailed' to positive prompt",
                "Add 'high quality' to positive prompt",
                "Add 'textured' to positive prompt"
            ],
            "poor_colors": [
                "Add specific color descriptors",
                "Add 'vibrant colors' to positive prompt",
                "Add 'color harmony' to positive prompt"
            ]
        }
    
    def validate_generated_image(self, image_path: str, prompt: str, negative_prompt: str = "") -> ValidationResult:
        """
        Validate a generated image and provide improvement suggestions.
        
        Args:
            image_path: Path to the generated image
            prompt: Prompt used for generation
            negative_prompt: Negative prompt used for generation
            
        Returns:
            ValidationResult with analysis and suggestions
        """
        try:
            # Check if image exists
            if not os.path.exists(image_path):
                return ValidationResult(
                    success=False,
                    quality_score=0.0,
                    issues=["Image file not found"],
                    suggestions=["Check file path and generation process"],
                    prompt_improvements=[],
                    should_regenerate=True
                )
            
            # Analyze image quality
            analysis = self._analyze_image_quality(image_path)
            
            # Identify issues
            issues = self._identify_issues(analysis, prompt, negative_prompt)
            
            # Generate suggestions
            suggestions = self._generate_suggestions(issues)
            
            # Generate prompt improvements
            prompt_improvements = self._generate_prompt_improvements(issues, prompt, negative_prompt)
            
            # Calculate overall quality score
            quality_score = self._calculate_quality_score(analysis)
            
            # Determine if regeneration is needed
            should_regenerate = quality_score < self.validation_threshold or len(issues) > 2
            
            return ValidationResult(
                success=quality_score >= self.validation_threshold,
                quality_score=quality_score,
                issues=issues,
                suggestions=suggestions,
                prompt_improvements=prompt_improvements,
                should_regenerate=should_regenerate
            )
            
        except Exception as e:
            logger.error(f"❌ Error validating image: {e}")
            return ValidationResult(
                success=False,
                quality_score=0.0,
                issues=[f"Validation error: {str(e)}"],
                suggestions=["Check image format and try again"],
                prompt_improvements=[],
                should_regenerate=True
            )
    
    def _analyze_image_quality(self, image_path: str) -> ImageAnalysis:
        """Analyze the quality of a generated image."""
        try:
            from PIL import Image
            import numpy as np
            
            # Load image
            img = Image.open(image_path)
            
            # Get basic information
            file_size = os.path.getsize(image_path)
            dimensions = img.size
            generation_time = time.time() - os.path.getmtime(image_path)
            
            # Convert to numpy array for analysis
            img_array = np.array(img)
            
            # Analyze quality indicators
            quality_indicators = {
                "clarity": self._analyze_clarity(img_array),
                "composition": self._analyze_composition(img_array),
                "style_consistency": self._analyze_style_consistency(img_array),
                "detail_quality": self._analyze_detail_quality(img_array),
                "color_balance": self._analyze_color_balance(img_array)
            }
            
            return ImageAnalysis(
                file_path=image_path,
                prompt_used="",  # Will be filled by caller
                negative_prompt="",  # Will be filled by caller
                generation_time=generation_time,
                file_size=file_size,
                dimensions=dimensions,
                quality_indicators=quality_indicators
            )
            
        except Exception as e:
            logger.error(f"❌ Error analyzing image quality: {e}")
            # Return default analysis
            return ImageAnalysis(
                file_path=image_path,
                prompt_used="",
                negative_prompt="",
                generation_time=0.0,
                file_size=0,
                dimensions=(0, 0),
                quality_indicators={k: 0.0 for k in self.quality_indicators.keys()}
            )
    
    def _analyze_clarity(self, img_array: np.ndarray) -> float:
        """Analyze image clarity using edge detection."""
        try:
            from scipy import ndimage
            
            # Convert to grayscale if needed
            if len(img_array.shape) == 3:
                gray = np.mean(img_array, axis=2)
            else:
                gray = img_array
            
            # Apply edge detection
            edges = ndimage.sobel(gray)
            
            # Calculate edge density (higher = sharper)
            edge_density = np.mean(np.abs(edges))
            
            # Normalize to 0-1 range
            clarity_score = min(1.0, edge_density / 50.0)
            
            return clarity_score
            
        except Exception:
            return 0.5  # Default score
    
    def _analyze_composition(self, img_array: np.ndarray) -> float:
        """Analyze image composition."""
        try:
            # Simple composition analysis based on brightness distribution
            if len(img_array.shape) == 3:
                brightness = np.mean(img_array, axis=2)
            else:
                brightness = img_array
            
            # Check for balanced brightness distribution
            brightness_std = np.std(brightness)
            brightness_mean = np.mean(brightness)
            
            # Good composition has moderate contrast
            composition_score = min(1.0, brightness_std / 50.0)
            
            return composition_score
            
        except Exception:
            return 0.5  # Default score
    
    def _analyze_style_consistency(self, img_array: np.ndarray) -> float:
        """Analyze style consistency."""
        try:
            # Analyze color palette consistency
            if len(img_array.shape) == 3:
                # Calculate color variance
                color_variance = np.var(img_array, axis=(0, 1))
                avg_variance = np.mean(color_variance)
                
                # Lower variance = more consistent style
                style_score = max(0.0, 1.0 - (avg_variance / 1000.0))
                
                return style_score
            else:
                return 0.5
                
        except Exception:
            return 0.5  # Default score
    
    def _analyze_detail_quality(self, img_array: np.ndarray) -> float:
        """Analyze detail quality."""
        try:
            # Analyze local variance (detail level)
            if len(img_array.shape) == 3:
                gray = np.mean(img_array, axis=2)
            else:
                gray = img_array
            
            # Calculate local variance using convolution
            from scipy import ndimage
            local_variance = ndimage.generic_filter(gray, np.var, size=5)
            
            # Higher local variance = more detail
            detail_score = min(1.0, np.mean(local_variance) / 100.0)
            
            return detail_score
            
        except Exception:
            return 0.5  # Default score
    
    def _analyze_color_balance(self, img_array: np.ndarray) -> float:
        """Analyze color balance."""
        try:
            if len(img_array.shape) != 3:
                return 0.5
            
            # Analyze color distribution
            r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
            
            # Check for color balance (similar means)
            r_mean, g_mean, b_mean = np.mean(r), np.mean(g), np.mean(b)
            
            # Calculate color balance score
            max_mean = max(r_mean, g_mean, b_mean)
            min_mean = min(r_mean, g_mean, b_mean)
            
            if max_mean > 0:
                balance_score = 1.0 - ((max_mean - min_mean) / max_mean)
            else:
                balance_score = 0.5
            
            return max(0.0, balance_score)
            
        except Exception:
            return 0.5  # Default score
    
    def _identify_issues(self, analysis: ImageAnalysis, prompt: str, negative_prompt: str) -> List[str]:
        """Identify issues based on image analysis and prompts."""
        issues = []
        
        # Check quality indicators
        for indicator, score in analysis.quality_indicators.items():
            if score < 0.4:
                if indicator == "clarity":
                    issues.append("blurry")
                elif indicator == "composition":
                    issues.append("poor_composition")
                elif indicator == "style_consistency":
                    issues.append("style_inconsistency")
                elif indicator == "detail_quality":
                    issues.append("low_detail")
                elif indicator == "color_balance":
                    issues.append("poor_colors")
        
        # Check file size (too small might indicate poor quality)
        if analysis.file_size < 50000:  # Less than 50KB
            issues.append("low_file_size")
        
        # Check dimensions
        if analysis.dimensions[0] < 512 or analysis.dimensions[1] < 512:
            issues.append("low_resolution")
        
        # Check prompt-specific issues
        if "blurry" in prompt.lower() and "blurry" not in negative_prompt.lower():
            issues.append("blurry_in_positive_prompt")
        
        if len(prompt.split()) < 5:
            issues.append("too_short_prompt")
        
        return list(set(issues))  # Remove duplicates
    
    def _generate_suggestions(self, issues: List[str]) -> List[str]:
        """Generate suggestions based on identified issues."""
        suggestions = []
        
        for issue in issues:
            if issue in self.issue_solutions:
                suggestions.extend(self.issue_solutions[issue])
        
        # Add general suggestions
        if not suggestions:
            suggestions.append("Consider adding more specific details to the prompt")
            suggestions.append("Try adding quality boosters like 'high quality' or 'detailed'")
        
        return suggestions
    
    def _generate_prompt_improvements(self, issues: List[str], prompt: str, negative_prompt: str) -> List[str]:
        """Generate specific prompt improvements."""
        improvements = []
        
        # Add improvements based on issues
        if "blurry" in issues:
            improvements.append(f"Enhanced prompt: {prompt}, sharp focus, high resolution")
        
        if "low_detail" in issues:
            improvements.append(f"Enhanced prompt: {prompt}, detailed, high quality")
        
        if "poor_colors" in issues:
            improvements.append(f"Enhanced prompt: {prompt}, vibrant colors, color harmony")
        
        if "style_inconsistency" in issues:
            improvements.append(f"Enhanced prompt: {prompt}, consistent style, professional")
        
        if "too_short_prompt" in issues:
            improvements.append(f"Enhanced prompt: {prompt}, detailed, professional, high quality")
        
        # Add negative prompt improvements
        if "blurry" in issues and "blurry" not in negative_prompt:
            improvements.append(f"Enhanced negative: {negative_prompt}, blurry, low quality")
        
        if "distorted" in issues and "distorted" not in negative_prompt:
            improvements.append(f"Enhanced negative: {negative_prompt}, distorted, deformed")
        
        return improvements
    
    def _calculate_quality_score(self, analysis: ImageAnalysis) -> float:
        """Calculate overall quality score."""
        if not analysis.quality_indicators:
            return 0.5
        
        # Weighted average of quality indicators
        weights = {
            "clarity": 0.3,
            "composition": 0.2,
            "style_consistency": 0.2,
            "detail_quality": 0.2,
            "color_balance": 0.1
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for indicator, weight in weights.items():
            if indicator in analysis.quality_indicators:
                total_score += analysis.quality_indicators[indicator] * weight
                total_weight += weight
        
        if total_weight > 0:
            return total_score / total_weight
        else:
            return 0.5
    
    def validate_batch(self, image_paths: List[str], prompts: List[str], 
                      negative_prompts: List[str] = None) -> List[ValidationResult]:
        """Validate multiple generated images."""
        if negative_prompts is None:
            negative_prompts = [""] * len(image_paths)
        
        results = []
        
        for i, (image_path, prompt, negative_prompt) in enumerate(zip(image_paths, prompts, negative_prompts)):
            logger.info(f"🔍 Validating image {i+1}/{len(image_paths)}")
            result = self.validate_generated_image(image_path, prompt, negative_prompt)
            results.append(result)
        
        return results
    
    def generate_validation_report(self, results: List[ValidationResult], output_path: str = None) -> str:
        """Generate a comprehensive validation report."""
        report = []
        report.append("=" * 60)
        report.append("PROMPT VALIDATION REPORT")
        report.append("=" * 60)
        
        # Summary statistics
        total_images = len(results)
        successful_images = sum(1 for r in results if r.success)
        avg_quality = sum(r.quality_score for r in results) / total_images if total_images > 0 else 0
        
        report.append(f"📊 Summary:")
        report.append(f"   Total images: {total_images}")
        report.append(f"   Successful: {successful_images}/{total_images} ({successful_images/total_images*100:.1f}%)")
        report.append(f"   Average quality score: {avg_quality:.2f}")
        report.append("")
        
        # Detailed results
        for i, result in enumerate(results):
            report.append(f"🎬 Image {i+1}:")
            report.append(f"   Quality score: {result.quality_score:.2f}")
            report.append(f"   Success: {'✅' if result.success else '❌'}")
            
            if result.issues:
                report.append(f"   Issues: {', '.join(result.issues)}")
            
            if result.suggestions:
                report.append(f"   Suggestions: {', '.join(result.suggestions)}")
            
            if result.prompt_improvements:
                report.append(f"   Prompt improvements:")
                for improvement in result.prompt_improvements:
                    report.append(f"     - {improvement}")
            
            report.append("")
        
        # Common issues and solutions
        all_issues = []
        for result in results:
            all_issues.extend(result.issues)
        
        if all_issues:
            issue_counts = {}
            for issue in all_issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
            
            report.append("🔧 Common Issues and Solutions:")
            for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
                report.append(f"   {issue}: {count} occurrences")
                if issue in self.issue_solutions:
                    for solution in self.issue_solutions[issue][:2]:  # Show top 2 solutions
                        report.append(f"     → {solution}")
            report.append("")
        
        report_text = "\n".join(report)
        
        # Save to file if output path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"📄 Validation report saved to: {output_path}")
        
        return report_text


def test_prompt_validator():
    """Test function for prompt validator."""
    validator = PromptValidator()
    
    # Test with a sample image path (you would need a real image for full testing)
    test_image_path = "test_image.png"
    
    test_prompt = "a cat playing in a garden"
    test_negative_prompt = "blurry, low quality"
    
    print("Testing Prompt Validator:")
    print("=" * 40)
    
    # Note: This will fail without a real image, but shows the interface
    try:
        result = validator.validate_generated_image(test_image_path, test_prompt, test_negative_prompt)
        print(f"Success: {result.success}")
        print(f"Quality Score: {result.quality_score:.2f}")
        print(f"Issues: {result.issues}")
        print(f"Suggestions: {result.suggestions}")
    except Exception as e:
        print(f"Test completed (expected error without real image): {e}")


if __name__ == "__main__":
    test_prompt_validator()
