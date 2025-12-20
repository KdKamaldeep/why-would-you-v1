#!/usr/bin/env python3
"""
Professional Prompt Enhancer Module - Advanced prompt optimization for diffusion models
"""

import logging
import re
import json
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
import os

logger = logging.getLogger(__name__)

@dataclass
class PromptAnalysis:
    """Analysis results for a prompt."""
    clarity_score: float
    structure_score: float
    specificity_score: float
    issues: List[str]
    suggestions: List[str]
    enhanced_prompt: str
    optimized_negative_prompt: str

class ProfessionalPromptEnhancer:
    """Advanced prompt optimization system for diffusion models."""
    
    def __init__(self):
        self.device = 'cuda' if self._check_cuda() else 'cpu'
        self.model = None
        self.tokenizer = None
        self._initialize_enhancement_model()
        
        # Professional prompt templates
        self.prompt_templates = {
            "character": {
                "structure": "{subject}, {appearance}, {clothing}, {pose}, {expression}, {lighting}, {style}, {quality}",
                "required_elements": ["subject", "appearance", "clothing", "pose", "expression"]
            },
            "scene": {
                "structure": "{setting}, {subject}, {action}, {lighting}, {atmosphere}, {style}, {quality}",
                "required_elements": ["setting", "subject", "action"]
            },
            "object": {
                "structure": "{object}, {description}, {context}, {lighting}, {style}, {quality}",
                "required_elements": ["object", "description"]
            }
        }
        
        # Quality boosters for different styles
        self.quality_boosters = {
            "cartoon": [
                "high quality", "detailed", "professional", "clean lines", "vibrant colors",
                "smooth shading", "crisp edges", "professional illustration"
            ],
            "realistic": [
                "photorealistic", "high resolution", "detailed textures", "natural lighting",
                "professional photography", "sharp focus", "4k quality"
            ],
            "anime": [
                "anime style", "detailed", "high quality", "clean art", "professional illustration",
                "smooth shading", "crisp lines", "studio quality"
            ]
        }
        
        # Common diffusion model issues and solutions
        self.issue_patterns = {
            "vague_subject": (r"\b(a|an|the)\s+(\w+)\b", "Replace with specific descriptions"),
            "missing_style": (r"(?<!style|art|illustration|painting|drawing)\b", "Add style descriptors"),
            "poor_structure": (r"^[^,]+$", "Add comma-separated elements"),
            "weak_quality": (r"(?<!quality|detailed|professional|high)\b", "Add quality boosters")
        }
    
    def _check_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _initialize_enhancement_model(self):
        """Initialize optional enhancement model for advanced analysis."""
        try:
            from transformers import AutoTokenizer, AutoModel
            # Use a smaller model for efficiency
            model_name = "microsoft/DialoGPT-medium"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            if self.device == 'cuda':
                self.model.to(self.device)
            logger.info("✅ Enhancement model loaded for advanced prompt analysis")
        except Exception as e:
            logger.info(f"📝 Using rule-based enhancement (model loading failed: {e})")
            self.model = None
    
    def analyze_prompt(self, prompt: str, style: str = "realistic") -> PromptAnalysis:
        """
        Comprehensive prompt analysis and optimization.
        
        Args:
            prompt: Original prompt to analyze
            style: Target style (realistic, anime, etc.)
            
        Returns:
            PromptAnalysis with scores, issues, and suggestions
        """
        issues = []
        suggestions = []
        
        # Analyze prompt structure
        structure_score = self._analyze_structure(prompt)
        if structure_score < 0.6:
            issues.append("Poor prompt structure - lacks clear elements")
            suggestions.append("Use comma-separated format with specific descriptors")
        
        # Analyze clarity
        clarity_score = self._analyze_clarity(prompt)
        if clarity_score < 0.7:
            issues.append("Unclear or vague descriptions")
            suggestions.append("Add specific details about appearance, actions, and context")
        
        # Analyze specificity
        specificity_score = self._analyze_specificity(prompt)
        if specificity_score < 0.5:
            issues.append("Too generic - lacks specific details")
            suggestions.append("Include specific colors, textures, poses, and expressions")
        
        # Generate enhanced prompt
        enhanced_prompt = self._enhance_prompt(prompt, style)
        
        # Generate optimized negative prompt
        optimized_negative = self._generate_negative_prompt(prompt, style)
        
        return PromptAnalysis(
            clarity_score=clarity_score,
            structure_score=structure_score,
            specificity_score=specificity_score,
            issues=issues,
            suggestions=suggestions,
            enhanced_prompt=enhanced_prompt,
            optimized_negative_prompt=optimized_negative
        )
    
    def _analyze_structure(self, prompt: str) -> float:
        """Analyze prompt structure quality."""
        # Check for comma separation
        comma_count = prompt.count(',')
        if comma_count < 2:
            return 0.3
        
        # Check for balanced elements
        elements = [e.strip() for e in prompt.split(',')]
        if len(elements) < 4:
            return 0.5
        
        # Check for proper formatting
        if any(len(elem) < 3 for elem in elements):
            return 0.6
        
        return min(1.0, comma_count / 8.0)
    
    def _analyze_clarity(self, prompt: str) -> float:
        """Analyze prompt clarity."""
        # Check for vague words
        vague_words = ['thing', 'stuff', 'something', 'nice', 'good', 'beautiful', 'pretty']
        vague_count = sum(1 for word in vague_words if word in prompt.lower())
        
        # Check for specific descriptors
        specific_words = ['red', 'blue', 'green', 'tall', 'short', 'round', 'square', 'bright', 'dark']
        specific_count = sum(1 for word in specific_words if word in prompt.lower())
        
        if vague_count > 0:
            return max(0.3, 1.0 - (vague_count * 0.2))
        
        return min(1.0, specific_count / 5.0)
    
    def _analyze_specificity(self, prompt: str) -> float:
        """Analyze prompt specificity."""
        # Count descriptive elements
        descriptive_patterns = [
            r'\b\w+ color\b', r'\b\w+ hair\b', r'\b\w+ eyes\b', r'\b\w+ clothing\b',
            r'\b\w+ pose\b', r'\b\w+ expression\b', r'\b\w+ lighting\b', r'\b\w+ background\b'
        ]
        
        specificity_count = 0
        for pattern in descriptive_patterns:
            if re.search(pattern, prompt.lower()):
                specificity_count += 1
        
        return min(1.0, specificity_count / 6.0)
    
    def _enhance_prompt(self, prompt: str, style: str = "realistic") -> str:
        """
        Enhance prompt using professional techniques.
        
        Args:
            prompt: Original prompt
            style: Target style
            
        Returns:
            Enhanced prompt
        """
        # Clean and normalize prompt
        cleaned_prompt = self._clean_prompt(prompt)
        
        # Identify prompt type and apply appropriate template
        prompt_type = self._classify_prompt_type(cleaned_prompt)
        
        # Extract key elements
        elements = self._extract_prompt_elements(cleaned_prompt)
        
        # Apply template structure
        if prompt_type in self.prompt_templates:
            template = self.prompt_templates[prompt_type]
            enhanced = self._apply_template(template, elements, style)
        else:
            # Generic enhancement
            enhanced = self._generic_enhancement(cleaned_prompt, style)
        
        # Add quality boosters
        enhanced = self._add_quality_boosters(enhanced, style)
        
        # No token limiting - WAN supports longer prompts than SD/SVD
        # enhanced = self._limit_tokens(enhanced, max_tokens=77)  # Disabled for WAN
        
        return enhanced
    
    def _clean_prompt(self, prompt: str) -> str:
        """Clean and normalize prompt."""
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', prompt.strip())
        
        # Remove problematic characters
        cleaned = re.sub(r'[^\w\s,.-]', '', cleaned)
        
        # Normalize commas
        cleaned = re.sub(r',\s*,', ',', cleaned)
        
        return cleaned
    
    def _classify_prompt_type(self, prompt: str) -> str:
        """Classify prompt type (character, scene, object)."""
        prompt_lower = prompt.lower()
        
        # Character indicators
        character_words = ['person', 'man', 'woman', 'boy', 'girl', 'child', 'character', 'face', 'portrait']
        if any(word in prompt_lower for word in character_words):
            return "character"
        
        # Scene indicators
        scene_words = ['scene', 'landscape', 'background', 'room', 'forest', 'city', 'street']
        if any(word in prompt_lower for word in scene_words):
            return "scene"
        
        # Default to object
        return "object"
    
    def _extract_prompt_elements(self, prompt: str) -> Dict[str, str]:
        """Extract key elements from prompt."""
        elements = {
            "subject": "",
            "appearance": "",
            "clothing": "",
            "pose": "",
            "expression": "",
            "lighting": "",
            "style": "",
            "quality": ""
        }
        
        # Simple extraction based on keywords
        prompt_lower = prompt.lower()
        
        # Extract subject (first noun phrase)
        words = prompt.split()
        for i, word in enumerate(words):
            if word.lower() not in ['a', 'an', 'the', 'with', 'in', 'on', 'at']:
                elements["subject"] = word
                break
        
        # Extract other elements based on patterns
        if 'wearing' in prompt_lower:
            clothing_match = re.search(r'wearing\s+([^,]+)', prompt_lower)
            if clothing_match:
                elements["clothing"] = clothing_match.group(1)
        
        if 'standing' in prompt_lower or 'sitting' in prompt_lower or 'lying' in prompt_lower:
            pose_match = re.search(r'(standing|sitting|lying)', prompt_lower)
            if pose_match:
                elements["pose"] = pose_match.group(1)
        
        if 'smiling' in prompt_lower or 'frowning' in prompt_lower or 'serious' in prompt_lower:
            expr_match = re.search(r'(smiling|frowning|serious)', prompt_lower)
            if expr_match:
                elements["expression"] = expr_match.group(1)
        
        return elements
    
    def _apply_template(self, template: Dict, elements: Dict[str, str], style: str) -> str:
        """Apply template structure to elements."""
        structure = template["structure"]
        
        # Fill in available elements
        for key, value in elements.items():
            if value:
                structure = structure.replace(f"{{{key}}}", value)
        
        # Remove empty placeholders
        structure = re.sub(r'\{[^}]+\}', '', structure)
        
        # Clean up extra commas
        structure = re.sub(r',\s*,', ',', structure)
        structure = structure.strip(',').strip()
        
        return structure
    
    def _generic_enhancement(self, prompt: str, style: str) -> str:
        """Generic prompt enhancement when template doesn't apply."""
        enhanced = prompt
        
        # Add style-specific enhancements
        if style == "realistic":
            enhanced += ", photorealistic, detailed textures, natural lighting"
        elif style == "anime":
            enhanced += ", anime style, detailed, clean art"
        else:
            enhanced += ", high quality, detailed, professional"
        
        return enhanced
    
    def _add_quality_boosters(self, prompt: str, style: str) -> str:
        """Add quality boosters based on style."""
        if style in self.quality_boosters:
            boosters = self.quality_boosters[style]
            # Add 2-3 quality boosters
            selected_boosters = boosters[:3]
            prompt += ", " + ", ".join(selected_boosters)
        
        return prompt
    
    def _generate_negative_prompt(self, prompt: str, style: str) -> str:
        """Generate optimized negative prompt."""
        base_negative = "blurry, low quality, distorted, deformed, ugly, bad anatomy"
        
        # Style-specific negative prompts
        style_negatives = {
            "realistic": "cartoon, anime, manga, illustration, painting, drawing, sketch",
            "anime": "realistic, photorealistic, 3d render, cgi, western cartoon",
            "default": "cartoon, anime, manga, illustration, painting, drawing, sketch"
        }
        
        if style in style_negatives:
            base_negative += ", " + style_negatives[style]
        
        # Add content-specific negatives based on prompt
        if 'person' in prompt.lower() or 'face' in prompt.lower():
            base_negative += ", extra limbs, missing limbs, mutated hands, mutated feet"
        
        if 'scene' in prompt.lower() or 'background' in prompt.lower():
            base_negative += ", empty, barren, plain"
        
        return base_negative
    
    def _limit_tokens(self, prompt: str, max_tokens: int = 77) -> str:
        """
        Limit prompt to maximum token count (DEPRECATED - disabled for WAN).
        
        WAN supports longer prompts than SD/SVD, so token limiting is no longer applied.
        This method now returns the prompt unchanged.
        """
        # Token limiting disabled for WAN - it supports longer prompts than SD/SVD
        return prompt
    
    def enhance_multiple_prompts(self, prompts: List[str], style: str = "realistic") -> List[Tuple[str, str]]:
        """
        Enhance multiple prompts with analysis.
        
        Args:
            prompts: List of original prompts
            style: Target style
            
        Returns:
            List of (enhanced_prompt, negative_prompt) tuples
        """
        results = []
        
        for i, prompt in enumerate(prompts):
            logger.info(f"🎯 Analyzing prompt {i+1}/{len(prompts)}")
            
            analysis = self.analyze_prompt(prompt, style)
            
            if analysis.issues:
                logger.warning(f"⚠️ Issues found in prompt {i+1}: {', '.join(analysis.issues)}")
                logger.info(f"💡 Suggestions: {', '.join(analysis.suggestions)}")
            
            results.append((analysis.enhanced_prompt, analysis.optimized_negative_prompt))
        
        return results
    
    def validate_prompt(self, prompt: str, style: str = "realistic") -> bool:
        """
        Validate if a prompt meets quality standards.
        
        Args:
            prompt: Prompt to validate
            style: Target style
            
        Returns:
            True if prompt meets standards
        """
        analysis = self.analyze_prompt(prompt, style)
        
        # Check if scores meet minimum thresholds
        if analysis.clarity_score < 0.6:
            return False
        if analysis.structure_score < 0.5:
            return False
        if analysis.specificity_score < 0.4:
            return False
        
        return True


# Backward compatibility
class PromptEnhancer(ProfessionalPromptEnhancer):
    """Legacy class for backward compatibility."""
    
    def enhance_prompt(self, original_prompt: str, max_length: int = 100, 
                      enhancement_type: str = "diffusion", max_tokens: int = 77) -> str:
        """Legacy method for backward compatibility."""
        analysis = self.analyze_prompt(original_prompt, style="realistic")
        return analysis.enhanced_prompt


def test_prompt_enhancement():
    """Test function for prompt enhancement."""
    enhancer = ProfessionalPromptEnhancer()
    
    test_prompts = [
        "a sunset over mountains",
        "a cat playing in a garden",
        "a magical forest with glowing mushrooms",
        "Sardar Patel sitting at wooden desk, maps of India spread out, white dhoti kurta, round spectacles, 1940s study room, warm sepia lighting, cinematic shadows, historical drama style"
    ]
    
    print("Testing Professional Prompt Enhancement:")
    print("=" * 60)
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n📝 Test {i+1}: {prompt}")
        analysis = enhancer.analyze_prompt(prompt, style="realistic")
        
        print(f"📊 Scores - Clarity: {analysis.clarity_score:.2f}, Structure: {analysis.structure_score:.2f}, Specificity: {analysis.specificity_score:.2f}")
        
        if analysis.issues:
            print(f"⚠️ Issues: {', '.join(analysis.issues)}")
            print(f"💡 Suggestions: {', '.join(analysis.suggestions)}")
        
        print(f"🎯 Enhanced: {analysis.enhanced_prompt}")
        print(f"🚫 Negative: {analysis.optimized_negative_prompt}")
        print("-" * 60)


if __name__ == "__main__":
    test_prompt_enhancement()
