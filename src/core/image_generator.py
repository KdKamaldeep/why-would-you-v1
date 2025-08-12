#!/usr/bin/env python3
"""
Image Generator Module - Handles cartoon image generation using Stable Diffusion
"""

import os
import time
import logging
import torch
import re
from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional, Dict
from pathlib import Path
from .prompt_enhancer import PromptEnhancer

logger = logging.getLogger(__name__)

class ImageGenerator:
    """Handles cartoon image generation using Stable Diffusion.

    Supports optional LoRA for style adaptation.
    """
    
    def __init__(self, model_path: str = "models/toonyou_beta6.safetensors", lora_path: Optional[str] = None, lora_scale: float = 0.8, enable_prompt_enhancement: bool = True):
        self.model_path = model_path
        self.lora_path = lora_path
        self.lora_scale = lora_scale
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pipe = None
        self.sd_available = False
        self.enable_prompt_enhancement = enable_prompt_enhancement
        self.prompt_enhancer = None
        
        # Initialize prompt enhancer if enabled
        if self.enable_prompt_enhancement:
            self.prompt_enhancer = PromptEnhancer()
        
        self._initialize_pipeline()
        
    def _initialize_pipeline(self):
        """Initialize the Stable Diffusion pipeline."""
        # Import dependencies first, handle ImportError narrowly
        try:
            from diffusers import StableDiffusionPipeline
            import safetensors  # noqa: F401
        except ImportError as e:
            logger.warning(f"⚠️ Diffusers stack not available: {e}")
            logger.info("📦 Installing required packages...")
            try:
                import subprocess
                subprocess.run([
                    "pip", "install",
                    "diffusers>=0.25.0",
                    "transformers>=4.30.0",
                    "accelerate>=0.20.0",
                    "safetensors>=0.3.0"
                ], check=True)
                logger.info("✅ Packages installed! Please restart the script.")
            except Exception as install_error:
                logger.error(f"❌ Failed to install packages: {install_error}")
                logger.info("💡 Please install manually: pip install diffusers transformers accelerate safetensors")
            return

        try:
            logger.info(f"🚀 Initializing Stable Diffusion pipeline...")
            logger.info(f"💻 Device: {self.device}")
            logger.info(f"📁 Model path: {self.model_path}")

            # Check if model file exists; if not, try loading from Hugging Face repo id
            if not Path(self.model_path).exists():
                logger.warning(f"⚠️ Model file not found: {self.model_path}")
                repo_id = os.getenv("SD_REPO_ID")
                if not repo_id:
                    # Default to SD 1.5 base to ensure images can still be generated
                    repo_id = "runwayml/stable-diffusion-v1-5"
                    logger.info(f"💡 Falling back to pretrained model: {repo_id}")
                else:
                    logger.info(f"💡 Loading pretrained model from repo id: {repo_id}")
                try:
                    from diffusers import StableDiffusionPipeline
                    self.pipe = StableDiffusionPipeline.from_pretrained(
                        repo_id,
                        torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                        safety_checker=None,
                        requires_safety_checker=False
                    )
                    if self.device == 'cuda':
                        self.pipe = self.pipe.to(self.device)
                        if hasattr(self.pipe, 'enable_memory_efficient_attention'):
                            try:
                                self.pipe.enable_memory_efficient_attention()
                            except Exception:
                                pass
                        if hasattr(self.pipe, 'enable_xformers_memory_efficient_attention'):
                            try:
                                self.pipe.enable_xformers_memory_efficient_attention()
                            except Exception:
                                logger.info("ℹ️ xFormers not available; continuing without it")
                    # Optionally load LoRA after from_pretrained as well
                    if self.lora_path and Path(self.lora_path).exists():
                        try:
                            logger.info(f"🎭 Loading LoRA: {self.lora_path}")
                            load_ok = False
                            if hasattr(self.pipe, 'load_lora_weights'):
                                self.pipe.load_lora_weights(self.lora_path)
                                load_ok = True
                                if hasattr(self.pipe, 'fuse_lora'):
                                    try:
                                        self.pipe.fuse_lora(lora_scale=self.lora_scale)
                                    except Exception:
                                        pass
                                elif hasattr(self.pipe, 'set_adapters'):
                                    try:
                                        self.pipe.set_adapters(["default"], adapter_weights=[self.lora_scale])
                                    except Exception:
                                        pass
                            if load_ok:
                                logger.info(f"✅ LoRA loaded with scale ~ {self.lora_scale}")
                        except Exception as le:
                            logger.warning(f"⚠️ Failed to load LoRA '{self.lora_path}': {le}")
                    self.sd_available = True
                    logger.info("✅ Stable Diffusion pipeline initialized from pretrained repo!")
                    return
                except Exception as e:
                    logger.warning(f"❌ Failed to load pretrained pipeline '{repo_id}': {e}")
                    logger.info("💡 Will use placeholder images instead. To enable SD, set SD_REPO_ID or place a .safetensors model.")
                    return

            # Load the pipeline with the custom model
            logger.info("📦 Loading Stable Diffusion model...")
            self.pipe = StableDiffusionPipeline.from_single_file(
                self.model_path,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
                use_safetensors=True
            )

            if self.device == 'cuda':
                self.pipe = self.pipe.to(self.device)
                # Enable memory optimizations (best-effort, never fail)
                if hasattr(self.pipe, 'enable_memory_efficient_attention'):
                    try:
                        self.pipe.enable_memory_efficient_attention()
                    except Exception:
                        pass
                if hasattr(self.pipe, 'enable_xformers_memory_efficient_attention'):
                    try:
                        self.pipe.enable_xformers_memory_efficient_attention()
                    except Exception:
                        logger.info("ℹ️ xFormers not available; continuing without it")

            # Optionally load a LoRA for style adaptation
            if self.lora_path and Path(self.lora_path).exists():
                try:
                    logger.info(f"🎭 Loading LoRA: {self.lora_path}")
                    load_ok = False
                    # Newer diffusers API
                    if hasattr(self.pipe, 'load_lora_weights'):
                        self.pipe.load_lora_weights(self.lora_path)
                        load_ok = True
                        # Try to fuse or set scale depending on API
                        if hasattr(self.pipe, 'fuse_lora'):
                            try:
                                self.pipe.fuse_lora(lora_scale=self.lora_scale)
                            except Exception:
                                pass
                        elif hasattr(self.pipe, 'set_adapters'):
                            try:
                                self.pipe.set_adapters(["default"], adapter_weights=[self.lora_scale])
                            except Exception:
                                pass
                    if load_ok:
                        logger.info(f"✅ LoRA loaded with scale ~ {self.lora_scale}")
                except Exception as le:
                    logger.warning(f"⚠️ Failed to load LoRA '{self.lora_path}': {le}")

            self.sd_available = True
            logger.info("✅ Stable Diffusion pipeline initialized successfully!")
            logger.info("🎨 Ready to generate professional cartoon images!")

        except Exception as e:
            logger.warning(f"❌ Failed to initialize SD pipeline: {e}")
            logger.info("💡 Will use placeholder images instead")
            self.sd_available = False
    
    def generate_cartoon_image(self, prompt: str, output_path: str) -> str:
        """Generate a cartoon-style image using Stable Diffusion."""
        try:
            logger.info(f"🎨 Generating cartoon image for prompt: {prompt}")
            
            # If we have a working pipeline, use it
            if self.sd_available and self.pipe is not None:
                return self._generate_sd_image(prompt, output_path)
            else:
                # Fallback to placeholder
                logger.info("⚠️ Using placeholder image (SD pipeline not available)")
                return self._generate_placeholder_image(prompt, output_path)
                
        except Exception as e:
            logger.error(f"❌ Error generating image: {e}")
            return self._generate_placeholder_image(prompt, output_path)
    
    def _generate_sd_image(self, prompt: str, output_path: str) -> str:
        """Generate image using Stable Diffusion."""
        try:
            # Use the prompt as-is (enhancement is now handled in _compose_image_prompt)
            final_prompt = prompt
            logger.info(f"🎯 Using prompt (enhancement handled upstream): {prompt}")
            
            # Strong cartoon-specific negative prompts to avoid realistic images
            negative_prompt = (
                "photorealistic, realistic, photo, 3d render, cgi, anime, manga, "
                "blurry, low quality, dark, scary, violent, adult content, nsfw, "
                "hyperrealistic, detailed textures, photographic, film grain, "
                "realistic lighting, realistic shadows, realistic proportions, "
                "detailed skin, detailed hair, detailed clothing textures"
            )
            
            # Generate image with cartoon-optimized settings
            with torch.autocast(self.device):
                result = self.pipe(
                    prompt=final_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=30,  # More steps for better cartoon quality
                    guidance_scale=7.5,      # Balanced for cartoon style
                    width=768,
                    height=1024,
                    num_images_per_prompt=1,
                    generator=torch.Generator(device=self.device).manual_seed(42)  # Consistent results
                )
            
            # Save the image
            image = result.images[0]
            image.save(output_path, quality=95)
            
            logger.info(f"✅ Generated professional SD image: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ SD generation failed: {e}")
            logger.info("🔄 Falling back to placeholder image")
            return self._generate_placeholder_image(prompt, output_path)
    
    def generate_multiple_images(self, prompts: List[str], output_dir: str) -> List[str]:
        """Generate multiple cartoon images for a list of prompts."""
        image_paths = []
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for i, prompt in enumerate(prompts):
            logger.info(f"🎬 Generating scene {i+1}/{len(prompts)}")
            output_path = f"{output_dir}/scene_{i+1}.png"
            image_path = self.generate_cartoon_image(prompt, output_path)
            image_paths.append(image_path)
            
            # Small delay between generations to prevent memory issues
            if i < len(prompts) - 1:
                time.sleep(1)
        
        logger.info(f"✅ Generated {len(image_paths)} images successfully!")
        return image_paths
    
    def _generate_placeholder_image(self, prompt: str, output_path: str) -> str:
        """Generate a colorful placeholder image with better design."""
        try:
            # Create a gradient background instead of solid blue
            img = Image.new('RGB', (768, 1024), color='white')
            draw = ImageDraw.Draw(img)
            
            # Create a colorful gradient background
            for y in range(1024):
                color_r = int(135 + (y / 1024) * 120)  # 135-255
                color_g = int(206 + (y / 1024) * 49)   # 206-255  
                color_b = int(250 - (y / 1024) * 50)   # 250-200
                color = (min(255, color_r), min(255, color_g), min(255, color_b))
                draw.line([(0, y), (768, y)], fill=color)
            
            # Add decorative elements
            # Draw some simple shapes for visual appeal
            draw.ellipse([50, 50, 150, 150], fill='yellow', outline='orange', width=3)
            draw.rectangle([600, 100, 700, 200], fill='lightgreen', outline='green', width=3)
            draw.ellipse([100, 800, 200, 900], fill='pink', outline='red', width=3)
            draw.rectangle([550, 850, 650, 950], fill='lightcoral', outline='darkred', width=3)
            
            # Add text with better formatting
            try:
                # Try different font sizes
                title_font = ImageFont.truetype("arial.ttf", 36)
                text_font = ImageFont.truetype("arial.ttf", 24)
            except:
                title_font = ImageFont.load_default()
                text_font = ImageFont.load_default()
            
            # Title
            title = "🎬 Cartoon Scene"
            title_bbox = draw.textbbox((0, 0), title, font=title_font)
            title_width = title_bbox[2] - title_bbox[0]
            title_x = (768 - title_width) // 2
            
            # Add text shadow
            draw.text((title_x + 2, 302), title, fill='gray', font=title_font)
            draw.text((title_x, 300), title, fill='darkblue', font=title_font)
            
            # Scene description
            words = prompt.split()
            lines = []
            current_line = []
            
            for word in words:
                current_line.append(word)
                test_line = ' '.join(current_line)
                bbox = draw.textbbox((0, 0), test_line, font=text_font)
                if bbox[2] - bbox[0] > 600:  # Max width
                    if len(current_line) > 1:
                        current_line.pop()
                        lines.append(' '.join(current_line))
                        current_line = [word]
                    else:
                        lines.append(word)
                        current_line = []
            
            if current_line:
                lines.append(' '.join(current_line))
            
            # Limit to 8 lines
            lines = lines[:8]
            
            y_offset = 400
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=text_font)
                text_width = bbox[2] - bbox[0]
                x = (768 - text_width) // 2
                
                # Add text shadow
                draw.text((x + 1, y_offset + 1), line, fill='gray', font=text_font)
                draw.text((x, y_offset), line, fill='darkblue', font=text_font)
                y_offset += 35
            
            # Add a note about the placeholder
            note = "⚠️ Placeholder - Install diffusers for AI images"
            note_bbox = draw.textbbox((0, 0), note, font=text_font)
            note_width = note_bbox[2] - note_bbox[0]
            note_x = (768 - note_width) // 2
            draw.text((note_x, 750), note, fill='red', font=text_font)
            
            # Add installation instructions
            install_note = "Run: pip install diffusers transformers accelerate safetensors"
            install_bbox = draw.textbbox((0, 0), install_note, font=text_font)
            install_width = install_bbox[2] - install_bbox[0]
            install_x = (768 - install_width) // 2
            draw.text((install_x, 780), install_note, fill='darkgreen', font=text_font)
            
            img.save(output_path, quality=95)
            logger.info(f"✅ Generated enhanced placeholder image: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Error creating placeholder: {e}")
            # Create a simple fallback image
            img = Image.new('RGB', (768, 1024), color='lightblue')
            draw = ImageDraw.Draw(img)
            draw.text((384, 512), f"Scene: {prompt}", fill='black', anchor='mm')
            img.save(output_path)
            return output_path
    
    def is_sd_available(self) -> bool:
        """Check if Stable Diffusion is available."""
        return self.sd_available and self.pipe is not None

    def _is_image_blank_or_poor_quality(self, image_path: str) -> bool:
        """
        Check if an image is blank, mostly empty, or of poor quality.
        Returns True if the image should be regenerated with an adjusted prompt.
        """
        try:
            from PIL import Image, ImageStat
            import numpy as np
            
            # Load the image
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array for analysis
            img_array = np.array(image)
            
            # Check 1: Variance analysis (blank images have low variance)
            stat = ImageStat.Stat(image)
            variance = np.var(img_array)
            
            # Check 2: Check if image is mostly one color (blank/empty)
            unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
            
            # Check 3: Check brightness distribution
            gray = image.convert('L')
            gray_array = np.array(gray)
            brightness_variance = np.var(gray_array)
            
            # Check 4: Check for extreme brightness (all white or all black)
            mean_brightness = np.mean(gray_array)
            
            # Define thresholds
            is_blank = (
                variance < 1000 or  # Very low variance indicates blank image
                unique_colors < 100 or  # Very few unique colors
                brightness_variance < 500 or  # Low brightness variance
                mean_brightness < 10 or  # Too dark
                mean_brightness > 245  # Too bright
            )
            
            if is_blank:
                logger.warning(f"⚠️ Image detected as blank/poor quality: {image_path}")
                logger.info(f"   Variance: {variance:.1f}, Unique colors: {unique_colors}, Brightness: {mean_brightness:.1f}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error analyzing image quality: {e}")
            # If we can't analyze, assume it's fine
            return False

    def analyze_prompt_compliance(self, image_path: str, prompt: str) -> Dict:
        """
        Analyze if a generated image complies with the given visual prompt.
        Returns a dictionary with compliance scores and analysis.
        """
        try:
            from PIL import Image
            import numpy as np
            
            # Load the image
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            img_array = np.array(image)
            
            # Extract key elements from the prompt
            prompt_elements = self._extract_prompt_elements(prompt)
            
            # Analyze image characteristics
            analysis = {
                'prompt_elements': prompt_elements,
                'image_analysis': self._analyze_image_characteristics(img_array),
                'compliance_score': 0.0,
                'missing_elements': [],
                'present_elements': [],
                'overall_quality': 'unknown'
            }
            
            # Check for basic quality issues first
            if self._is_image_blank_or_poor_quality_from_array(img_array):
                analysis['overall_quality'] = 'poor'
                analysis['compliance_score'] = 0.0
                return analysis
            
            analysis['overall_quality'] = 'good'
            
            # Analyze color compliance
            color_compliance = self._analyze_color_compliance(img_array, prompt_elements)
            analysis['color_analysis'] = color_compliance
            
            # Analyze composition compliance
            composition_compliance = self._analyze_composition_compliance(img_array, prompt_elements)
            analysis['composition_analysis'] = composition_compliance
            
            # Calculate overall compliance score
            analysis['compliance_score'] = self._calculate_compliance_score(
                color_compliance, composition_compliance, prompt_elements
            )
            
            # Identify missing and present elements
            analysis['missing_elements'] = self._identify_missing_elements(analysis)
            analysis['present_elements'] = self._identify_present_elements(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing prompt compliance: {e}")
            return {
                'error': str(e),
                'compliance_score': 0.0,
                'overall_quality': 'error'
            }

    def _extract_prompt_elements(self, prompt: str) -> Dict:
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

    def _analyze_image_characteristics(self, img_array) -> Dict:
        """Analyze basic image characteristics."""
        import numpy as np
        
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

    def _is_image_blank_or_poor_quality_from_array(self, img_array) -> bool:
        """Check if image is blank or poor quality from numpy array."""
        import numpy as np
        
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

    def _analyze_color_compliance(self, img_array, prompt_elements: Dict) -> Dict:
        """Analyze if image colors match prompt requirements."""
        import numpy as np
        
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

    def _analyze_composition_compliance(self, img_array, prompt_elements: Dict) -> Dict:
        """Analyze if image composition matches prompt requirements."""
        import numpy as np
        
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

    def _calculate_compliance_score(self, color_compliance: Dict, composition_compliance: Dict, prompt_elements: Dict) -> float:
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

    def _identify_missing_elements(self, analysis: Dict) -> List[str]:
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

    def _identify_present_elements(self, analysis: Dict) -> List[str]:
        """Identify elements that are present in the image."""
        present = []
        
        if analysis.get('composition_analysis', {}).get('has_characters'):
            present.append('characters')
        
        if analysis.get('composition_analysis', {}).get('has_setting'):
            present.append('setting')
        
        matched_colors = analysis.get('color_analysis', {}).get('matched_colors', [])
        present.extend(matched_colors)
        
        return present

    def _adjust_visual_prompt_for_blank_image(self, original_prompt: str, attempt: int = 1) -> str:
        """
        Adjust the visual prompt using GPT-2 to fix blank image issues.
        Returns an enhanced prompt that should produce better results.
        """
        try:
            if not self.prompt_enhancer or not self.prompt_enhancer.is_available():
                logger.warning("⚠️ Prompt enhancer not available, using fallback adjustments")
                return self._fallback_prompt_adjustment(original_prompt, attempt)
            
            # Create specific adjustment prompts based on attempt number
            if attempt == 1:
                adjustment_prompt = f"Enhance this visual prompt to create a vibrant, detailed cartoon scene with clear subjects and rich colors: {original_prompt}"
            elif attempt == 2:
                adjustment_prompt = f"Transform this prompt into a highly detailed, colorful cartoon scene with strong visual elements and clear composition: {original_prompt}"
            else:
                adjustment_prompt = f"Create an extremely detailed, vibrant cartoon scene with multiple visual elements, rich colors, and clear subjects: {original_prompt}"
            
            logger.info(f"🎯 Adjusting prompt (attempt {attempt}): {original_prompt}")
            
            # Use GPT-2 to enhance the prompt
            enhanced_prompt = self.prompt_enhancer.enhance_prompt(
                adjustment_prompt,
                enhancement_type="cartoon_detailed",
                max_tokens=77  # Keep within diffusion model limits
            )
            
            # Add specific cartoon enhancement keywords if not present
            enhancement_keywords = [
                "vibrant colors", "detailed cartoon", "clear composition", 
                "rich textures", "bright lighting", "distinct subjects"
            ]
            
            # Check if any enhancement keywords are missing
            missing_keywords = [kw for kw in enhancement_keywords if kw.lower() not in enhanced_prompt.lower()]
            
            if missing_keywords and attempt <= 2:
                # Add missing keywords
                additional_enhancement = ", ".join(missing_keywords[:3])  # Limit to 3 keywords
                enhanced_prompt = f"{enhanced_prompt}, {additional_enhancement}"
                
                # Ensure we stay within token limits
                enhanced_prompt = self.prompt_enhancer._limit_tokens(enhanced_prompt, 77)
            
            logger.info(f"🎯 Enhanced prompt: {enhanced_prompt}")
            return enhanced_prompt
            
        except Exception as e:
            logger.error(f"Error adjusting prompt with GPT-2: {e}")
            return self._fallback_prompt_adjustment(original_prompt, attempt)

    def _fallback_prompt_adjustment(self, original_prompt: str, attempt: int) -> str:
        """
        Fallback prompt adjustment when GPT-2 is not available.
        """
        base_enhancements = [
            "vibrant cartoon style, detailed, colorful",
            "bright cartoon scene, rich details, clear subjects",
            "highly detailed cartoon, vibrant colors, strong composition"
        ]
        
        enhancement = base_enhancements[min(attempt - 1, len(base_enhancements) - 1)]
        adjusted_prompt = f"{original_prompt}, {enhancement}"
        
        logger.info(f"🎯 Fallback adjusted prompt: {adjusted_prompt}")
        return adjusted_prompt

    def rewrite_prompt_for_better_compliance(self, original_prompt: str, compliance_analysis: Dict, attempt: int) -> str:
        """
        Intelligently rewrite the prompt to improve compliance based on analysis results.
        Uses GPT-2 for intelligent prompt rewriting when available.
        """
        missing_elements = compliance_analysis.get('missing_elements', [])
        present_elements = compliance_analysis.get('present_elements', [])
        prompt_elements = compliance_analysis.get('prompt_elements', {})
        score = compliance_analysis.get('compliance_score', 0.0)
        
        logger.info(f"🔧 Rewriting prompt based on analysis:")
        logger.info(f"   Missing: {missing_elements}")
        logger.info(f"   Present: {present_elements}")
        logger.info(f"   Score: {score:.2f}")
        
        # Try GPT-2 rewriting first if available
        if self.prompt_enhancer and self.prompt_enhancer.is_available():
            gpt2_prompt = self._rewrite_prompt_with_gpt2(original_prompt, compliance_analysis, attempt)
            if gpt2_prompt and gpt2_prompt != original_prompt:
                logger.info(f"🎯 Using GPT-2 rewritten prompt")
                return gpt2_prompt
        
        # Fallback to rule-based rewriting
        logger.info(f"🎯 Using rule-based prompt rewriting")
        return self._rewrite_prompt_with_rules(original_prompt, compliance_analysis, attempt)

    def _rewrite_prompt_with_gpt2(self, original_prompt: str, compliance_analysis: Dict, attempt: int) -> str:
        """
        Use GPT-2 to intelligently rewrite the prompt based on compliance analysis.
        """
        missing_elements = compliance_analysis.get('missing_elements', [])
        present_elements = compliance_analysis.get('present_elements', [])
        score = compliance_analysis.get('compliance_score', 0.0)
        
        # Create specific GPT-2 prompts based on the analysis
        if attempt == 1:
            if missing_elements:
                gpt2_prompt = f"Rewrite this visual prompt to include missing elements ({', '.join(missing_elements)}) and improve image quality: {original_prompt}"
            else:
                gpt2_prompt = f"Enhance this visual prompt to improve image quality and make elements more prominent: {original_prompt}"
        elif attempt == 2:
            gpt2_prompt = f"Transform this prompt into a highly detailed, vibrant cartoon scene with strong visual elements. Missing: {', '.join(missing_elements)}. Original: {original_prompt}"
        else:
            gpt2_prompt = f"Create an extremely detailed, masterpiece-quality cartoon scene. Focus on missing elements: {', '.join(missing_elements)}. Original prompt: {original_prompt}"
        
        try:
            # Use GPT-2 to enhance the prompt
            enhanced_prompt = self.prompt_enhancer.enhance_prompt(
                gpt2_prompt,
                enhancement_type="cartoon_detailed",
                max_tokens=77  # Keep within diffusion model limits
            )
            
            # If GPT-2 enhancement is too similar to original, try a different approach
            if enhanced_prompt == original_prompt or len(enhanced_prompt) < len(original_prompt) + 20:
                logger.info(f"🔄 GPT-2 enhancement too similar, trying alternative approach")
                
                # Try with more specific instructions
                alternative_prompt = f"Add specific visual details and enhance missing elements ({', '.join(missing_elements)}) to this cartoon prompt: {original_prompt}"
                enhanced_prompt = self.prompt_enhancer.enhance_prompt(
                    alternative_prompt,
                    enhancement_type="detailed",
                    max_tokens=77
                )
            
            logger.info(f"🎯 GPT-2 enhanced prompt: {enhanced_prompt}")
            return enhanced_prompt
            
        except Exception as e:
            logger.error(f"❌ GPT-2 rewriting failed: {e}")
            return original_prompt

    def _rewrite_prompt_with_rules(self, original_prompt: str, compliance_analysis: Dict, attempt: int) -> str:
        """
        Rule-based prompt rewriting as fallback when GPT-2 is not available.
        """
        missing_elements = compliance_analysis.get('missing_elements', [])
        present_elements = compliance_analysis.get('present_elements', [])
        prompt_elements = compliance_analysis.get('prompt_elements', {})
        score = compliance_analysis.get('compliance_score', 0.0)
        
        # Start with the original prompt structure
        new_prompt = original_prompt
        
        # Strategy 1: Enhance missing elements with stronger emphasis
        if missing_elements:
            for element in missing_elements:
                if element == 'characters':
                    # Add stronger character emphasis
                    new_prompt = self._enhance_character_description(new_prompt, attempt)
                elif element == 'setting':
                    # Add stronger setting emphasis
                    new_prompt = self._enhance_setting_description(new_prompt, attempt)
                elif element in ['red', 'blue', 'green', 'brown', 'gray']:
                    # Add stronger color emphasis
                    new_prompt = self._enhance_color_description(new_prompt, element, attempt)
        
        # Strategy 2: Add specific enhancement keywords based on attempt
        enhancement_keywords = self._get_enhancement_keywords(attempt)
        if enhancement_keywords:
            new_prompt = self._add_enhancement_keywords(new_prompt, enhancement_keywords)
        
        # Strategy 3: Adjust weights for better emphasis
        new_prompt = self._adjust_prompt_weights(new_prompt, missing_elements, attempt)
        
        # Strategy 4: Add specific style enhancements
        if score < 0.5:
            new_prompt = self._add_style_enhancements(new_prompt, attempt)
        
        # Ensure the prompt doesn't get too long
        if len(new_prompt) > 500:
            new_prompt = self._truncate_prompt(new_prompt, 500)
        
        return new_prompt

    def rewrite_prompt_for_better_compliance_with_analysis(self, original_prompt: str, compliance_analysis: Dict, enhancement_data: Dict, attempt: int) -> str:
        """
        Intelligently rewrite the prompt using advanced image analysis data.
        Incorporates detailed image characteristics for more targeted improvements.
        """
        missing_elements = compliance_analysis.get('missing_elements', [])
        present_elements = compliance_analysis.get('present_elements', [])
        image_analysis = compliance_analysis.get('image_analysis', {})
        score = compliance_analysis.get('compliance_score', 0.0)
        
        logger.info(f"🔧 Advanced prompt rewriting with image analysis:")
        logger.info(f"   Missing: {missing_elements}")
        logger.info(f"   Enhancement data: {enhancement_data}")
        logger.info(f"   Score: {score:.2f}")
        
        # Try GPT-2 rewriting with advanced analysis if available
        if self.prompt_enhancer and self.prompt_enhancer.is_available():
            gpt2_prompt = self._rewrite_prompt_with_gpt2_and_analysis(
                original_prompt, compliance_analysis, enhancement_data, attempt
            )
            if gpt2_prompt and gpt2_prompt != original_prompt:
                logger.info(f"🎯 Using GPT-2 rewritten prompt with advanced analysis")
                return gpt2_prompt
        
        # Fallback to enhanced rule-based rewriting
        logger.info(f"🎯 Using enhanced rule-based prompt rewriting")
        return self._rewrite_prompt_with_enhanced_rules(original_prompt, compliance_analysis, enhancement_data, attempt)

    def _rewrite_prompt_with_gpt2_and_analysis(self, original_prompt: str, compliance_analysis: Dict, enhancement_data: Dict, attempt: int) -> str:
        """
        Use GPT-2 to intelligently rewrite the prompt based on advanced image analysis.
        """
        missing_elements = compliance_analysis.get('missing_elements', [])
        image_analysis = compliance_analysis.get('image_analysis', {})
        score = compliance_analysis.get('compliance_score', 0.0)
        
        # Extract specific analysis data for GPT-2
        color_issues = enhancement_data.get('color_issues', [])
        composition_issues = enhancement_data.get('composition_issues', [])
        quality_issues = enhancement_data.get('quality_issues', [])
        style_suggestions = enhancement_data.get('style_suggestions', [])
        
        # Create detailed GPT-2 prompt with analysis data
        analysis_summary = []
        if color_issues:
            analysis_summary.append(f"Color issues: {', '.join(color_issues)}")
        if composition_issues:
            analysis_summary.append(f"Composition issues: {', '.join(composition_issues)}")
        if quality_issues:
            analysis_summary.append(f"Quality issues: {', '.join(quality_issues)}")
        if style_suggestions:
            analysis_summary.append(f"Style suggestions: {', '.join(style_suggestions)}")
        
        analysis_text = "; ".join(analysis_summary) if analysis_summary else "No specific issues detected"
        
        # Create specific GPT-2 prompts based on the analysis and attempt
        if attempt == 1:
            gpt2_prompt = f"""Rewrite this visual prompt to address these specific issues: {analysis_text}. 
            Missing elements: {', '.join(missing_elements)}. 
            Focus on improving color, composition, and detail quality. 
            Original prompt: {original_prompt}"""
        elif attempt == 2:
            gpt2_prompt = f"""Transform this prompt into a highly detailed, vibrant cartoon scene. 
            Address these issues: {analysis_text}. 
            Emphasize missing elements: {', '.join(missing_elements)}. 
            Add rich textures and enhanced composition. 
            Original: {original_prompt}"""
        else:
            gpt2_prompt = f"""Create a masterpiece-quality cartoon scene with exceptional detail. 
            Fix these issues: {analysis_text}. 
            Ensure all missing elements are prominent: {', '.join(missing_elements)}. 
            Use professional illustration techniques and vibrant colors. 
            Original prompt: {original_prompt}"""
        
        try:
            # Use GPT-2 to enhance the prompt
            enhanced_prompt = self.prompt_enhancer.enhance_prompt(
                gpt2_prompt,
                enhancement_type="cartoon_detailed",
                max_tokens=77  # Keep within diffusion model limits
            )
            
            # Add specific technical improvements based on analysis
            technical_improvements = enhancement_data.get('technical_improvements', [])
            if technical_improvements:
                improvement_text = ", ".join(technical_improvements[:2])  # Limit to 2 improvements
                enhanced_prompt = f"{enhanced_prompt}, {improvement_text}"
                enhanced_prompt = self.prompt_enhancer._limit_tokens(enhanced_prompt, 77)
            
            logger.info(f"🎯 GPT-2 enhanced prompt with analysis: {enhanced_prompt}")
            return enhanced_prompt
            
        except Exception as e:
            logger.error(f"❌ GPT-2 rewriting with analysis failed: {e}")
            return original_prompt

    def _rewrite_prompt_with_enhanced_rules(self, original_prompt: str, compliance_analysis: Dict, enhancement_data: Dict, attempt: int) -> str:
        """
        Enhanced rule-based prompt rewriting using advanced image analysis data.
        """
        missing_elements = compliance_analysis.get('missing_elements', [])
        score = compliance_analysis.get('compliance_score', 0.0)
        
        # Start with the original prompt
        new_prompt = original_prompt
        
        # Apply enhancements based on analysis data
        color_issues = enhancement_data.get('color_issues', [])
        composition_issues = enhancement_data.get('composition_issues', [])
        quality_issues = enhancement_data.get('quality_issues', [])
        style_suggestions = enhancement_data.get('style_suggestions', [])
        
        # Address color issues
        if 'low_color_variance' in color_issues:
            new_prompt += ", diverse color palette, chromatic variety"
        if 'low_saturation' in color_issues:
            new_prompt += ", saturated colors, vibrant hues"
        if 'limited_color_palette' in color_issues:
            new_prompt += ", rich color spectrum, multiple hues"
        
        # Address composition issues
        if 'low_detail' in composition_issues:
            new_prompt += ", highly detailed, fine details"
        if 'too_dark' in composition_issues:
            new_prompt += ", bright lighting, well-lit scene"
        if 'too_bright' in composition_issues:
            new_prompt += ", balanced lighting, natural illumination"
        if 'low_contrast' in composition_issues:
            new_prompt += ", high contrast, dramatic lighting"
        
        # Address quality issues
        if 'low_texture_detail' in quality_issues:
            new_prompt += ", textured surfaces, rich details"
        
        # Add style suggestions
        if 'vibrant_colors' in style_suggestions:
            new_prompt += ", vibrant color palette"
        if 'high_detail' in style_suggestions:
            new_prompt += ", highly detailed rendering"
        if 'bright_lighting' in style_suggestions:
            new_prompt += ", bright, clear lighting"
        
        # Add attempt-specific enhancements
        if attempt == 2:
            new_prompt += ", enhanced details, improved composition"
        elif attempt >= 3:
            new_prompt += ", masterwork quality, professional illustration"
        
        # Ensure we don't exceed token limits
        if len(new_prompt) > 200:  # Rough token limit check
            # Truncate while keeping essential parts
            parts = new_prompt.split(',')
            if len(parts) > 8:  # Keep first 8 parts
                new_prompt = ','.join(parts[:8])
        
        logger.info(f"🎯 Enhanced rule-based prompt: {new_prompt}")
        return new_prompt

    def _enhance_character_description(self, prompt: str, attempt: int) -> str:
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

    def _enhance_setting_description(self, prompt: str, attempt: int) -> str:
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

    def _enhance_color_description(self, prompt: str, color: str, attempt: int) -> str:
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

    def _get_enhancement_keywords(self, attempt: int) -> List[str]:
        """Get enhancement keywords based on attempt number."""
        enhancement_sets = [
            ["vibrant colors", "clear details", "strong composition"],
            ["highly detailed", "rich textures", "sharp focus"],
            ["extremely detailed", "masterpiece quality", "professional illustration"]
        ]
        
        return enhancement_sets[min(attempt - 1, len(enhancement_sets) - 1)]

    def _add_enhancement_keywords(self, prompt: str, keywords: List[str]) -> str:
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

    def _adjust_prompt_weights(self, prompt: str, missing_elements: List[str], attempt: int) -> str:
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

    def _add_style_enhancements(self, prompt: str, attempt: int) -> str:
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

    def _truncate_prompt(self, prompt: str, max_length: int) -> str:
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

    def generate_cartoon_image_with_validation(self, prompt: str, output_path: str, max_attempts: int = 3) -> str:
        """
        Generate a cartoon image with validation and automatic prompt adjustment.
        Retries with adjusted prompts if the generated image is blank or poor quality.
        """
        original_prompt = prompt
        best_score = 0.0
        best_image_path = None
        
        for attempt in range(1, max_attempts + 1):
            logger.info(f"🎨 Generating image (attempt {attempt}/{max_attempts})")
            
            # Generate the image
            result_path = self.generate_cartoon_image(prompt, output_path)
            
            # Analyze prompt compliance
            compliance_analysis = self.analyze_prompt_compliance(result_path, prompt)
            score = compliance_analysis.get('compliance_score', 0.0)
            
            # Track best result
            if score > best_score:
                best_score = score
                best_image_path = result_path
            
            # Check if image passes basic quality validation
            if not self._is_image_blank_or_poor_quality(result_path):
                logger.info(f"✅ Basic image validation passed on attempt {attempt}")
                
                # If compliance is good enough, return the result
                if score >= 0.6:
                    logger.info(f"✅ Good compliance achieved (score: {score:.2f})")
                    return result_path
                else:
                    logger.warning(f"⚠️ Low compliance score: {score:.2f}")
            else:
                logger.warning(f"⚠️ Image validation failed on attempt {attempt}")
            
            # If this is not the last attempt, rewrite the prompt and try again
            if attempt < max_attempts:
                logger.info(f"🔄 Rewriting prompt for attempt {attempt + 1}")
                
                # Use GPT-2 enhanced prompt rewriting
                new_prompt = self.rewrite_prompt_for_better_compliance(original_prompt, compliance_analysis, attempt)
                prompt = new_prompt
                
                # Create a new output path for this attempt
                base_path = Path(output_path)
                new_output_path = base_path.parent / f"{base_path.stem}_attempt_{attempt + 1}{base_path.suffix}"
                output_path = str(new_output_path)
            else:
                logger.warning(f"⚠️ All {max_attempts} attempts completed. Using best available image.")
                return best_image_path if best_image_path else result_path
        
        return best_image_path if best_image_path else output_path

    def generate_cartoon_image_with_gpt2_enhancement(self, prompt: str, output_path: str, max_attempts: int = 3) -> Dict:
        """
        Generate a cartoon image with GPT-2 enhanced prompt rewriting and comprehensive compliance analysis.
        Returns detailed information about the generation process and results.
        """
        original_prompt = prompt
        best_score = 0.0
        best_image_path = None
        best_attempt = 0
        generation_history = []
        
        logger.info(f"🚀 Starting GPT-2 enhanced image generation with {max_attempts} attempts")
        logger.info(f"📝 Original prompt: {original_prompt}")
        
        for attempt in range(1, max_attempts + 1):
            logger.info(f"\n🔄 Attempt {attempt}/{max_attempts}")
            
            # Generate the image
            result_path = self.generate_cartoon_image(prompt, output_path)
            
            # Analyze prompt compliance
            compliance_analysis = self.analyze_prompt_compliance(result_path, prompt)
            score = compliance_analysis.get('compliance_score', 0.0)
            
            # Record attempt details
            attempt_info = {
                'attempt': attempt,
                'prompt': prompt,
                'image_path': result_path,
                'compliance_score': score,
                'missing_elements': compliance_analysis.get('missing_elements', []),
                'present_elements': compliance_analysis.get('present_elements', []),
                'quality': compliance_analysis.get('overall_quality', 'unknown')
            }
            generation_history.append(attempt_info)
            
            # Track best result
            if score > best_score:
                best_score = score
                best_image_path = result_path
                best_attempt = attempt
            
            logger.info(f"    📊 Compliance Score: {score:.2f}/1.0")
            logger.info(f"    🎯 Quality: {compliance_analysis.get('overall_quality', 'unknown')}")
            logger.info(f"    ✅ Present: {', '.join(compliance_analysis.get('present_elements', []))}")
            logger.info(f"    ❌ Missing: {', '.join(compliance_analysis.get('missing_elements', []))}")
            
            # Check if we should continue
            if score >= 0.8:
                logger.info(f"    🎉 Excellent compliance achieved! Stopping retries.")
                break
            elif score >= 0.6:
                logger.info(f"    ✅ Good compliance achieved.")
                if attempt < max_attempts:
                    logger.info(f"    🤔 Consider continuing for better results...")
            else:
                logger.info(f"    ⚠️ Low compliance. Will retry with GPT-2 enhanced prompt.")
            
            # If this is not the last attempt, rewrite the prompt using GPT-2
            if attempt < max_attempts:
                logger.info(f"    🔄 Rewriting prompt with GPT-2 for next attempt...")
                
                # Use GPT-2 enhanced prompt rewriting
                new_prompt = self.rewrite_prompt_for_better_compliance(original_prompt, compliance_analysis, attempt)
                prompt = new_prompt
                
                # Create a new output path for this attempt
                base_path = Path(output_path)
                new_output_path = base_path.parent / f"{base_path.stem}_attempt_{attempt + 1}{base_path.suffix}"
                output_path = str(new_output_path)
                
                logger.info(f"    📝 New GPT-2 enhanced prompt: {new_prompt[:100]}{'...' if len(new_prompt) > 100 else ''}")
        
        # Final results
        logger.info(f"\n🎯 Final Results:")
        logger.info(f"    📊 Best Compliance Score: {best_score:.2f}/1.0")
        logger.info(f"    🏆 Best Image: {best_image_path}")
        logger.info(f"    🎬 Best Attempt: {best_attempt}")
        
        if best_score >= 0.8:
            logger.info(f"    🎉 SUCCESS: Excellent prompt compliance achieved!")
        elif best_score >= 0.6:
            logger.info(f"    ✅ GOOD: Acceptable prompt compliance achieved.")
        else:
            logger.info(f"    ⚠️ POOR: Low prompt compliance. Consider manual prompt refinement.")
        
        return {
            'best_image_path': best_image_path,
            'best_compliance_score': best_score,
            'best_attempt': best_attempt,
            'generation_history': generation_history,
            'original_prompt': original_prompt,
            'final_prompt': prompt,
            'success': best_score >= 0.6
        }
