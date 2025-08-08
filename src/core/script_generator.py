#!/usr/bin/env python3
"""
Script Generator Module - Handles story and script generation using OpenAI GPT-4
"""

import json
import logging
import requests
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class ScriptGenerator:
    """Handles script generation using OpenAI GPT-4."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1/chat/completions"
        
    def generate_script(self, prompt: str, duration: int) -> Dict:
        """Generate a 3-scene story script for the video."""
        scene_duration = max(8, duration // 3)  # Minimum 8 seconds per scene
        gpt_prompt = f"""
        Create a {duration}-second YouTube Shorts story based on this prompt: "{prompt}"
        
        Requirements:
        - Create exactly 3 scenes, each approximately {scene_duration} seconds long
        - Make it engaging and entertaining for social media
        - Include VERY detailed visual descriptions for accurate cartoon-style image generation
        - Add humor, emotion, and personality
        - Optimized for vertical video format (768x1024)
        - Include clear, engaging narration text for each scene
        - Make sure the visual descriptions match the story content EXACTLY
        - Include specific details about characters, expressions, actions, and settings
        
        Return the response as a JSON object with:
        {{
            "title": "Engaging story title",
            "description": "Brief description of the complete story",
            "total_duration": {duration},
            "scenes": [
                {{
                    "duration": {scene_duration},
                    "description": "Detailed description of what happens in this scene",
                    "visual_prompt": "VERY detailed cartoon-style description for Stable Diffusion: include character appearance, facial expression, pose, setting, colors, lighting, mood, and any objects or actions. Be specific about cartoon/animated style.",
                    "narration": "Clear, engaging text to be narrated by AI voice (2-3 sentences)",
                    "subtitle": "Concise subtitle text that matches the narration"
                }},
                {{
                    "duration": {scene_duration},
                    "description": "Detailed description of what happens in this scene", 
                    "visual_prompt": "VERY detailed cartoon-style description for Stable Diffusion: include character appearance, facial expression, pose, setting, colors, lighting, mood, and any objects or actions. Be specific about cartoon/animated style.",
                    "narration": "Clear, engaging text to be narrated by AI voice (2-3 sentences)",
                    "subtitle": "Concise subtitle text that matches the narration"
                }},
                {{
                    "duration": {scene_duration},
                    "description": "Detailed description of what happens in this scene",
                    "visual_prompt": "VERY detailed cartoon-style description for Stable Diffusion: include character appearance, facial expression, pose, setting, colors, lighting, mood, and any objects or actions. Be specific about cartoon/animated style.",
                    "narration": "Clear, engaging text to be narrated by AI voice (2-3 sentences)",
                    "subtitle": "Concise subtitle text that matches the narration"
                }}
            ],
            "tags": ["cartoon", "story", "fun"]
        }}
        
        IMPORTANT: 
        - Make the story creative, fun, and appropriate for all ages
        - Ensure visual descriptions are EXTREMELY detailed and accurate to the story
        - Each scene should clearly connect to the overall narrative
        - Use vivid, descriptive language for image generation
        - Include emotional expressions and dynamic poses for characters
        """
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": gpt_prompt}],
            "temperature": 0.8,
            "max_tokens": 2000
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            script = json.loads(result["choices"][0]["message"]["content"])
            logger.info(f"Generated script for prompt: {prompt}")
            return script
        except Exception as e:
            logger.error(f"Error generating script: {e}")
            # Fallback script
            return self._generate_fallback_script(prompt, duration)

    def generate_script_from_custom(
        self,
        title: str,
        description: str,
        scenes: List[Dict],
        default_scene_duration: int = 8
    ) -> Dict:
        """Build a script object from user-provided storyboard scenes.

        Each scene in `scenes` should contain at least a `visual_prompt` key and may optionally
        include `narration`, `subtitle`, and `duration`.
        """
        total_duration = 0
        normalized_scenes: List[Dict] = []
        for idx, scene in enumerate(scenes):
            visual_prompt = scene.get("visual_prompt") or scene.get("prompt") or scene.get("description")
            if not visual_prompt:
                # Skip invalid scene entries silently but log
                logger.warning(f"Custom scene #{idx+1} missing visual_prompt/description; skipping")
                continue
            duration = int(scene.get("duration", default_scene_duration))
            total_duration += duration
            normalized_scenes.append({
                "duration": duration,
                "description": scene.get("description", visual_prompt),
                "visual_prompt": visual_prompt,
                "narration": scene.get("narration", scene.get("subtitle", "")),
                "subtitle": scene.get("subtitle", scene.get("narration", "")) or f"Scene {idx+1}"
            })

        if not normalized_scenes:
            logger.warning("No valid custom scenes provided; falling back to default script generation")
            return self._generate_fallback_script(title, max(default_scene_duration*3, 24))

        script: Dict = {
            "title": title,
            "description": description,
            "total_duration": total_duration,
            "scenes": normalized_scenes,
            "tags": ["cartoon", "storybook", "adventure"]
        }
        logger.info(f"Built script from {len(normalized_scenes)} custom scenes (total {total_duration}s)")
        return script
    
    def _generate_fallback_script(self, prompt: str, duration: int) -> Dict:
        """Generate a simple fallback script if API fails."""
        scene_duration = duration // 3
        return {
            "title": f"Story: {prompt}",
            "description": f"A fun cartoon story about {prompt}",
            "scenes": [
                {
                    "duration": scene_duration,
                    "description": f"Scene 1: Introduction to {prompt}",
                    "visual_prompt": f"Cartoon illustration of {prompt}, colorful, fun, animated style, high quality",
                    "narration": f"Once upon a time, there was {prompt}. Let me tell you this amazing story!",
                    "subtitle": f"Story: {prompt}"
                },
                {
                    "duration": scene_duration,
                    "description": f"Scene 2: The adventure continues",
                    "visual_prompt": f"Cartoon scene showing {prompt} in action, vibrant colors, detailed",
                    "narration": f"The adventure continues as {prompt} faces exciting challenges!",
                    "subtitle": "The Adventure Continues"
                },
                {
                    "duration": scene_duration,
                    "description": f"Scene 3: Happy ending",
                    "visual_prompt": f"Cartoon happy ending scene with {prompt}, joyful, celebration, colorful",
                    "narration": f"And they all lived happily ever after! What an amazing story about {prompt}!",
                    "subtitle": "Happy Ending!"
                }
            ],
            "tags": [prompt, "cartoon", "story", "fun"]
        }
