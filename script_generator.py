#!/usr/bin/env python3
"""
Script Generator Module - Handles story and script generation using OpenAI GPT-4
"""

import json
import logging
import requests
from typing import Dict

logger = logging.getLogger(__name__)

class ScriptGenerator:
    """Handles script generation using OpenAI GPT-4."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1/chat/completions"
        
    def generate_script(self, prompt: str, duration: int) -> Dict:
        """Generate a 3-scene story script for the video."""
        gpt_prompt = f"""
        Create a {duration}-second YouTube Shorts story based on this prompt: "{prompt}"
        
        Requirements:
        - Create exactly 3 scenes, each {duration//3} seconds long
        - Make it engaging and entertaining for social media
        - Include detailed visual descriptions for cartoon-style image generation
        - Add humor and personality
        - Optimized for vertical video format (768x1024)
        - Include narration text for each scene
        
        Return the response as a JSON object with:
        {{
            "title": "Story title",
            "description": "Brief description",
            "scenes": [
                {{
                    "duration": {duration//3},
                    "description": "What happens in this scene",
                    "visual_prompt": "Detailed cartoon-style description for Stable Diffusion image generation",
                    "narration": "Text to be narrated by ElevenLabs",
                    "subtitle": "Text to display as subtitle"
                }},
                {{
                    "duration": {duration//3},
                    "description": "What happens in this scene", 
                    "visual_prompt": "Detailed cartoon-style description for Stable Diffusion image generation",
                    "narration": "Text to be narrated by ElevenLabs",
                    "subtitle": "Text to display as subtitle"
                }},
                {{
                    "duration": {duration//3},
                    "description": "What happens in this scene",
                    "visual_prompt": "Detailed cartoon-style description for Stable Diffusion image generation", 
                    "narration": "Text to be narrated by ElevenLabs",
                    "subtitle": "Text to display as subtitle"
                }}
            ],
            "tags": ["cartoon", "story", "fun"]
        }}
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
