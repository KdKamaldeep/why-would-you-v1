#!/usr/bin/env python3
"""
Voice Generator Module - Handles narration generation using ElevenLabs
"""

import logging
import requests
from pydub import AudioSegment
from typing import List

logger = logging.getLogger(__name__)

class VoiceGenerator:
    """Handles audio generation using ElevenLabs."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.elevenlabs.io/v1"
        
    def generate_narration(self, text: str, voice_id: str, output_path: str) -> str:
        """Generate narration audio using ElevenLabs."""
        url = f"{self.base_url}/text-to-speech/{voice_id}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }
        
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Generated narration audio: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            return self._generate_silent_audio(output_path)
    
    def generate_narration_from_script(self, script: dict, voice_id: str, output_path: str) -> str:
        """Generate narration from script scenes."""
        narration_text = " ".join([scene['narration'] for scene in script['scenes']])
        return self.generate_narration(narration_text, voice_id, output_path)
    
    def generate_individual_narrations(self, script: dict, voice_id: str, output_dir: str) -> List[str]:
        """Generate individual narration files for each scene."""
        audio_paths = []
        for i, scene in enumerate(script['scenes']):
            output_path = f"{output_dir}/audio_{i+1}.mp3"
            audio_path = self.generate_narration(scene['narration'], voice_id, output_path)
            audio_paths.append(audio_path)
        return audio_paths
    
    def _generate_silent_audio(self, output_path: str) -> str:
        """Generate silent audio as fallback."""
        audio = AudioSegment.silent(duration=3000)  # 3 seconds
        audio.export(output_path, format="mp3")
        logger.info(f"Generated silent audio: {output_path}")
        return output_path
