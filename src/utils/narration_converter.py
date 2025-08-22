#!/usr/bin/env python3
"""
Narration to SSML Converter

This module converts narration text with inline cues to SSML format
for use with Coqui TTS voice synthesis.
"""

import re
import json
import logging
import os
import glob
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class NarrationConverter:
    """Converts narration text with inline cues to SSML format"""
    
    # Mapping of cues to SSML tags
    CUE_MAPPINGS = {
        # Pitch and rate combinations
        r'\[nervous\]': '<prosody pitch="+3st" rate="95%">',
        r'\[excited\]': '<prosody pitch="+2st" rate="105%">',
        r'\[slow, wise\]': '<prosody pitch="-3st" rate="88%">',
        r'\[low pitch, confident\]': '<prosody pitch="-2st" rate="92%">',
        r'\[high pitch\]': '<prosody pitch="+3st" rate="95%">',
        r'\[low pitch\]': '<prosody pitch="-2st" rate="92%">',
        r'\[fast\]': '<prosody rate="110%">',
        r'\[slow\]': '<prosody rate="85%">',
        r'\[whisper\]': '<prosody pitch="-1st" rate="90%" volume="-20%">',
        r'\[loud\]': '<prosody pitch="+1st" rate="100%" volume="+20%">',
        
        # Emphasis
        r'\[emphasis strong\]': '<emphasis level="strong">',
        r'\[emphasis moderate\]': '<emphasis level="moderate">',
        r'\[emphasis reduced\]': '<emphasis level="reduced">',
        
        # Pauses
        r'\[pause\]': '<break time="600ms"/>',
        r'\[long pause\]': '<break time="1000ms"/>',
        r'\[short pause\]': '<break time="300ms"/>',
        
        # Character-specific cues
        r'\[old\]': '<prosody pitch="-2st" rate="85%">',
        r'\[young\]': '<prosody pitch="+2st" rate="110%">',
        r'\[baby\]': '<prosody pitch="+4st" rate="120%">',
        r'\[giant\]': '<prosody pitch="-3st" rate="80%">',
        r'\[robot\]': '<prosody pitch="-1st" rate="90%" volume="-10%">',
        r'\[magical\]': '<prosody pitch="+1st" rate="95%">',
        r'\[evil\]': '<prosody pitch="-2st" rate="85%">',
        r'\[heroic\]': '<prosody pitch="+1st" rate="100%" volume="+10%">',
    }
    
    # Closing tags for prosody and emphasis
    CLOSING_TAGS = {
        'prosody': '</prosody>',
        'emphasis': '</emphasis>'
    }
    
    def __init__(self, voice_dir: str = "tts-speaker"):
        """
        Initialize the narration converter
        
        Args:
            voice_dir: Directory containing voice files
        """
        self.voice_dir = voice_dir
        self.compiled_patterns = {
            pattern: re.compile(pattern, re.IGNORECASE) 
            for pattern in self.CUE_MAPPINGS.keys()
        }
        self._voice_files_cache = None
    
    def _get_voice_files(self) -> Dict[str, str]:
        """Get a mapping of voice names to file paths"""
        if self._voice_files_cache is not None:
            return self._voice_files_cache
        
        voice_files = {}
        
        if os.path.exists(self.voice_dir):
            # Look for audio files in the voice directory, prioritizing WAV files
            # Process WAV files first, then other formats
            audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a']
            
            for ext in audio_extensions:
                pattern = os.path.join(self.voice_dir, ext)
                for file_path in glob.glob(pattern):
                    file_name = os.path.basename(file_path)
                    name_without_ext = os.path.splitext(file_name)[0]
                    
                    # Only store if we don't already have a WAV version of this voice
                    if name_without_ext not in voice_files or voice_files[name_without_ext].lower().endswith('.wav'):
                        voice_files[name_without_ext] = file_path
                    
                    # Also store without the last part (e.g., "hi-IN-SwaraNeural-female" for "hi-IN-SwaraNeural-cheerful-female")
                    parts = name_without_ext.split('-')
                    if len(parts) >= 4:
                        base_voice = '-'.join(parts[:-1])  # Remove the last part
                        # Only store if we don't already have a WAV version of this base voice
                        if base_voice not in voice_files or voice_files[base_voice].lower().endswith('.wav'):
                            voice_files[base_voice] = file_path
        
        self._voice_files_cache = voice_files
        return voice_files
    
    def resolve_voice_file(self, voice_name: str) -> Optional[str]:
        """
        Resolve a voice name to an actual file path
        
        Args:
            voice_name: The voice name from the scene (e.g., "hi-IN-SwaraNeural-female")
            
        Returns:
            Path to the voice file if found, None otherwise
        """
        if not voice_name:
            return None
        
        voice_files = self._get_voice_files()
        
        # Try exact match first
        if voice_name in voice_files:
            return voice_files[voice_name]
        
        # Try with common extensions, prioritizing WAV files
        for ext in ['.wav', '.mp3', '.flac', '.m4a']:
            full_name = voice_name + ext
            if full_name in voice_files:
                return voice_files[full_name]
        
        # Try partial matches (for cases like "hi-IN-SwaraNeural-female" matching "hi-IN-SwaraNeural-cheerful-female")
        # Prioritize WAV files over MP3 files
        candidates = []
        for file_name, file_path in voice_files.items():
            if voice_name in file_name or file_name.startswith(voice_name):
                # Score candidates: WAV files get higher priority
                score = 0
                if file_path.lower().endswith('.wav'):
                    score = 2
                elif file_path.lower().endswith('.mp3'):
                    score = 1
                candidates.append((score, file_path))
        
        if candidates:
            # Sort by score (highest first) and return the best match
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]
        
        logger.warning(f"Voice file not found for voice name: {voice_name}")
        return None
    
    def convert_narration(self, narration_text: str, use_ssml: bool = True) -> str:
        """
        Convert narration text with inline cues to SSML format or plain text
        
        Args:
            narration_text: The narration text containing inline cues in square brackets
            use_ssml: If True, convert to SSML. If False, remove cues and return plain text
            
        Returns:
            SSML formatted text wrapped in <speak> tags, or plain text with cues removed
        """
        if not narration_text or not isinstance(narration_text, str):
            return narration_text
        
        if not use_ssml:
            # Remove all cues and return plain text
            converted_text = narration_text
            for pattern in self.CUE_MAPPINGS.keys():
                compiled_pattern = re.compile(pattern, re.IGNORECASE)
                converted_text = compiled_pattern.sub('', converted_text)
            # Clean up extra whitespace
            converted_text = re.sub(r'\s+', ' ', converted_text).strip()
            return converted_text
        
        # Start with the original text
        converted_text = narration_text
        
        # Track open tags to properly close them
        open_tags = []
        
        # Process each cue pattern
        for pattern, ssml_tag in self.CUE_MAPPINGS.items():
            compiled_pattern = self.compiled_patterns[pattern]
            
            def replace_cue(match):
                nonlocal converted_text, open_tags
                
                # Determine if this is an opening or closing tag
                if ssml_tag.startswith('<prosody') or ssml_tag.startswith('<emphasis'):
                    # Opening tag - track it
                    tag_type = 'prosody' if ssml_tag.startswith('<prosody') else 'emphasis'
                    open_tags.append(tag_type)
                    return ssml_tag
                else:
                    # Self-closing tag (like break)
                    return ssml_tag
            
            # Replace all occurrences of this cue
            converted_text = compiled_pattern.sub(replace_cue, converted_text)
        
        # Close any remaining open tags in reverse order
        for tag_type in reversed(open_tags):
            if tag_type in self.CLOSING_TAGS:
                converted_text += self.CLOSING_TAGS[tag_type]
        
        # Wrap in speak tags
        return f"<speak>{converted_text}</speak>"
    
    def convert_storyboard(self, storyboard_data: Dict[str, Any], use_ssml: bool = True) -> Dict[str, Any]:
        """
        Convert all narration fields in a storyboard to SSML format or plain text
        
        Args:
            storyboard_data: The storyboard JSON data
            use_ssml: If True, convert to SSML. If False, remove cues and return plain text
            
        Returns:
            Updated storyboard data with SSML-formatted or plain text narration
        """
        if not isinstance(storyboard_data, dict):
            return storyboard_data
        
        # Create a copy to avoid modifying the original
        converted_data = storyboard_data.copy()
        
        # Process scenes if they exist
        if 'scenes' in converted_data and isinstance(converted_data['scenes'], list):
            for scene in converted_data['scenes']:
                if isinstance(scene, dict):
                    # Convert narration if present
                    if 'narration' in scene:
                        scene['narration'] = self.convert_narration(scene['narration'], use_ssml)
                    
                    # Resolve voice file if voice property is present
                    if 'voice' in scene:
                        voice_file = self.resolve_voice_file(scene['voice'])
                        if voice_file:
                            scene['voice_file'] = voice_file
                            logger.info(f"Resolved voice '{scene['voice']}' to file: {voice_file}")
                        else:
                            logger.warning(f"Could not resolve voice file for: {scene['voice']}")
        
        return converted_data
    
    def convert_storyboard_file(self, input_file: str, output_file: Optional[str] = None, use_ssml: bool = True) -> str:
        """
        Convert a storyboard JSON file to SSML format or plain text
        
        Args:
            input_file: Path to the input JSON file
            output_file: Path to the output JSON file (optional)
            use_ssml: If True, convert to SSML. If False, remove cues and return plain text
            
        Returns:
            Path to the output file
        """
        try:
            # Read the input file
            with open(input_file, 'r', encoding='utf-8') as f:
                storyboard_data = json.load(f)
            
            # Convert the narration
            converted_data = self.convert_storyboard(storyboard_data, use_ssml)
            
            # Determine output file path
            if output_file is None:
                base_name = input_file.rsplit('.', 1)[0]
                suffix = "_ssml" if use_ssml else "_plain"
                output_file = f"{base_name}{suffix}.json"
            
            # Write the converted data
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(converted_data, f, ensure_ascii=False, indent=2)
            
            format_type = "SSML" if use_ssml else "plain text"
            logger.info(f"Converted {input_file} to {output_file} ({format_type})")
            return output_file
            
        except Exception as e:
            logger.error(f"Error converting storyboard file {input_file}: {e}")
            raise
    
    def batch_convert_storyboards(self, input_directory: str, output_directory: Optional[str] = None, use_ssml: bool = True) -> List[str]:
        """
        Convert all JSON storyboard files in a directory
        
        Args:
            input_directory: Directory containing storyboard JSON files
            output_directory: Directory to save converted files (optional)
            use_ssml: If True, convert to SSML. If False, remove cues and return plain text
            
        Returns:
            List of output file paths
        """
        import os
        import glob
        
        if output_directory is None:
            output_directory = input_directory
        
        # Create output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)
        
        # Find all JSON files
        json_files = glob.glob(os.path.join(input_directory, "*.json"))
        
        converted_files = []
        for json_file in json_files:
            try:
                # Generate output filename
                base_name = os.path.basename(json_file).rsplit('.', 1)[0]
                suffix = "_ssml" if use_ssml else "_plain"
                output_file = os.path.join(output_directory, f"{base_name}{suffix}.json")
                
                # Convert the file
                self.convert_storyboard_file(json_file, output_file, use_ssml)
                converted_files.append(output_file)
                
            except Exception as e:
                logger.error(f"Error converting {json_file}: {e}")
                continue
        
        return converted_files
    
    def get_supported_cues(self) -> List[str]:
        """Get a list of all supported cues"""
        cues = []
        for pattern in self.CUE_MAPPINGS.keys():
            # Extract the cue text from the regex pattern
            cue = pattern.replace(r'\[', '').replace(r'\]', '').replace(r'\\', '')
            cues.append(cue)
        return sorted(cues)
    
    def get_available_voices(self) -> List[str]:
        """Get a list of all available voice names"""
        voice_files = self._get_voice_files()
        return sorted(voice_files.keys())


def main():
    """Command line interface for the narration converter"""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Convert narration text to SSML format or plain text")
    parser.add_argument("input", help="Input JSON file or directory")
    parser.add_argument("-o", "--output", help="Output file or directory")
    parser.add_argument("--batch", action="store_true", help="Process all JSON files in directory")
    parser.add_argument("--plain", action="store_true", help="Convert to plain text (remove cues) instead of SSML")
    parser.add_argument("--list-cues", action="store_true", help="List all supported cues and exit")
    parser.add_argument("--list-voices", action="store_true", help="List all available voices and exit")
    parser.add_argument("--voice-dir", default="tts-speaker", help="Directory containing voice files")
    
    args = parser.parse_args()
    
    converter = NarrationConverter(voice_dir=args.voice_dir)
    
    # List supported cues if requested
    if args.list_cues:
        print("Supported narration cues:")
        print("=" * 30)
        for cue in converter.get_supported_cues():
            print(f"  [{cue}]")
        return
    
    # List available voices if requested
    if args.list_voices:
        print("Available voices:")
        print("=" * 20)
        for voice in converter.get_available_voices():
            voice_file = converter.resolve_voice_file(voice)
            print(f"  {voice} -> {voice_file}")
        return
    
    try:
        use_ssml = not args.plain
        format_type = "SSML" if use_ssml else "plain text"
        
        if args.batch:
            # Batch mode
            converted_files = converter.batch_convert_storyboards(args.input, args.output, use_ssml)
            print(f"Converted {len(converted_files)} files to {format_type}")
            for file_path in converted_files:
                print(f"  {file_path}")
        else:
            # Single file mode
            output_file = converter.convert_storyboard_file(args.input, args.output, use_ssml)
            print(f"Converted to {format_type}: {output_file}")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
