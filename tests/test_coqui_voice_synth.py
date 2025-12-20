#!/usr/bin/env python3
"""
Comprehensive test script for Coqui Voice Synthesizer
Tests model loading, voice generation, and various configurations
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional

# Ensure 'src' is on the import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.coqui_voice_synthesizer import CoquiVoiceSynthesizer, CoquiVoiceConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_model_loading(language: str = "en", gpu: bool = True):
    """Test model loading with different configurations."""
    print(f"\n{'='*70}")
    print(f"Test 1: Model Loading (Language: {language}, GPU: {gpu})")
    print(f"{'='*70}")
    
    try:
        config = CoquiVoiceConfig(language=language, gpu=gpu)
        print(f"📋 Configuration:")
        print(f"   Model: {config.model_name}")
        print(f"   Language: {config.language}")
        print(f"   GPU: {config.gpu}")
        print(f"   Voice Dir: {config.voice_dir}")
        
        print(f"\n🚀 Initializing Coqui Voice Synthesizer...")
        synthesizer = CoquiVoiceSynthesizer(config)
        
        print(f"✅ Model loaded successfully!")
        print(f"   Loaded model: {synthesizer.config.model_name}")
        print(f"   Device: {synthesizer.tts.device if hasattr(synthesizer.tts, 'device') else 'unknown'}")
        
        if hasattr(synthesizer.tts, 'languages'):
            print(f"   Supported languages: {synthesizer.tts.languages}")
        
        return synthesizer
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_voice_generation(
    synthesizer: CoquiVoiceSynthesizer,
    text: str,
    output_path: str,
    language: str = "en",
    voice_clone_audio: Optional[str] = None
):
    """Test voice generation with given text."""
    print(f"\n{'='*70}")
    print(f"Test 2: Voice Generation (Language: {language})")
    print(f"{'='*70}")
    
    print(f"📝 Text to synthesize:")
    print(f"   {text[:100]}{'...' if len(text) > 100 else ''}")
    print(f"   Length: {len(text)} characters")
    
    if voice_clone_audio:
        print(f"🎤 Voice clone audio: {voice_clone_audio}")
    
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🎙️  Generating voice...")
        result_path = synthesizer.synthesize_voice(
            narration_lines=[text],
            output_path=str(output_file),
            voice_clone_audio=voice_clone_audio
        )
        
        if Path(result_path).exists():
            size_kb = Path(result_path).stat().st_size / 1024
            duration = get_audio_duration(result_path)
            print(f"✅ Audio generated successfully!")
            print(f"   Output: {result_path}")
            print(f"   Size: {size_kb:.1f} KB")
            if duration:
                print(f"   Duration: {duration:.2f} seconds")
            return result_path
        else:
            print(f"❌ Audio file was not created at: {result_path}")
            return None
            
    except Exception as e:
        print(f"❌ Voice generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_audio_duration(audio_path: str) -> Optional[float]:
    """Get audio duration in seconds."""
    try:
        import librosa
        duration = librosa.get_duration(path=audio_path)
        return duration
    except Exception:
        try:
            import soundfile as sf
            info = sf.info(audio_path)
            return info.duration
        except Exception:
            return None


def test_english():
    """Test English voice generation."""
    print(f"\n{'#'*70}")
    print(f"# ENGLISH VOICE GENERATION TEST")
    print(f"{'#'*70}")
    
    synthesizer = test_model_loading(language="en", gpu=True)
    if not synthesizer:
        return False
    
    text = (
        "Hello! This is a comprehensive test of the Coqui XTTS voice synthesizer. "
        "We are testing English voice generation with a longer sample text to verify "
        "that the audio output is working correctly. This test helps ensure that "
        "the model is properly loaded and can generate high-quality speech."
    )
    
    output_path = "test_output/test_english.wav"
    result = test_voice_generation(
        synthesizer=synthesizer,
        text=text,
        output_path=output_path,
        language="en"
    )
    
    return result is not None


def test_hindi():
    """Test Hindi voice generation."""
    print(f"\n{'#'*70}")
    print(f"# HINDI VOICE GENERATION TEST")
    print(f"{'#'*70}")
    
    synthesizer = test_model_loading(language="hi", gpu=True)
    if not synthesizer:
        return False
    
    text = (
        "नमस्ते! यह कोकी एक्सटीटीएस आवाज़ सिंथेसाइज़र का एक व्यापक परीक्षण है। "
        "हम हिंदी आवाज़ जनरेशन का परीक्षण कर रहे हैं एक लंबे नमूना पाठ के साथ "
        "यह सुनिश्चित करने के लिए कि ऑडियो आउटपुट सही ढंग से काम कर रहा है। "
        "यह परीक्षण यह सुनिश्चित करने में मदद करता है कि मॉडल ठीक से लोड हो गया है "
        "और उच्च गुणवत्ता वाला भाषण उत्पन्न कर सकता है।"
    )
    
    output_path = "test_output/test_hindi.wav"
    result = test_voice_generation(
        synthesizer=synthesizer,
        text=text,
        output_path=output_path,
        language="hi"
    )
    
    return result is not None


def test_voice_cloning(voice_file: str):
    """Test voice cloning with a reference audio file."""
    print(f"\n{'#'*70}")
    print(f"# VOICE CLONING TEST")
    print(f"{'#'*70}")
    
    voice_path = Path(voice_file)
    if not voice_path.exists():
        print(f"❌ Voice file not found: {voice_file}")
        return False
    
    synthesizer = test_model_loading(language="en", gpu=True)
    if not synthesizer:
        return False
    
    text = (
        "This is a test of voice cloning. The generated speech should sound "
        "similar to the reference audio file provided."
    )
    
    output_path = "test_output/test_voice_clone.wav"
    result = test_voice_generation(
        synthesizer=synthesizer,
        text=text,
        output_path=output_path,
        language="en",
        voice_clone_audio=str(voice_path)
    )
    
    return result is not None


def test_multilingual():
    """Test multiple languages."""
    print(f"\n{'#'*70}")
    print(f"# MULTILINGUAL TEST")
    print(f"{'#'*70}")
    
    languages = [
        ("en", "Hello! This is an English test."),
        ("hi", "नमस्ते! यह एक हिंदी परीक्षण है।"),
        ("es", "¡Hola! Esta es una prueba en español."),
        ("fr", "Bonjour! Ceci est un test en français."),
    ]
    
    results = []
    for lang_code, text in languages:
        print(f"\n--- Testing {lang_code.upper()} ---")
        synthesizer = test_model_loading(language=lang_code, gpu=True)
        if synthesizer:
            output_path = f"test_output/test_{lang_code}.wav"
            result = test_voice_generation(
                synthesizer=synthesizer,
                text=text,
                output_path=output_path,
                language=lang_code
            )
            results.append((lang_code, result is not None))
        else:
            results.append((lang_code, False))
    
    print(f"\n{'='*70}")
    print("Multilingual Test Results:")
    for lang_code, success in results:
        status = "✅" if success else "❌"
        print(f"   {status} {lang_code.upper()}: {'PASS' if success else 'FAIL'}")
    
    return all(success for _, success in results)


def test_workspace_cache():
    """Test workspace cache detection."""
    print(f"\n{'#'*70}")
    print(f"# WORKSPACE CACHE TEST")
    print(f"{'#'*70}")
    
    workspace_path = Path("/workspace/models/tts/XTTS-v2")
    local_path = Path("models/tts/XTTS-v2")
    
    print(f"📁 Checking model cache locations:")
    print(f"   Workspace: {workspace_path}")
    print(f"   Exists: {workspace_path.exists()}")
    if workspace_path.exists():
        print(f"   Is directory: {workspace_path.is_dir()}")
        files = list(workspace_path.glob("*"))
        print(f"   Files/dirs: {len(files)}")
    
    print(f"\n   Local: {local_path}")
    print(f"   Exists: {local_path.exists()}")
    if local_path.exists():
        print(f"   Is directory: {local_path.is_dir()}")
        files = list(local_path.glob("*"))
        print(f"   Files/dirs: {len(files)}")
    
    # Test loading with workspace preference
    synthesizer = test_model_loading(language="en", gpu=True)
    if synthesizer:
        print(f"\n✅ Model loaded from: {synthesizer.config.model_name}")
        return True
    return False


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test Coqui Voice Synthesizer")
    parser.add_argument("--test", choices=["all", "english", "hindi", "multilingual", "workspace", "clone"],
                       default="all", help="Which test to run")
    parser.add_argument("--voice-file", type=str, help="Voice file for cloning test")
    parser.add_argument("--language", type=str, default="en", help="Language code for custom test")
    parser.add_argument("--text", type=str, help="Custom text to synthesize")
    parser.add_argument("--gpu", action="store_true", default=True, help="Use GPU if available")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    
    args = parser.parse_args()
    
    print("="*70)
    print("COQUI VOICE SYNTHESIZER TEST SUITE")
    print("="*70)
    
    # Create output directory
    Path("test_output").mkdir(exist_ok=True)
    
    use_gpu = args.gpu and not args.cpu
    
    results = []
    
    if args.test == "all":
        results.append(("Workspace Cache", test_workspace_cache()))
        results.append(("English", test_english()))
        results.append(("Hindi", test_hindi()))
        if args.voice_file:
            results.append(("Voice Cloning", test_voice_cloning(args.voice_file)))
    elif args.test == "english":
        results.append(("English", test_english()))
    elif args.test == "hindi":
        results.append(("Hindi", test_hindi()))
    elif args.test == "multilingual":
        results.append(("Multilingual", test_multilingual()))
    elif args.test == "workspace":
        results.append(("Workspace Cache", test_workspace_cache()))
    elif args.test == "clone":
        if not args.voice_file:
            print("❌ --voice-file is required for clone test")
            return 1
        results.append(("Voice Cloning", test_voice_cloning(args.voice_file)))
    
    # Custom test
    if args.text:
        print(f"\n{'#'*70}")
        print(f"# CUSTOM TEST")
        print(f"{'#'*70}")
        synthesizer = test_model_loading(language=args.language, gpu=use_gpu)
        if synthesizer:
            output_path = f"test_output/test_custom_{args.language}.wav"
            result = test_voice_generation(
                synthesizer=synthesizer,
                text=args.text,
                output_path=output_path,
                language=args.language
            )
            results.append(("Custom", result is not None))
    
    # Print summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status}: {test_name}")
    
    all_passed = all(success for _, success in results)
    print(f"\n{'='*70}")
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print(f"{'='*70}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
