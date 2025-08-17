# TTS Fixes Documentation

## Overview

This document describes the fixes implemented for the Coqui TTS system to address:
1. **GPT2InferenceModel GenerationMixin warnings** from transformers v4.50+
2. **Hindi language support improvements** for better multilingual TTS

## Issues Addressed

### 1. GPT2InferenceModel Warning

**Problem**: Starting from transformers v4.50, models that don't directly inherit from `GenerationMixin` show warnings about losing `generate` capabilities.

**Warning Message**:
```
GPT2InferenceModel has generative capabilities, as `prepare_inputs_for_generation` is explicitly overwritten. 
However, it doesn't directly inherit from `GenerationMixin`. From v4.50 onwards, `PreTrainedModel` will NOT 
inherit from `GenerationMixin`, and this model will lose the ability to call `generate` and other related functions.
```

**Solution**: Added comprehensive warning suppression in `src/core/coqui_voice_synthesizer.py`:

```python
# Suppress GPT2InferenceModel GenerationMixin warnings from transformers v4.50+
warnings.filterwarnings("ignore", message=".*GPT2InferenceModel has generative capabilities.*")
warnings.filterwarnings("ignore", message=".*doesn't directly inherit from GenerationMixin.*")
warnings.filterwarnings("ignore", message=".*PreTrainedModel will NOT inherit from GenerationMixin.*")
warnings.filterwarnings("ignore", message=".*this model will lose the ability to call generate.*")
warnings.filterwarnings("ignore", message=".*modify your model class such that it inherits from GenerationMixin.*")
```

### 2. Hindi Language Support

**Problem**: Hindi language TTS had issues with:
- Model selection for Hindi
- Speaker voice discovery
- Text encoding for Devanagari script
- Synthesis strategy failures

**Solutions Implemented**:

#### A. Enhanced Model Loading
- **Language-specific prioritization**: Hindi now prioritizes XTTS v2 models
- **Language verification**: Check if loaded model supports target language
- **Robust fallback**: Multiple model options for different scenarios

```python
if lang == "hi":  # Hindi
    fallback_models.extend([
        "coqui/XTTS-v2",
        "tts_models/multilingual/multi-dataset/xtts_v2",
        "tts_models/multilingual/multi-dataset/your_tts",
    ])
```

#### B. Improved Speaker Discovery
- **Hindi-specific patterns**: Look for files like `hindi_speaker.wav`, `male_hindi_speaker.wav`
- **Environment variables**: Support `HINDI_SPEAKER_WAV` environment variable
- **Multiple search directories**: Check `tts-speaker`, `tts_speaker`, `tts_voices`, `voices`
- **Smart ranking**: Prioritize files with language codes and gender hints

#### C. Text Encoding Support
- **Unicode normalization**: Properly handle Devanagari script
- **Text validation**: Ensure Hindi text is correctly encoded

```python
if self.config.language == "hi":
    import unicodedata
    full_text = unicodedata.normalize('NFC', full_text)
```

#### D. Robust Synthesis Strategies
- **Multiple fallback strategies**: Try speaker_wav, speaker token, then default
- **Kernel size error handling**: Automatically extend short text to meet model requirements (minimum 50 chars)
- **Smart text padding**: Language-specific meaningful content addition
- **Text cleaning and deduplication**: Remove empty lines and duplicate text
- **Better error handling**: Detailed error messages for debugging
- **Language-aware logging**: Track synthesis progress for different languages

## Usage Examples

### English TTS
```python
from core.coqui_voice_synthesizer import CoquiVoiceSynthesizer, CoquiVoiceConfig

config = CoquiVoiceConfig(language="en")
synthesizer = CoquiVoiceSynthesizer(config)

text = ["Hello! This is a test of the English TTS system."]
output_path = synthesizer.synthesize_voice(text, "output.wav")
```

### Hindi TTS
```python
from core.coqui_voice_synthesizer import CoquiVoiceSynthesizer, CoquiVoiceConfig

config = CoquiVoiceConfig(language="hi")
synthesizer = CoquiVoiceSynthesizer(config)

text = ["नमस्ते! यह हिंदी भाषा में आवाज़ जनरेशन का परीक्षण है।"]
output_path = synthesizer.synthesize_voice(text, "output.wav")
```

### Custom Speaker Voice
```bash
# Set environment variable for Hindi speaker
export HINDI_SPEAKER_WAV="/path/to/hindi_voice.wav"

# Or place voice file in tts-speaker directory
# tts-speaker/male_hindi_speaker.wav
```

## Testing

Run the comprehensive test suite to verify fixes:

```bash
python test_tts_fixes.py
```

This will test:
- ✅ Warning suppression for GPT2InferenceModel
- ✅ Speaker discovery for different languages
- ✅ English TTS functionality
- ✅ Hindi TTS functionality

## Configuration Options

### Environment Variables
- `HINDI_SPEAKER_WAV`: Path to Hindi speaker voice file
- `ENGLISH_SPEAKER_WAV`: Path to English speaker voice file
- `SPANISH_SPEAKER_WAV`: Path to Spanish speaker voice file

### Model Configuration
```python
config = CoquiVoiceConfig(
    language="hi",           # Target language (en, hi, es, etc.)
    model_name="coqui/XTTS-v2",  # Preferred model
    gpu=True,               # Use GPU if available
    speaker="default",      # Default speaker
    voice_dir="tts_voices/" # Voice directory
)
```

## Troubleshooting

### Common Issues

1. **No Hindi speaker found**
   - Place a Hindi voice file in `tts-speaker/` directory
   - Use environment variable `HINDI_SPEAKER_WAV`
   - XTTS will use built-in speakers if no custom voice found

2. **Model loading fails**
   - Check internet connection for model download
   - Ensure sufficient disk space for model files
   - Try different model names in fallback list

3. **Synthesis fails**
   - Check text encoding for non-English languages
   - Verify model supports target language
   - Try different synthesis strategies

4. **Kernel size error**
   - Text is too short for the TTS model (minimum 50 characters required)
   - System automatically extends text with meaningful content
   - Language-specific padding (Hindi uses Devanagari script, English uses Latin)
   - Multiple fallback strategies for different model types

5. **Text duplication and empty lines**
   - Input text contains duplicate lines or empty strings
   - System automatically cleans and deduplicates text
   - Removes empty lines and duplicate content
   - Provides fallback text if no valid content remains

### Debug Mode
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Dependencies

Updated `requirements.txt` includes:
- `transformers==4.49.0` (compatible version)
- `TTS==0.22.0` (stable version)
- Additional audio processing libraries for better language support

## Future Improvements

1. **More Languages**: Extend support for additional languages
2. **Voice Cloning**: Improve voice cloning capabilities
3. **Quality Optimization**: Fine-tune synthesis parameters
4. **Batch Processing**: Support for multiple audio generation

## References

- [Coqui TTS Documentation](https://docs.coqui.ai/)
- [XTTS v2 Model](https://huggingface.co/coqui/XTTS-v2)
- [Transformers v4.50 Changes](https://github.com/huggingface/transformers/releases)
