# 🎬 Video Reel Generator - Professional Edition

A complete AI-powered system for generating platform-ready vertical Reels/Shorts videos optimized for YouTube and Instagram.

## 🚀 **Features**

- 🎬 **WAN 2.1 Text-to-Video** - Direct text-to-video generation (no image step)
- 📱 **Platform-Ready Output** - 1080×1920, H.264/AAC, 30fps for YouTube/Instagram
- 🎵 **Smart Audio Mixing** - Voice + optional background music
- 📝 **GPT-4 Story Generation** - Intelligent script creation
- 🎤 **Coqui TTS Voice Generation** - Local XTTS v2 model with multilingual support
- ⚡ **Unlimited Length** - Generate videos of any duration
- 🎯 **Professional Quality** - Hollywood-grade output
- 🌍 **Multilingual Support** - Hindi, English, and other languages

## 📁 **Project Structure**

```
WhyWouldYou-v1/
├── 🎬 src/
│   ├── core/                    # Core system modules
│   │   ├── generate_cartoon_short.py
│   │   ├── image_generator.py
│   │   ├── animation_generator.py
│   │   ├── svd_animator.py      # SVD motion animation
│   │   ├── coqui_voice_synthesizer.py
│   │   ├── script_generator.py
│   │   └── video_processor.py
│   │
│   ├── interfaces/              # User interfaces
│   │   ├── simple_cartoon_generator.py  # Main CLI interface
│   │   ├── batch_generate.py
│   │   └── quick_start.py
│   │
│   └── utils/                   # Utilities
│       ├── setup.py
│       └── download_models.sh
│
├── 📚 docs/                     # Documentation
│   ├── README.md
│   ├── USAGE_GUIDE.md
│   └── PROJECT_CLEANUP_SUMMARY.md
│
├── 🧪 tests/                    # Testing
│   └── test_image_generation.py
│
├── ⚙️ config.env               # Configuration
├── 📦 requirements.txt          # Dependencies
└── 🚀 main.py                  # Main entry point
```

## 🎬 **Video Formats & Platform-Ready Reels**

The system generates platform-ready vertical Reels/Shorts optimized for YouTube and Instagram.

### **Platform-Ready Reel Output (Default)**
- **Dimensions**: 1080×1920 pixels (vertical)
- **Frame Rate**: 30 fps
- **Codec**: H.264 (yuv420p) + AAC audio
- **Perfect for**: YouTube Shorts, Instagram Reels, TikTok
- **Output**: `outputs/final_reel.mp4` (upload-ready)

### **Vertical Modes**

**Pad Mode (Default - Safe)**
- No cropping, preserves full content
- Safe for faces and important elements
- Adds letterboxing if needed
- Usage: `--vertical-mode pad`

**Crop Mode (Fills Frame)**
- Crops to fill vertical frame
- More "native" vertical look
- May crop important content
- Usage: `--vertical-mode crop`

### **Example Usage:**
```bash
# Pad mode (safe, default) - preserves all content
python -m src.interfaces.simple_cartoon_generator \
  --prompt "A dragon learns to bake cookies" \
  --vertical \
  --vertical-mode pad \
  --music music/background.mp3

# Crop mode (fills frame) - more native vertical look
python -m src.interfaces.simple_cartoon_generator \
  --prompt "Adventure story" \
  --vertical \
  --vertical-mode crop

# Custom dimensions and music volumes
python -m src.interfaces.simple_cartoon_generator \
  --prompt "Your story" \
  --vertical \
  --out-width 1080 \
  --out-height 1920 \
  --out-fps 30 \
  --music assets/bgm.mp3 \
  --music-volume 0.12 \
  --voice-volume 1.0

# Disable reel creation (use stitched video only)
python -m src.interfaces.simple_cartoon_generator \
  --prompt "Story here" \
  --no-reel
```

### **Reel Rendering Features:**
- ✅ **Platform-ready format**: 1080×1920, 30fps, H.264/AAC
- ✅ **Smart audio mixing**: Voice + optional background music
- ✅ **Volume control**: Adjust voice and music levels independently
- ✅ **Two vertical modes**: Pad (safe) or Crop (fills frame)
- ✅ **Works without audio**: Creates silent reel if no voice/music provided

## 🚀 **Quick Start**
```bash
# Clone the repository
git clone <repository-url>
cd WhyWouldYou-v1

# Install dependencies (compatible versions for Python 3.11)
pip install -r requirements.txt

# Setup the project
python -m src.utils.setup

# Download models
bash src/utils/download_models.sh
```

**Note**: The current configuration uses compatible versions:
- `transformers==4.49.0` and `TTS==0.22.0` for Python 3.11 compatibility
- Warning suppressions handle deprecation messages
- All functionality is preserved with stable versions

### **2. Configuration**
```bash
# Copy and edit configuration
cp config.env .env
# Add your API keys to .env file
```

### **3. Generate Your First Cartoon**
```bash
# Simple generation
python main.py "A dragon learns to bake cookies"

# Advanced generation with all features
python3 -m src.interfaces.simple_cartoon_generator \
  --prompt "Animal friends adventure" \
  --storyboard storyboards/horror.json \
  --no-reuse \
  --model-type realistic \
  --style indian \
  --video-format "shorts" \
  --animate \
  --animator ffmpeg \
  --language hi \
  --scene 1

# With storyboard
python -m src.interfaces.simple_cartoon_generator --prompt "A brave lion opens a smoothie shop" --storyboard storyboards/example.json

# Interactive interface
python -m src.interfaces.quick_start

# Batch generation
python -m src.interfaces.batch_generate
```

## 🎬 **WAN 2.1 Text-to-Video Pipeline**

### **Direct Text-to-Video Generation**
- **WAN 2.1 T2V** - Generates videos directly from text prompts
- **No image generation step** - Streamlined pipeline
- **Multi-scene support** - Automatic scene stitching
- **Local processing** - Runs entirely on your machine

### **Pipeline Flow:**
```
Script/Scenes JSON → WAN 2.1 (text2video) → Coqui TTS → FFmpeg stitch → Reel Renderer
```

### **Output Files:**
- **Scene clips**: `outputs/scenes/scene_1.mp4`, `scene_2.mp4`, etc.
- **Stitched video**: `outputs/stitched.mp4` (intermediate)
- **Final reel**: `outputs/final_reel.mp4` (upload-ready, 1080×1920, H.264/AAC, 30fps)

### **Usage Examples:**
```bash
# Basic generation (creates platform-ready reel by default)
python -m src.interfaces.simple_cartoon_generator --prompt "A cat walks on grass"

# With background music
python -m src.interfaces.simple_cartoon_generator \
  --prompt "Adventure story" \
  --music music/background.mp3 \
  --music-volume 0.12

# Crop mode (fills vertical frame)
python -m src.interfaces.simple_cartoon_generator \
  --prompt "Your story" \
  --vertical-mode crop

# Custom settings
python -m src.interfaces.simple_cartoon_generator \
  --prompt "Story here" \
  --out-width 1080 \
  --out-height 1920 \
  --out-fps 30 \
  --voice-volume 1.0 \
  --music-volume 0.15
```

## 📁 **Output Structure**

After generation, you'll find:

```
output/
├── scenes/
│   ├── scene_1.mp4          # Individual scene videos (WAN output)
│   ├── scene_2.mp4
│   └── scene_3.mp4
├── stitched.mp4              # Stitched scene videos (intermediate)
├── final_reel.mp4            # Platform-ready reel (upload-ready)
│                             # 1080×1920, H.264/AAC, 30fps
├── script.json               # Generated story script
├── storyboard.json           # Human-readable storyboard
├── audio_scene_1.wav         # Voiceover audio clips
├── audio_scene_2.wav
└── subtitles.srt             # Subtitle file (if enabled)
```

**Upload-ready file**: `outputs/final_reel.mp4` (or `output/final_reel.mp4` depending on `--output` flag)

## 📝 **Story Generation**

### **GPT-4 Powered:**
- Intelligent 3-scene story creation
- Detailed visual prompts
- Natural dialogue generation
- Duration-aware scripting

## 🎤 **Voice Generation**

### **Coqui TTS Integration:**
- **Local XTTS v2 model** - No API keys required
- **Multilingual support** - Hindi, English, Spanish, and more
- **Natural-sounding narration** with voice cloning
- **Professional audio quality** with automatic timing sync
- **Automatic speaker discovery** for language-appropriate voices

### **Language Support:**
```bash
# Hindi narration
python3 -m src.interfaces.simple_cartoon_generator --language hi

# English narration
python3 -m src.interfaces.simple_cartoon_generator --language en

# Spanish narration
python3 -m src.interfaces.simple_cartoon_generator --language es
```

### **Voice Cloning:**
- Use custom audio files for voice cloning
- Automatic speaker WAV discovery
- Multiple voice options per language

## ⚙️ **Configuration**

### **Environment Variables (.env):**
```env
OPENAI_API_KEY=your_openai_api_key_here
# Optional: Custom speaker WAV files
HINDI_SPEAKER_WAV=path/to/hindi_speaker.wav
ENGLISH_SPEAKER_WAV=path/to/english_speaker.wav
```

### **Video Settings:**
- **FPS**: 15 (optimized for social media)
- **Resolution**: 768x1024 (vertical format)
- **Duration**: Configurable (unlimited)
- **Style**: Realistic/anime

## 🧪 **Testing**

```bash
# Test image generation
python tests/test_image_generation.py

# Test animation system
python -c "from src.core.animation_generator import AnimationGenerator; print('✅ Animation system ready')"

# Test attention mask fix
python test_warning_fixes.py

# Test TTS warning suppression
python test_tts_warning_fixes.py

# Test subtitle functionality
python test_subtitle_functionality.py

# Reinstall dependencies (if needed)
python reinstall_dependencies.py
```

## 🔧 **Recent Updates & Fixes**

### **SVD Motion Animation (New)**
- **Feature**: Added SVD (Stable Video Diffusion) motion animation
- **Implementation**: AI-powered motion using ComfyUI workflows
- **Usage**: `--animator svd` for realistic motion animation
- **Status**: ✅ **Implemented**

### **Local TTS Integration (Updated)**
- **Feature**: Replaced ElevenLabs with local Coqui TTS XTTS v2 model
- **Benefits**: No API keys required, multilingual support, voice cloning
- **Languages**: Hindi, English, Spanish, and more
- **Status**: ✅ **Implemented**

### **Realistic Vision v4 Support (New)**
- **Feature**: Added Realistic Vision v4 model for realistic image generation
- **Usage**: `--model-type realistic` for photorealistic images
- **Status**: ✅ **Implemented**

### **Enhanced Animation System (Updated)**
- **Feature**: Dual animation system with FFmpeg and SVD options
- **Benefits**: More animation choices, better quality, unlimited length
- **Status**: ✅ **Implemented**

### **Attention Mask Issue (Fixed)**
- **Problem**: "The attention mask is not set and cannot be inferred from input because pad token is same as eos token"
- **Solution**: Properly configured tokenizer pad_token during pipeline initialization
- **Impact**: Eliminates warnings and improves text processing reliability
- **Status**: ✅ **Resolved**

### **CLIP Deprecation Warnings (Fixed)**
- **Problem**: "CLIPFeatureExtractor is deprecated" and "Some weights of the model checkpoint were not used"
- **Solution**: Updated transformers library and added warning suppression
- **Impact**: Eliminates deprecation warnings and unused weight warnings
- **Status**: ✅ **Resolved**

### **Subtitle Functionality (Added)**
- **Feature**: Added subtitle support to image generation
- **Implementation**: Professional subtitle rendering with background and text shadows
- **Usage**: Pass subtitle parameter to `generate_cartoon_image()` method
- **Status**: ✅ **Implemented**

### **TTS Compatibility Issues (Fixed)**
- **Problem**: "GPT2InferenceModel object has no attribute 'generate'", torchaudio deprecation warnings, and attention mask warnings
- **Solution**: Reverted to compatible versions (transformers==4.49.0, TTS==0.22.0) and added comprehensive warning suppression
- **Impact**: Restores XTTS voice generation functionality and eliminates all TTS-related warnings
- **Status**: ✅ **Resolved**

## 📚 **Documentation**

- **[Usage Guide](docs/USAGE_GUIDE.md)** - Detailed usage instructions
- **[Project Cleanup Summary](docs/PROJECT_CLEANUP_SUMMARY.md)** - Development history

## 🎯 **Usage Examples**

### **Simple Generation:**
```bash
python main.py "Space pirates discover treasure"
```

### **Advanced Generation with All Features:**
```bash
python3 -m src.interfaces.simple_cartoon_generator \
  --prompt "Animal friends adventure" \
  --storyboard storyboards/horror.json \
  --no-reuse \
  --model-type realistic \
  --style indian \
  --video-format "shorts" \
  --animate \
  --animator ffmpeg \
  --language hi \
  --scene 1
```

### **SVD Animation (Direct - No ComfyUI Required!):**
```bash
# Direct SVD animation - no ComfyUI needed!
python3 -m src.interfaces.simple_cartoon_generator \
  --prompt "Adventure story" \
  --animator svd \
  --motion-bucket-id 127 \
  --fps-id 6

# Test SVD implementation
python test_direct_svd.py
```

### **Custom Duration:**
```python
from src.core.generate_cartoon_short import CartoonShortsGenerator, VideoConfig

config = VideoConfig(
    prompt="A magical cat teaches other animals to dance",
    duration=60,  # 60-second video
    output_path="my_video"
)

generator = CartoonShortsGenerator(config)
output_path = generator.generate()
```

### **Batch Generation:**
```bash
python -m src.interfaces.batch_generate
# Follow the interactive prompts
```

## 🎉 **Results**

Your system generates:
- ✅ **Platform-ready vertical reels** (1080×1920, H.264/AAC, 30fps)
- ✅ **Direct text-to-video** using WAN 2.1 T2V (no image generation step)
- ✅ **Intelligent storytelling** powered by GPT-4
- ✅ **Natural multilingual narration** with local XTTS v2 model
- ✅ **Smart audio mixing** (voice + optional background music)
- ✅ **Two vertical modes** (pad for safety, crop for native look)
- ✅ **Upload-ready output** optimized for YouTube/Instagram

## 🚀 **Ready to Create Professional Video Reels!**

**Platform-ready vertical Reels/Shorts optimized for YouTube and Instagram!** 🎬✨
