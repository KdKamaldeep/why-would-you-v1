# 🎬 Cartoon Shorts Generator - Professional Edition

A complete AI-powered system for generating professional-quality cartoon videos with unlimited length capability and advanced animation options.

## 🚀 **Features**

- 🎨 **Stable Diffusion Image Generation** - Professional cartoon-style images with Realistic Vision v4 support
- 🎭 **Face-Based Character Generation** - Use existing faces for character consistency
- 🎬 **Dual Animation System** - FFmpeg effects + SVD motion animation
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
│   │   ├── simple_cartoon_generator.py
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

## 🎬 **Video Formats**

The system now supports multiple video formats:

### **YouTube Shorts (9:16 Aspect Ratio)**
- **Dimensions**: 768x1024 pixels
- **Perfect for**: TikTok, Instagram Reels, YouTube Shorts
- **Usage**: `--video-format shorts` (default)

### **Normal Video (16:9 Aspect Ratio)**
- **Dimensions**: 1920x1080 pixels  
- **Perfect for**: YouTube, Vimeo, general video platforms
- **Usage**: `--video-format normal`

### **Example Usage:**
```bash
# Create YouTube Shorts (default)
python main.py "A dragon learns to bake cookies" --video-format shorts

# Create normal video
python main.py "A dragon learns to bake cookies" --video-format normal
```

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

# Face-based character generation (using storyboard)
python -m src.interfaces.simple_cartoon_generator --prompt "A brave lion opens a smoothie shop" --storyboard storyboards/example.json

# Interactive interface
python -m src.interfaces.quick_start

# Batch generation
python -m src.interfaces.batch_generate
```

## 🎬 **Animation System**

### **Dual Animation Options:**

#### **1. FFmpeg Animation (Default)**
**6 Professional Effects:**
1. **Cinematic Zoom-Pan** - Smooth camera movements
2. **Smooth Slide Animation** - Organic motion
3. **Organic Rotation** - Natural spinning effects
4. **Parallax Motion** - Depth and perspective
5. **Breathing Effect** - Subtle pulsing
6. **Drift Animation** - Gentle floating motion

#### **2. SVD Motion Animation (Advanced)**
- **AI-powered motion** using Stable Video Diffusion
- **Realistic movement** with 25-frame sequences
- **Automatic looping** for longer videos
- **Configurable motion intensity** (0-255)
- **Requires ComfyUI** for full functionality

### **Usage Examples:**
```bash
# FFmpeg animation (default)
python3 -m src.interfaces.simple_cartoon_generator --prompt "Adventure story" --animator ffmpeg

# SVD motion animation (requires ComfyUI)
python3 -m src.interfaces.simple_cartoon_generator --prompt "Adventure story" --animator svd
```

### **Unlimited Length Capability:**
- No 24-frame limits like AnimateDiff
- Generate 30s, 60s, or longer videos
- Professional quality throughout
- Smart frame management

## 🎨 **Image Generation**

### **Stable Diffusion Integration:**
- Professional cartoon-style images
- Enhanced prompts for better results
- Automatic fallback to placeholders
- Memory-optimized processing

### **Model Support:**
- **Realistic Vision v4** (realistic style) - Default for realistic generation
- **Anything v5** (cartoon style)
- **AnimaGine XL** (anime style)
- **DreamShaper v8** (artistic style)
- **Deliberate v3** (detailed style)

### **Model Selection:**
```bash
# Realistic style
python3 -m src.interfaces.simple_cartoon_generator --model-type realistic

# Cartoon style
python3 -m src.interfaces.simple_cartoon_generator --model-type cartoon

# Anime style
python3 -m src.interfaces.simple_cartoon_generator --model-type anime
```

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
- **Style**: Cartoon/anime

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

### **SVD Animation (Requires ComfyUI):**
```bash
# First, run ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt
python main.py --listen 127.0.0.1 --port 8188

# Then use SVD animation
python3 -m src.interfaces.simple_cartoon_generator \
  --prompt "Adventure story" \
  --animator svd \
  --motion-bucket-id 127 \
  --fps-id 6
```

### **Custom Duration:**
```python
from src.core.generate_cartoon_short import CartoonShortsGenerator, VideoConfig

config = VideoConfig(
    prompt="A magical cat teaches other animals to dance",
    duration=60,  # 60-second video
    output_path="my_cartoon"
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
- ✅ **Professional cartoon videos** with unlimited length
- ✅ **High-quality animations** using FFmpeg and SVD techniques
- ✅ **Intelligent storytelling** powered by GPT-4
- ✅ **Natural multilingual narration** with local XTTS v2 model
- ✅ **Cinematic effects** for engaging content
- ✅ **Realistic images** with Realistic Vision v4 support
- ✅ **Multiple video formats** (Shorts, Normal, Custom)

## 🚀 **Ready to Create Professional Cartoons!**

**No more limitations. No more complexity. Just unlimited professional-quality cartoon generation!** 🎬✨
