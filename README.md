# 🎬 Cartoon Shorts Generator - Professional Edition

A complete AI-powered system for generating professional-quality cartoon videos with unlimited length capability.

## 🚀 **Features**

- 🎨 **Stable Diffusion Image Generation** - Professional cartoon-style images
- 🎬 **FFmpeg Animation System** - 6 professional animation effects
- 📝 **GPT-4 Story Generation** - Intelligent script creation
- 🎤 **ElevenLabs Voice Generation** - Natural narration
- ⚡ **Unlimited Length** - Generate videos of any duration
- 🎯 **Professional Quality** - Hollywood-grade output

## 📁 **Project Structure**

```
WhyWouldYou-v1/
├── 🎬 src/
│   ├── core/                    # Core system modules
│   │   ├── generate_cartoon_short.py
│   │   ├── image_generator.py
│   │   ├── animation_generator.py
│   │   ├── script_generator.py
│   │   ├── voice_generator.py
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

## 🚀 **Quick Start**

### **1. Installation**
```bash
# Clone the repository
git clone <repository-url>
cd WhyWouldYou-v1

# Install dependencies
pip install -r requirements.txt

# Setup the project
python -m src.utils.setup

# Download models
bash src/utils/download_models.sh
```

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

# Interactive interface
python -m src.interfaces.quick_start

# Batch generation
python -m src.interfaces.batch_generate
```

## 🎬 **Animation System**

### **6 Professional Effects:**
1. **Cinematic Zoom-Pan** - Smooth camera movements
2. **Smooth Slide Animation** - Organic motion
3. **Organic Rotation** - Natural spinning effects
4. **Parallax Motion** - Depth and perspective
5. **Breathing Effect** - Subtle pulsing
6. **Drift Animation** - Gentle floating motion

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
- Anything v5 (cartoon style)
- AnimaGine XL (anime style)
- SDXL Lightning LoRA (fast generation)

## 📝 **Story Generation**

### **GPT-4 Powered:**
- Intelligent 3-scene story creation
- Detailed visual prompts
- Natural dialogue generation
- Duration-aware scripting

## 🎤 **Voice Generation**

### **ElevenLabs Integration:**
- Natural-sounding narration
- Multiple voice options
- Professional audio quality
- Automatic timing sync

## ⚙️ **Configuration**

### **Environment Variables (.env):**
```env
OPENAI_API_KEY=your_openai_api_key_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
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
python test_attention_mask_fix.py

# Test subtitle functionality
python test_subtitle_functionality.py

# Update dependencies to fix warnings
python update_dependencies.py

# Fix TTS and torchaudio issues
python fix_tts_issues.py
```

## 🔧 **Recent Fixes**

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
- **Problem**: "GPT2InferenceModel object has no attribute 'generate'" and torchaudio deprecation warnings
- **Solution**: Updated TTS to >=0.25.0 and added warning suppression
- **Impact**: Fixes XTTS voice generation and eliminates torchaudio warnings
- **Status**: ✅ **Resolved**

## 📚 **Documentation**

- **[Usage Guide](docs/USAGE_GUIDE.md)** - Detailed usage instructions
- **[Project Cleanup Summary](docs/PROJECT_CLEANUP_SUMMARY.md)** - Development history

## 🎯 **Usage Examples**

### **Simple Generation:**
```bash
python main.py "Space pirates discover treasure"
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
- ✅ **High-quality animations** using advanced FFmpeg techniques
- ✅ **Intelligent storytelling** powered by GPT-4
- ✅ **Natural narration** with ElevenLabs voices
- ✅ **Cinematic effects** for engaging content

## 🚀 **Ready to Create Professional Cartoons!**

**No more limitations. No more complexity. Just unlimited professional-quality cartoon generation!** 🎬✨
