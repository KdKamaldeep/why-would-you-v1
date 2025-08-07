# 🎬 Cartoon Shorts Generator

A complete Python CLI tool for creating vertical cartoon-style YouTube Shorts videos using AI-powered content generation.

## ✨ Features

- **🤖 AI Script Generation**: Uses OpenAI GPT-4 to create engaging scripts
- **🎨 Cartoon Image Generation**: Generates cartoon-style images using Stable Diffusion (ToonYou model)
- **🎵 AI Narration**: Creates natural-sounding voiceovers using ElevenLabs
- **🎭 Lip-Sync**: Applies lip-sync to characters using Wav2Lip
- **🎬 Video Processing**: Compiles everything into vertical YouTube Shorts format
- **📝 Subtitles**: Automatically adds styled subtitles
- **🎯 Vertical Format**: Optimized for 9:16 aspect ratio (1080x1920)

## 🛠️ Tech Stack

- **OpenAI GPT-4** → Script and visual prompt generation
- **Stable Diffusion (ToonYou)** → Cartoon image generation
- **ElevenLabs** → High-quality text-to-speech narration
- **Wav2Lip** → Lip-sync character faces to narration
- **FFmpeg** → Video processing, stitching, and final compilation
- **Python** → Core orchestration and automation

## 📋 Prerequisites

### System Requirements
- Python 3.8+
- FFmpeg installed and accessible in PATH
- CUDA-compatible GPU (optional, for faster processing)

### API Keys Required
1. **OpenAI API Key** - [Get it here](https://platform.openai.com/api-keys)
2. **ElevenLabs API Key** - [Get it here](https://elevenlabs.io/)
3. **Replicate API Key** - [Get it here](https://replicate.com/) (for Stable Diffusion)

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd WhyWouldYou-v1
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install FFmpeg

**Windows:**
```bash
# Using Chocolatey
choco install ffmpeg

# Or download from https://ffmpeg.org/download.html
# Extract to C:\ffmpeg and add C:\ffmpeg\bin to PATH
```

**macOS:**
```bash
# Using Homebrew
brew install ffmpeg

# Or download from https://evermeet.cx/ffmpeg/
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install ffmpeg

# Verify installation
ffmpeg -version
```

### 4. Install Wav2Lip

**Clone and setup Wav2Lip:**
```bash
# Clone the repository
git clone https://github.com/Rudrabha/Wav2Lip.git
cd Wav2Lip

# Download pretrained models
wget 'https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?share=EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxpQY6fQSlA' -O 'checkpoints/wav2lip.pth'
wget 'https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?share=EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxpQY6fQSlA' -O 'checkpoints/wav2lip_gan.pth'

# Install Wav2Lip dependencies
pip install -r requirements.txt

# Return to project root
cd ..
```

### 5. Download AI Models

**Create models directory:**
```bash
mkdir -p models
cd models
```

**Download Stable Diffusion models:**
```bash
# ToonYou model (cartoon style)
wget https://huggingface.co/ckpt/ToonYou/resolve/main/ToonYou_beta6.safetensors -O toonyou_beta6.safetensors

# MeinaMix model (anime style)
wget https://huggingface.co/Meina/MeinaMix/resolve/main/MeinaMix.safetensors -O meina_mix.safetensors
```

**Download AnimateDiff models:**
```bash
# AnimateDiff base model
wget https://huggingface.co/guoyww/animatediff/resolve/main/v1-5-pruned.ckpt -O animatediff_v1-5-pruned.ckpt

# AnimateDiff LoRA (animov)
wget https://huggingface.co/guoyww/animatediff/resolve/main/v1-5-pruned-emaonly.ckpt -O animatediff_v1-5-pruned-emaonly.ckpt
```

**Download motion LoRAs:**
```bash
# Create LoRA directory
mkdir -p loras
cd loras

# Download animov LoRA
wget https://huggingface.co/guoyww/animatediff/resolve/main/animov.safetensors -O animov.safetensors

cd ..
```

### 6. Set Up Environment Variables

Copy the configuration template:
```bash
cp config.env .env
```

Edit `.env` and add your API keys:
```bash
# Required API Keys
OPENAI_API_KEY=sk-your-openai-key-here
ELEVENLABS_API_KEY=your-elevenlabs-key-here
REPLICATE_API_KEY=your-replicate-key-here

# Optional Settings
DEFAULT_VOICE_ID=pNInz6obpgDQGcFmaJgB
DEFAULT_LANGUAGE=en
DEFAULT_DURATION=60
```

## 🎯 Usage

### Basic Usage
```bash
python generate_cartoon_short.py --topic "Amazing facts about space"
```

### Advanced Usage
```bash
python generate_cartoon_short.py \
  --topic "Why do cats purr?" \
  --duration 45 \
  --output "my_video" \
  --voice "pNInz6obpgDQGcFmaJgB" \
  --language "en"
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--topic` | Video topic (required) | - |
| `--duration` | Video duration in seconds | 60 |
| `--output` | Output directory | "output" |
| `--style` | Visual style | "cartoon" |
| `--voice` | ElevenLabs voice ID | "pNInz6obpgDQGcFmaJgB" |
| `--language` | Narration language | "en" |

## 📁 Output Structure

After running the script, you'll find:

```
output/
├── scene_1.png              # Generated cartoon images
├── scene_2.png
├── audio_1.mp3              # Generated narration audio
├── audio_2.mp3
├── clip_1.mp4               # Animated video clips
├── clip_2.mp4
├── subtitle_1.mp4           # Clips with subtitles
├── subtitle_2.mp4
├── final_clip_1.mp4         # Final clips with audio
├── final_clip_2.mp4
├── Your_Video_Title.mp4     # Final compiled video
└── metadata.json            # Video metadata and script
```

## 🎨 Customization

### Voice Selection
You can choose from different ElevenLabs voices:
- `pNInz6obpgDQGcFmaJgB` - Adam (Male)
- `21m00Tcm4TlvDq8ikWAM` - Rachel (Female)
- `AZnzlk1XvdvUeBnXmlld` - Domi (Female)
- `EXAVITQu4vr4xnSDxMaL` - Bella (Female)

### Visual Styles
The script supports different cartoon styles:
- `cartoon` - Classic cartoon style
- `anime` - Anime-inspired style
- `comic` - Comic book style

## 🔧 Troubleshooting

### Common Issues

1. **FFmpeg not found**
   ```bash
   # Ensure FFmpeg is in your PATH
   ffmpeg -version
   ```

2. **API Key errors**
   ```bash
   # Check your .env file
   cat .env
   ```

3. **CUDA/GPU issues**
   ```bash
   # Check if CUDA is available
   python -c "import torch; print(torch.cuda.is_available())"
   ```

4. **Memory issues**
   - Reduce image resolution in the script
   - Process fewer scenes at once
   - Use CPU instead of GPU

### Logs
Check the log file for detailed information:
```bash
tail -f cartoon_shorts.log
```

## 📊 Performance Tips

1. **Use GPU acceleration** when available
2. **Batch processing** for multiple videos
3. **Optimize image prompts** for better results
4. **Use shorter durations** for faster generation
5. **Pre-generate assets** for reuse

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for GPT-4 API
- ElevenLabs for text-to-speech
- Replicate for Stable Diffusion hosting
- The Wav2Lip project
- FFmpeg community

## 📞 Support

If you encounter any issues:
1. Check the troubleshooting section
2. Review the logs in `cartoon_shorts.log`
3. Open an issue on GitHub
4. Check the documentation

---

**Happy video creating! 🎬✨**
