#!/bin/bash
# Model Download Script for Cartoon Shorts Generator
# This script downloads all required AI models for the project

echo "🎬 Downloading AI Models for Cartoon Shorts Generator"
echo "=" * 50

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models
mkdir -p Wav2Lip/checkpoints
mkdir -p loras

echo "✅ Directories created"

# Download Stable Diffusion Models
echo "📋 Downloading Stable Diffusion Models..."

# ToonYou model (cartoon style) - Alternative: Use Anything v5 model
echo "⬇️  Downloading Anything v5 model (cartoon style alternative)..."
curl -L "https://huggingface.co/genai-archive/anything-v5/resolve/main/anything-v5.safetensors" \
     -o "models/toonyou_beta6.safetensors" \
     --progress-bar

if [ $? -eq 0 ]; then
    echo "✅ Anything v5 model downloaded successfully"
else
    echo "❌ Failed to download Anything v5 model"
fi

# AnimaGine XL model (anime style) - Alternative for MeinaMix
echo "⬇️  Downloading AnimaGine XL model (anime style)..."
curl -L "https://huggingface.co/cagliostrolab/animagine-xl-3.1/resolve/main/animagine-xl-3.1.safetensors" \
     -o "models/meina_mix.safetensors" \
     --progress-bar

if [ $? -eq 0 ]; then
    echo "✅ AnimaGine XL model downloaded successfully"
else
    echo "❌ Failed to download AnimaGine XL model"
fi

# AnimateDiff Models
echo "📋 Downloading AnimateDiff Models..."

# AnimateDiff Motion Module v2
echo "⬇️  Downloading AnimateDiff Motion Module v2..."
curl -L "https://huggingface.co/conrevo/AnimateDiff-A1111/resolve/main/motion_module/mm_sd15_v2.safetensors" \
     -o "models/mm_sd_v15_v2.safetensors" \
     --progress-bar

if [ $? -eq 0 ]; then
    echo "✅ AnimateDiff Motion Module v2 downloaded successfully"
else
    echo "❌ Failed to download AnimateDiff Motion Module v2"
fi

# AnimateDiff Motion Module v3
echo "⬇️  Downloading AnimateDiff Motion Module v3..."
curl -L "https://huggingface.co/conrevo/AnimateDiff-A1111/resolve/main/motion_module/mm_sd15_v3.safetensors" \
     -o "models/mm_sd_v15_v3.safetensors" \
     --progress-bar

if [ $? -eq 0 ]; then
    echo "✅ AnimateDiff Motion Module v3 downloaded successfully"
else
    echo "❌ Failed to download AnimateDiff Motion Module v3"
fi

# LoRA Models for cartoon style
echo "📋 Downloading LoRA Models..."

# SDXL Lightning LoRA (fast generation)
echo "⬇️  Downloading SDXL Lightning LoRA..."
curl -L "https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_4step_lora.safetensors" \
     -o "loras/sdxl_lightning_4step.safetensors" \
     --progress-bar

if [ $? -eq 0 ]; then
    echo "✅ SDXL Lightning LoRA downloaded successfully"
else
    echo "❌ Failed to download SDXL Lightning LoRA"
fi

# Face detection model for Wav2Lip
echo "⬇️  Downloading face detection model..."
curl -L "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" \
     -o "Wav2Lip/face_detection/detection/sfd/s3fd.pth" \
     --progress-bar

if [ $? -eq 0 ]; then
    echo "✅ Face detection model downloaded successfully"
else
    echo "❌ Failed to download face detection model"
fi

echo ""
echo "=" * 50
echo "🎉 Model download completed!"
echo ""
echo "📊 Downloaded Models Summary:"
echo "  ├── Stable Diffusion Models:"
echo "  │   ├── Anything v5 (cartoon style): models/toonyou_beta6.safetensors"
echo "  │   └── AnimaGine XL (anime style): models/meina_mix.safetensors"
echo "  ├── AnimateDiff Models:"
echo "  │   ├── Motion module v2: models/mm_sd_v15_v2.safetensors"
echo "  │   └── Motion module v3: models/mm_sd_v15_v3.safetensors"
echo "  ├── LoRA Models:"
echo "  │   └── SDXL Lightning LoRA: loras/sdxl_lightning_4step.safetensors"
echo "  └── Wav2Lip Models:"
echo "      ├── Main model: Wav2Lip/checkpoints/wav2lip.pth"
echo "      ├── GAN model: Wav2Lip/checkpoints/wav2lip_gan.pth"
echo "      └── Face detection: Wav2Lip/face_detection/detection/sfd/s3fd.pth"
echo ""
echo "💡 Next steps:"
echo "  1. Make sure you have cloned Wav2Lip repository"
echo "  2. Edit .env file and add your API keys"
echo "  3. Run: python generate_cartoon_short.py --prompt 'Your story prompt'"
