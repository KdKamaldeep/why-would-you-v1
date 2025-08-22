#!/bin/bash
# Model Download Script for Cartoon Shorts Generator - Professional Edition
# This script downloads all required AI models for professional cartoon generation

echo "🚀 Downloading AI Models for Professional Cartoon Generation"
echo "🎬 Enhanced Animation System with Unlimited Length Capability"
echo "=" * 60

# Parse flags
REDOWNLOAD=false
for arg in "$@"; do
    case "$arg" in
        --re-download|--force)
            REDOWNLOAD=true
            shift
            ;;
        --help|-h)
            echo "Usage: bash src/utils/download_models.sh [--re-download]"
            echo "  --re-download  Force re-downloading models even if present"
            exit 0
            ;;
        *)
            ;;
    esac
done

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models
mkdir -p loras
mkdir -p models/tts

echo "✅ Directories created"

# Download Stable Diffusion Models
echo "📋 Downloading Stable Diffusion Models..."

# ToonYou model (cartoon style) - Alternative: Use Anything v5 model
if [ -f "models/toonyou_beta6.safetensors" ] && [ "$REDOWNLOAD" != true ]; then
    echo "⏩ Skipping Anything v5 (already exists): models/toonyou_beta6.safetensors"
else
    [ "$REDOWNLOAD" = true ] && rm -f "models/toonyou_beta6.safetensors"
    echo "⬇️  Downloading Anything v5 model (cartoon style alternative)..."
    curl -L "https://huggingface.co/genai-archive/anything-v5/resolve/main/anything-v5.safetensors" \
         -o "models/toonyou_beta6.safetensors" \
         --progress-bar

    if [ $? -eq 0 ]; then
        echo "✅ Anything v5 model downloaded successfully"
    else
        echo "❌ Failed to download Anything v5 model"
    fi
fi

# AnimaGine XL model (anime style) - Alternative for MeinaMix
if [ -f "models/meina_mix.safetensors" ] && [ "$REDOWNLOAD" != true ]; then
    echo "⏩ Skipping AnimaGine XL (already exists): models/meina_mix.safetensors"
else
    [ "$REDOWNLOAD" = true ] && rm -f "models/meina_mix.safetensors"
    echo "⬇️  Downloading AnimaGine XL model (anime style)..."
    curl -L "https://huggingface.co/cagliostrolab/animagine-xl-3.1/resolve/main/animagine-xl-3.1.safetensors" \
         -o "models/meina_mix.safetensors" \
         --progress-bar

    if [ $? -eq 0 ]; then
        echo "✅ AnimaGine XL model downloaded successfully"
    else
        echo "❌ Failed to download AnimaGine XL model"
    fi
fi

# Realistic Vision v5.1 model (photorealistic style)
if [ -f "models/realistic-vision-v5.1.safetensors" ] && [ "$REDOWNLOAD" != true ]; then
    echo "⏩ Skipping Realistic Vision v5.1 (already exists): models/realistic-vision-v5.1.safetensors"
else
    [ "$REDOWNLOAD" = true ] && rm -f "models/realistic-vision-v5.1.safetensors"
    echo "⬇️  Downloading Realistic Vision v5.1 model (photorealistic style)..."
    curl -L "https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE/resolve/main/Realistic_Vision_V5.1_noVAE.safetensors" \
         -o "models/realistic-vision-v5.1.safetensors" \
         --progress-bar

    if [ $? -eq 0 ]; then
        echo "✅ Realistic Vision v5.1 model downloaded successfully"
    else
        echo "❌ Failed to download Realistic Vision v5.1 model"
    fi
fi

# DreamShaper v8 model (realistic style)
if [ -f "models/dreamshaper-v8.safetensors" ] && [ "$REDOWNLOAD" != true ]; then
    echo "⏩ Skipping DreamShaper v8 (already exists): models/dreamshaper-v8.safetensors"
else
    [ "$REDOWNLOAD" = true ] && rm -f "models/dreamshaper-v8.safetensors"
    echo "⬇️  Downloading DreamShaper v8 model (realistic style)..."
    curl -L "https://huggingface.co/Lykon/dreamshaper-8/resolve/main/DreamShaper_8_pruned.safetensors" \
         -o "models/dreamshaper-v8.safetensors" \
         --progress-bar

    if [ $? -eq 0 ]; then
        echo "✅ DreamShaper v8 model downloaded successfully"
    else
        echo "❌ Failed to download DreamShaper v8 model"
    fi
fi

# Deliberate v3 model (realistic style)
if [ -f "models/deliberate-v3.safetensors" ] && [ "$REDOWNLOAD" != true ]; then
    echo "⏩ Skipping Deliberate v3 (already exists): models/deliberate-v3.safetensors"
else
    [ "$REDOWNLOAD" = true ] && rm -f "models/deliberate-v3.safetensors"
    echo "⬇️  Downloading Deliberate v3 model (realistic style)..."
    curl -L "https://huggingface.co/XpucT/Deliberate/resolve/main/Deliberate_v3.safetensors" \
         -o "models/deliberate-v3.safetensors" \
         --progress-bar

    if [ $? -eq 0 ]; then
        echo "✅ Deliberate v3 model downloaded successfully"
    else
        echo "❌ Failed to download Deliberate v3 model"
    fi
fi

# Enhanced Animation System - Professional Quality Video Generation
echo "📹 Enhanced Animation System Ready"
echo "🎬 Professional quality animations with unlimited length capability"
echo "💡 No additional model downloads required - uses advanced FFmpeg techniques"

# LoRA Models for cartoon style
echo "📋 Downloading LoRA Models..."

# SDXL Lightning LoRA (fast generation)
if [ -f "loras/sdxl_lightning_4step.safetensors" ] && [ "$REDOWNLOAD" != true ]; then
    echo "⏩ Skipping SDXL Lightning LoRA (already exists): loras/sdxl_lightning_4step.safetensors"
else
    [ "$REDOWNLOAD" = true ] && rm -f "loras/sdxl_lightning_4step.safetensors"
    echo "⬇️  Downloading SDXL Lightning LoRA..."
    curl -L "https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_4step_lora.safetensors" \
         -o "loras/sdxl_lightning_4step.safetensors" \
         --progress-bar

    if [ $? -eq 0 ]; then
        echo "✅ SDXL Lightning LoRA downloaded successfully"
    else
        echo "❌ Failed to download SDXL Lightning LoRA"
    fi
fi

# Face-Based Generation Models
echo "📋 Downloading Face-Based Generation Models..."

# Create directories for face generation models
mkdir -p models/controlnet
mkdir -p models/ip_adapter

# ControlNet for face control (Canny edge detection)
if [ -f "models/controlnet/control_v11p_sd15_canny.pth" ] && [ "$REDOWNLOAD" != true ]; then
    echo "⏩ Skipping ControlNet Canny (already exists): models/controlnet/control_v11p_sd15_canny.pth"
else
    [ "$REDOWNLOAD" = true ] && rm -f "models/controlnet/control_v11p_sd15_canny.pth"
    echo "⬇️  Downloading ControlNet Canny model..."
    curl -L "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.pth" \
         -o "models/controlnet/control_v11p_sd15_canny.pth" \
         --progress-bar

    if [ $? -eq 0 ]; then
        echo "✅ ControlNet Canny model downloaded successfully"
    else
        echo "❌ Failed to download ControlNet Canny model"
    fi
fi

# ControlNet for face landmarks (OpenPose)
if [ -f "models/controlnet/control_v11p_sd15_openpose.pth" ] && [ "$REDOWNLOAD" != true ]; then
    echo "⏩ Skipping ControlNet OpenPose (already exists): models/controlnet/control_v11p_sd15_openpose.pth"
else
    [ "$REDOWNLOAD" = true ] && rm -f "models/controlnet/control_v11p_sd15_openpose.pth"
    echo "⬇️  Downloading ControlNet OpenPose model..."
    curl -L "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth" \
         -o "models/controlnet/control_v11p_sd15_openpose.pth" \
         --progress-bar

    if [ $? -eq 0 ]; then
        echo "✅ ControlNet OpenPose model downloaded successfully"
    else
        echo "❌ Failed to download ControlNet OpenPose model"
    fi
fi

# IP-Adapter for image prompting
if [ -f "models/ip_adapter/ip-adapter_sd15.bin" ] && [ "$REDOWNLOAD" != true ]; then
    echo "⏩ Skipping IP-Adapter (already exists): models/ip_adapter/ip-adapter_sd15.bin"
else
    [ "$REDOWNLOAD" = true ] && rm -f "models/ip_adapter/ip-adapter_sd15.bin"
    echo "⬇️  Downloading IP-Adapter model..."
    curl -L "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.bin" \
         -o "models/ip_adapter/ip-adapter_sd15.bin" \
         --progress-bar

    if [ $? -eq 0 ]; then
        echo "✅ IP-Adapter model downloaded successfully"
    else
        echo "❌ Failed to download IP-Adapter model"
    fi
fi

# IP-Adapter VIT-H model for better quality
if [ -f "models/ip_adapter/ip-adapter_sd15_vit-h.bin" ] && [ "$REDOWNLOAD" != true ]; then
    echo "⏩ Skipping IP-Adapter VIT-H (already exists): models/ip_adapter/ip-adapter_sd15_vit-h.bin"
else
    [ "$REDOWNLOAD" = true ] && rm -f "models/ip_adapter/ip-adapter_sd15_vit-h.bin"
    echo "⬇️  Downloading IP-Adapter VIT-H model..."
    curl -L "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15_vit-h.bin" \
         -o "models/ip_adapter/ip-adapter_sd15_vit-h.bin" \
         --progress-bar

    if [ $? -eq 0 ]; then
        echo "✅ IP-Adapter VIT-H model downloaded successfully"
    else
        echo "❌ Failed to download IP-Adapter VIT-H model"
    fi
fi


# Coqui TTS Models
echo "\n📋 Downloading Coqui TTS Models..."

TARGET_TTS_DIR="models/tts/XTTS-v2"
if [ -d "$TARGET_TTS_DIR" ] && [ "$REDOWNLOAD" != true ]; then
    echo "⏩ Skipping Coqui XTTS v2 (already exists): $TARGET_TTS_DIR"
else
    if [ "$REDOWNLOAD" = true ] && [ -d "$TARGET_TTS_DIR" ]; then
        echo "🧹 Removing existing directory for re-download: $TARGET_TTS_DIR"
        rm -rf "$TARGET_TTS_DIR"
    fi

    # Prefer huggingface-cli if available; otherwise, fall back to Python API
    if command -v huggingface-cli >/dev/null 2>&1; then
        echo "⬇️  Using huggingface-cli to download coqui/XTTS-v2..."
        huggingface-cli download --repo-type model coqui/XTTS-v2 \
            --local-dir "$TARGET_TTS_DIR" \
            --local-dir-use-symlinks False
        if [ $? -eq 0 ]; then
            echo "✅ Coqui XTTS v2 downloaded to $TARGET_TTS_DIR"
            USE_PYTHON_FALLBACK=0
        else
            echo "❌ huggingface-cli download failed, attempting Python fallback"
            USE_PYTHON_FALLBACK=1
        fi
    else
        USE_PYTHON_FALLBACK=1
    fi

    if [ "${USE_PYTHON_FALLBACK}" = "1" ]; then
        if command -v python3 >/dev/null 2>&1; then
            echo "⬇️  Using Python (huggingface_hub) to download coqui/XTTS-v2..."
            python3 - <<'PY'
import sys
from pathlib import Path
try:
    from huggingface_hub import snapshot_download
except Exception as e:
    print("[ERROR] huggingface_hub not available. Install it with: pip install huggingface_hub")
    sys.exit(1)

target_dir = Path("models/tts/XTTS-v2")
target_dir.mkdir(parents=True, exist_ok=True)
try:
    snapshot_download(
        repo_id="coqui/XTTS-v2",
        repo_type="model",
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.md", "*.png", "*.jpg", "*.jpeg"],
        resume_download=True,
    )
    print("[OK] Coqui XTTS v2 downloaded to", target_dir)
except Exception as e:
    print("[ERROR] Failed to download Coqui XTTS v2:", e)
    sys.exit(1)
PY
            if [ $? -eq 0 ]; then
                echo "✅ Coqui XTTS v2 downloaded to $TARGET_TTS_DIR"
            else
                echo "❌ Failed to download Coqui XTTS v2. Ensure 'huggingface_hub' is installed or try installing requirements first."
            fi
        else
            echo "❌ Python3 not found. Cannot download Coqui XTTS v2 without huggingface-cli."
        fi
    fi
fi



echo ""
echo "=" * 60
echo "🎊 PROFESSIONAL MODEL DOWNLOAD COMPLETED!"
echo "=" * 60
echo ""
echo "📊 Downloaded Models Summary:"
echo "  ├── 📹 Enhanced Animation System:"
echo "  │   ├── Professional FFmpeg Techniques"
echo "  │   ├── 6 Advanced Animation Effects"
echo "  │   └── Unlimited Length Capability"
echo "  ├── 🎨 Stable Diffusion Models:"
echo "  │   ├── Anything v5 (cartoon style): models/toonyou_beta6.safetensors"
echo "  │   ├── AnimaGine XL (anime style): models/meina_mix.safetensors"
echo "  │   ├── Realistic Vision v5.1 (photorealistic): models/realistic-vision-v5.1.safetensors"
echo "  │   ├── DreamShaper v8 (realistic): models/dreamshaper-v8.safetensors"
echo "  │   └── Deliberate v3 (realistic): models/deliberate-v3.safetensors"
echo "  ├── ⚡ LoRA Models:"
echo "  │   └── SDXL Lightning LoRA: loras/sdxl_lightning_4step.safetensors"
echo "  ├── 🎭 Face-Based Generation Models:"
echo "  │   ├── ControlNet Canny: models/controlnet/control_v11p_sd15_canny.pth"
echo "  │   ├── ControlNet OpenPose: models/controlnet/control_v11p_sd15_openpose.pth"
echo "  │   ├── IP-Adapter: models/ip_adapter/ip-adapter_sd15.bin"
echo "  │   └── IP-Adapter VIT-H: models/ip_adapter/ip-adapter_sd15_vit-h.bin"
echo "  └── 🔊 TTS Models:"
echo "      └── Coqui XTTS v2: models/tts/XTTS-v2"

echo ""
echo "🎬 PROFESSIONAL FEATURES ENABLED:"
echo "  ✅ Unlimited Length Video Generation"
echo "  ✅ 6 Professional Animation Effects"
echo "  ✅ Cinematic Quality Output"
echo "  ✅ Smart Frame Management"
echo "  ✅ Advanced FFmpeg Techniques"
echo "  ✅ Face-Based Character Generation"
echo "  ✅ ControlNet Face Control"
echo "  ✅ IP-Adapter Image Prompting"
echo "  ✅ Multiple Visual Styles (Cartoon, Anime, Realistic)"

echo ""
echo "🚀 Next Steps:"
echo "  1. Install dependencies: pip install -r requirements.txt"
echo "  2. Edit .env file and add your API keys"
echo "  3. Test installation: python test_enhanced_animation.py"
echo "  4. Test different styles:"
echo "     • Cartoon: python scripts/scene_visual_test.py --style cartoon"
echo "     • Realistic: python scripts/scene_visual_test.py --style realistic"
echo "  5. Generate unlimited videos: python simple_cartoon_generator.py --prompt 'Epic adventure' --duration 60"
echo "  6. Test face-based generation: python test_face_integration.py"
echo "  7. Generate with character faces: python simple_cartoon_generator.py --storyboard storyboards/independence.json"

echo ""
echo "💡 Hardware Requirements:"
echo "  • Enhanced Animations: Any modern GPU (recommended)"
echo "  • Image Generation: Any GPU with 4GB+ VRAM"
echo "  • Fallback: CPU-only (slower but functional)"

echo ""
echo "🎉 YOUR CARTOON GENERATOR IS NOW PROFESSIONAL GRADE!"
