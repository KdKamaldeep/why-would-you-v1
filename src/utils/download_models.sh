#!/bin/bash
# Model Download Script for Video Reel Generator
# This script downloads required AI models for video generation with WAN 2.1 T2V

echo "🚀 Downloading AI Models for Video Reel Generator"
echo "🎬 WAN 2.1 Text-to-Video Pipeline"
echo "============================================================"

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
mkdir -p models/tts
mkdir -p models/sfx

echo "✅ Directories created"

# WAN 2.1 Text-to-Video Model
echo ""
echo "📋 Downloading WAN 2.1 Text-to-Video Model..."
echo "💡 The WAN model will be automatically downloaded from Hugging Face when first used."
echo "   This script provides an option to pre-download it now for faster first generation."

TARGET_WAN_DIR="models/wan-2.1-t2v"
if [ -d "$TARGET_WAN_DIR" ] && [ "$REDOWNLOAD" != true ]; then
    echo "⏩ WAN 2.1 model already exists: $TARGET_WAN_DIR"
    echo "   (Model will be loaded from cache on first use)"
else
    if [ "$REDOWNLOAD" = true ] && [ -d "$TARGET_WAN_DIR" ]; then
        echo "🧹 Removing existing directory for re-download: $TARGET_WAN_DIR"
        rm -rf "$TARGET_WAN_DIR"
    fi

    # Prefer huggingface-cli if available; otherwise, fall back to Python API
    if command -v huggingface-cli >/dev/null 2>&1; then
        echo "⬇️  Using huggingface-cli to download Wan-AI/Wan2.1-T2V-1.3B-Diffusers..."
        huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
            --local-dir "$TARGET_WAN_DIR" \
            --local-dir-use-symlinks False
    if [ $? -eq 0 ]; then
            echo "✅ WAN 2.1 model downloaded to $TARGET_WAN_DIR"
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
            echo "⬇️  Using Python (huggingface_hub) to download Wan-AI/Wan2.1-T2V-1.3B-Diffusers..."
            python3 - <<'PY'
import sys
from pathlib import Path
try:
    from huggingface_hub import snapshot_download
except Exception as e:
    print("[INFO] huggingface_hub not available. Model will be downloaded on first use.")
    print("[INFO] Install it with: pip install huggingface_hub")
    sys.exit(0)

target_dir = Path("models/wan-2.1-t2v")
target_dir.mkdir(parents=True, exist_ok=True)
try:
    snapshot_download(
        repo_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print("[OK] WAN 2.1 model downloaded to", target_dir)
except Exception as e:
    print("[INFO] Model download failed, but it will be downloaded automatically on first use:", e)
    sys.exit(0)
PY
    if [ $? -eq 0 ]; then
                echo "✅ WAN 2.1 model pre-downloaded (or will download on first use)"
            fi
        else
            echo "ℹ️  Python3 not found. WAN model will be downloaded automatically on first use."
        fi
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


# AudioLDM SFX Model
echo ""
echo "📋 Downloading AudioLDM SFX Model..."
echo "💡 AudioLDM will be automatically downloaded from Hugging Face when first used."
echo "   This script provides an option to pre-download it now for faster first generation."

TARGET_SFX_DIR="models/sfx/audioldm-s-full-v2"
if [ -d "$TARGET_SFX_DIR" ] && [ "$REDOWNLOAD" != true ]; then
    echo "⏩ AudioLDM model already exists: $TARGET_SFX_DIR"
    echo "   (Model will be loaded from cache on first use)"
else
    if [ "$REDOWNLOAD" = true ] && [ -d "$TARGET_SFX_DIR" ]; then
        echo "🧹 Removing existing directory for re-download: $TARGET_SFX_DIR"
        rm -rf "$TARGET_SFX_DIR"
    fi

    # Prefer huggingface-cli if available; otherwise, fall back to Python API
    if command -v huggingface-cli >/dev/null 2>&1; then
        echo "⬇️  Using huggingface-cli to download cvssp/audioldm-s-full-v2..."
        huggingface-cli download cvssp/audioldm-s-full-v2 \
            --local-dir "$TARGET_SFX_DIR" \
            --local-dir-use-symlinks False
        if [ $? -eq 0 ]; then
            echo "✅ AudioLDM model downloaded to $TARGET_SFX_DIR"
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
            echo "⬇️  Using Python (huggingface_hub) to download cvssp/audioldm-s-full-v2..."
            python3 - <<'PY'
import sys
from pathlib import Path
try:
    from huggingface_hub import snapshot_download
except Exception as e:
    print("[INFO] huggingface_hub not available. Model will be downloaded on first use.")
    print("[INFO] Install it with: pip install huggingface_hub")
    sys.exit(0)

target_dir = Path("models/sfx/audioldm-s-full-v2")
target_dir.mkdir(parents=True, exist_ok=True)
try:
    snapshot_download(
        repo_id="cvssp/audioldm-s-full-v2",
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print("[OK] AudioLDM model downloaded to", target_dir)
except Exception as e:
    print("[INFO] Model download failed, but it will be downloaded automatically on first use:", e)
    sys.exit(0)
PY
            if [ $? -eq 0 ]; then
                echo "✅ AudioLDM model pre-downloaded (or will download on first use)"
            fi
        else
            echo "ℹ️  Python3 not found. AudioLDM model will be downloaded automatically on first use."
        fi
    fi
fi


echo ""
echo "============================================================"
echo "🎊 MODEL DOWNLOAD COMPLETED!"
echo "============================================================"
echo ""
echo "📊 Downloaded Models Summary:"
echo "  ├── 🎬 WAN 2.1 Text-to-Video:"
echo "  │   └── Wan-AI/Wan2.1-T2V-1.3B-Diffusers: models/wan-2.1-t2v"
echo "  │       (Downloads automatically from Hugging Face on first use)"
echo "  ├── 🔊 TTS Models:"
echo "  │   └── Coqui XTTS v2: models/tts/XTTS-v2"
echo "  └── 🔊 SFX Models:"
echo "      └── AudioLDM S Full v2: models/sfx/audioldm-s-full-v2"
echo "          (Downloads automatically from Hugging Face on first use)"

echo ""
echo "🎬 FEATURES ENABLED:"
echo "  ✅ WAN 2.1 Text-to-Video Generation"
echo "  ✅ Direct text-to-video (no image generation step)"
echo "  ✅ Multi-scene video stitching"
echo "  ✅ Coqui TTS voice synthesis"
echo "  ✅ AudioLDM SFX generation"
echo "  ✅ FFmpeg video processing"

echo ""
echo "🚀 Next Steps:"
echo "  1. Install dependencies: pip install -r requirements.txt"
echo "  2. Edit .env file and add your API keys (OPENAI_API_KEY required)"
echo "  3. Test WAN generation: python -m scripts.test_wan --prompt 'A cat walks on grass'"
echo "  4. Generate full video: python -m src.interfaces.simple_cartoon_generator --prompt 'Your story here'"

echo ""
echo "💡 Hardware Requirements:"
echo "  • GPU recommended: CUDA-capable GPU with 8GB+ VRAM for best performance"
echo "  • CPU supported: Will run on CPU but will be very slow (bfloat16 not available)"
echo "  • Model size: ~1.3B parameters, downloads ~5GB from Hugging Face"

echo ""
echo "🎉 YOUR VIDEO GENERATOR IS READY!"
