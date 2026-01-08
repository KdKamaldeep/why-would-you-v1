#!/usr/bin/env bash
set -euo pipefail

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-models}"
mkdir -p \
  "${CHECKPOINTS_DIR}/musetalk" \
  "${CHECKPOINTS_DIR}/musetalkV15" \
  "${CHECKPOINTS_DIR}/syncnet" \
  "${CHECKPOINTS_DIR}/dwpose" \
  "${CHECKPOINTS_DIR}/face-parse-bisent" \
  "${CHECKPOINTS_DIR}/sd-vae" \
  "${CHECKPOINTS_DIR}/whisper"

echo "Using checkpoints dir: ${CHECKPOINTS_DIR}"
echo "HF_ENDPOINT: ${HF_ENDPOINT:-<default>}"

# Sanity
python -c "import huggingface_hub; print('huggingface_hub:', huggingface_hub.__version__)" >/dev/null 2>&1 \
  || { echo "❌ huggingface_hub not installed. Run: python -m pip install -U huggingface_hub"; exit 1; }

command -v gdown >/dev/null 2>&1 || { echo "❌ gdown not found. Run: python -m pip install -U gdown"; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "❌ curl not found. Install curl"; exit 1; }

python - <<'PY'
import os
from huggingface_hub import hf_hub_download

base = os.environ.get("CHECKPOINTS_DIR", "models")

def get(repo, filename, subdir=None):
    dest = os.path.join(base, subdir) if subdir else base
    os.makedirs(dest, exist_ok=True)
    path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        local_dir=dest,
        local_dir_use_symlinks=False,
    )
    print("✅", repo, filename, "->", path)

# MuseTalk V1.0
get("TMElyralab/MuseTalk", "musetalk/musetalk.json")
get("TMElyralab/MuseTalk", "musetalk/pytorch_model.bin")

# MuseTalk V1.5
get("TMElyralab/MuseTalk", "musetalkV15/musetalk.json")
get("TMElyralab/MuseTalk", "musetalkV15/unet.pth")

# SD VAE
get("stabilityai/sd-vae-ft-mse", "config.json", subdir="sd-vae")
get("stabilityai/sd-vae-ft-mse", "diffusion_pytorch_model.bin", subdir="sd-vae")

# Whisper tiny
get("openai/whisper-tiny", "config.json", subdir="whisper")
get("openai/whisper-tiny", "pytorch_model.bin", subdir="whisper")
get("openai/whisper-tiny", "preprocessor_config.json", subdir="whisper")

# DWPose
get("yzd-v/DWPose", "dw-ll_ucoco_384.pth", subdir="dwpose")

# SyncNet (LatentSync)
get("ByteDance/LatentSync", "latentsync_syncnet.pt", subdir="syncnet")
PY

# Face parse (BiSeNet) + ResNet18 backbone
gdown --id 154JgKpzCPW82qINcVieuPH3fZ2e0P812 \
  -O "${CHECKPOINTS_DIR}/face-parse-bisent/79999_iter.pth"

curl -L "https://download.pytorch.org/models/resnet18-5c106cde.pth" \
  -o "${CHECKPOINTS_DIR}/face-parse-bisent/resnet18-5c106cde.pth"

echo "✅ All weights downloaded into: ${CHECKPOINTS_DIR}"

