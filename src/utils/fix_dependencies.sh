#!/usr/bin/env bash
set -euo pipefail

# Fix/verify dependencies for Linux, and attempt to install xFormers if CUDA is available.
# Usage:
#   bash src/utils/fix_dependencies.sh
#
# Notes:
# - Run inside your project venv
# - Torch should already be installed (CPU or CUDA). This script will NOT reinstall torch.

echo "=== WhyWouldYou-v1 dependency fixer (Linux) ==="

PYTHON_BIN=${PYTHON:-python3}
PIP_BIN=${PIP:-pip3}

command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "Error: $PYTHON_BIN not found."; exit 1; }
command -v "$PIP_BIN" >/dev/null 2>&1 || { echo "Error: $PIP_BIN not found."; exit 1; }

echo "- Python: $($PYTHON_BIN --version 2>&1)"
echo "- Pip:    $($PIP_BIN --version 2>&1)"

# Ensure basic build tooling for optional source builds
if command -v apt >/dev/null 2>&1; then
  echo "- Ensuring build tools (Debian/Ubuntu)"
  sudo apt update -y >/dev/null 2>&1 || true
  sudo apt install -y build-essential >/dev/null 2>&1 || true
fi

echo "- Upgrading pip/setuptools/wheel"
$PIP_BIN install --upgrade pip setuptools wheel >/dev/null

# Collect torch/CUDA info
read -r HAS_TORCH TORCH_VER CUDA_VER CUDA_AVAIL <<EOF
$($PYTHON_BIN - <<'PY'
try:
    import torch
    has_torch = 'yes'
    torch_ver = getattr(torch, '__version__', 'unknown')
    cuda_ver = getattr(torch.version, 'cuda', None) or 'cpu'
    cuda_avail = 'yes' if torch.cuda.is_available() else 'no'
    print(has_torch, torch_ver, cuda_ver, cuda_avail)
except Exception:
    print('no', 'none', 'cpu', 'no')
PY
)
EOF

echo "- Torch installed: $HAS_TORCH"
echo "- Torch version:   $TORCH_VER"
echo "- CUDA reported:   $CUDA_VER"
echo "- CUDA available:  $CUDA_AVAIL"

# Ensure pinned HF stack per requirements.txt
echo "- Installing/aligning core ML stack (diffusers/transformers/accelerate/etc.)"
$PIP_BIN install -r requirements.txt

# Attempt xformers install only if CUDA available
if [[ "$CUDA_AVAIL" == "yes" ]]; then
  echo "- CUDA detected; attempting to install xformers (optional)"
  set +e
  $PIP_BIN install --no-cache-dir xformers
  X_STATUS=$?
  set -e

  if [[ $X_STATUS -ne 0 ]]; then
    echo "⚠️  xformers install failed via PyPI. Trying alternative candidates..."
    set +e
    # Try a few likely versions. If all fail, we keep going without xformers.
    $PIP_BIN install --no-cache-dir 'xformers==0.0.28.post3'
    X_STATUS=$?
    if [[ $X_STATUS -ne 0 ]]; then
      $PIP_BIN install --no-cache-dir 'xformers==0.0.28.post2'
      X_STATUS=$?
    fi
    if [[ $X_STATUS -ne 0 ]]; then
      $PIP_BIN install --no-cache-dir 'xformers==0.0.27.post2'
      X_STATUS=$?
    fi
    set -e
    if [[ $X_STATUS -ne 0 ]]; then
      echo "❌ Could not install xformers prebuilt wheels for Torch=$TORCH_VER CUDA=$CUDA_VER."
      echo "   - You can continue without xformers (the app will run more slowly)."
      echo "   - Or try building from source (heavy):"
      echo "       $PIP_BIN install ninja"
      echo "       $PIP_BIN install git+https://github.com/facebookresearch/xformers.git"
    else
      echo "✅ Installed xformers successfully."
    fi
  else
    echo "✅ Installed xformers successfully."
  fi
else
  echo "- CUDA not available; skipping xformers."
fi

echo "\n✅ Dependency check complete."
echo "- If you just installed/changed core libs, consider restarting your shell or Python session."

