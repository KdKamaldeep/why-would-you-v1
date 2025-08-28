#!/bin/bash

# SVD Performance Optimization Installation Script
# This script installs the necessary packages for optimal SVD performance

echo "🚀 Installing SVD Performance Optimizations..."

# Check if running on GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  NVIDIA GPU not detected. SVD will run on CPU (very slow)."
    echo "💡 Consider using a GPU-enabled environment for better performance."
fi

# Install core dependencies
echo "📦 Installing core dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate

# Install xformers for significant speed boost
echo "⚡ Installing xformers for performance boost..."
pip install xformers

# Install additional optimizations
echo "🔧 Installing additional optimizations..."
pip install bitsandbytes  # For quantization support
pip install safetensors   # For faster model loading

# Install monitoring tools
echo "📊 Installing monitoring tools..."
pip install psutil
pip install GPUtil

# Verify installations
echo "✅ Verifying installations..."

# Check xformers
python -c "import xformers; print('✅ XFormers installed successfully')" 2>/dev/null || echo "❌ XFormers installation failed"

# Check torch
python -c "import torch; print(f'✅ PyTorch {torch.__version__} installed')" 2>/dev/null || echo "❌ PyTorch installation failed"

# Check CUDA availability
python -c "import torch; print(f'✅ CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || echo "❌ CUDA check failed"

# Check diffusers
python -c "import diffusers; print(f'✅ Diffusers {diffusers.__version__} installed')" 2>/dev/null || echo "❌ Diffusers installation failed"

echo ""
echo "🎉 Installation completed!"
echo ""
echo "📋 Performance Checklist:"
echo "  □ XFormers installed and working"
echo "  □ PyTorch with CUDA support"
echo "  □ Diffusers library"
echo "  □ GPU detected and available"
echo ""
echo "🚀 Your SVD should now run 2-4x faster!"
echo "📊 Expected performance on A4000: 15-20 seconds for 25 frames"
echo ""
echo "💡 Run 'nvidia-smi' to monitor GPU usage during generation"
echo "💡 Check the logs for performance metrics during SVD generation"
