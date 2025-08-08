@echo off
echo 🔧 Fixing PyTorch Version Conflicts
echo ==================================

echo This will uninstall and reinstall PyTorch packages to fix version conflicts.
echo.
set /p choice="Continue? (y/N): "
if /i not "%choice%"=="y" if /i not "%choice%"=="yes" (
    echo Operation cancelled.
    pause
    exit /b 0
)

echo.
echo 🗑️ Uninstalling conflicting packages...
pip uninstall torch torchvision torchaudio xformers -y

echo.
echo 📦 Installing compatible PyTorch versions...
echo 🎯 Trying CUDA version first...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

if errorlevel 1 (
    echo ⚠️ CUDA installation failed, trying CPU version...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
)

echo.
echo 📦 Installing AI packages...
pip install diffusers accelerate safetensors transformers

echo.
echo 📦 Installing xformers (optional)...
pip install xformers
if errorlevel 1 (
    echo ⚠️ xformers failed - this is optional
)

echo.
echo 🔍 Testing installation...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

echo.
echo ✅ PyTorch conflicts should now be resolved!
echo You can now run: python simple_cartoon_generator.py --prompt "Your story"
pause
