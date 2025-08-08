@echo off
echo 🔧 Fixing Image Generation Issues
echo ================================

echo 📦 Installing required AI packages...
python install_ai_dependencies.py

echo.
echo 🔍 Checking if models are downloaded...
if not exist models\toonyou_beta6.safetensors (
    echo ❌ Models not found! Running model download...
    call download_models.bat
) else (
    echo ✅ Models found!
)

echo.
echo 🎬 Testing image generation...
python -c "from image_generator import ImageGenerator; ig = ImageGenerator(); print('✅ Image generator initialized successfully!')"

echo.
echo ✨ Setup complete! Try generating a cartoon now:
echo python simple_cartoon_generator.py --prompt "A cute cat playing with yarn"
pause
