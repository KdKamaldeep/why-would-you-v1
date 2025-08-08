@echo off
REM Model Download Script for Cartoon Shorts Generator (Windows)
REM This script downloads all required AI models for the project

echo 🎬 Downloading AI Models for Cartoon Shorts Generator
echo ==================================================

REM Create necessary directories
echo 📁 Creating directories...
mkdir models 2>nul
mkdir Wav2Lip\checkpoints 2>nul
mkdir loras 2>nul
echo ✅ Directories created

REM Download Stable Diffusion Models
echo.
echo 📋 Downloading Stable Diffusion Models...

REM ToonYou model (cartoon style)
echo ⬇️  Downloading ToonYou model...
curl -L "https://huggingface.co/ckpt/ToonYou/resolve/main/ToonYou_beta6.safetensors" -o "models/toonyou_beta6.safetensors" --progress-bar
if %errorlevel% equ 0 (
    echo ✅ ToonYou model downloaded successfully
) else (
    echo ❌ Failed to download ToonYou model
)

REM MeinaMix model (anime style)
echo ⬇️  Downloading MeinaMix model...
curl -L "https://huggingface.co/Meina/MeinaMix/resolve/main/MeinaMix.safetensors" -o "models/meina_mix.safetensors" --progress-bar
if %errorlevel% equ 0 (
    echo ✅ MeinaMix model downloaded successfully
) else (
    echo ❌ Failed to download MeinaMix model
)

REM AnimateDiff Models
echo.
echo 📋 Downloading AnimateDiff Models...

REM AnimateDiff v1.5 model
echo ⬇️  Downloading AnimateDiff v1.5 model...
curl -L "https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_mm.ckpt" -o "models/animatediff_v1-5-pruned.ckpt" --progress-bar
if %errorlevel% equ 0 (
    echo ✅ AnimateDiff model downloaded successfully
) else (
    echo ❌ Failed to download AnimateDiff model
)

REM AnimateDiff Motion Module
echo ⬇️  Downloading AnimateDiff Motion Module...
curl -L "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt" -o "models/mm_sd_v15_v2.ckpt" --progress-bar
if %errorlevel% equ 0 (
    echo ✅ AnimateDiff Motion Module downloaded successfully
) else (
    echo ❌ Failed to download AnimateDiff Motion Module
)

REM LoRA Models for cartoon style
echo.
echo 📋 Downloading LoRA Models...

REM Cartoon LoRA
echo ⬇️  Downloading Cartoon LoRA...
curl -L "https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/resolve/main/pytorch_lora_weights.safetensors" -o "loras/animov.safetensors" --progress-bar
if %errorlevel% equ 0 (
    echo ✅ Cartoon LoRA downloaded successfully
) else (
    echo ❌ Failed to download Cartoon LoRA
)

REM Wav2Lip Models
echo.
echo 📋 Downloading Wav2Lip Models...

REM Wav2Lip pretrained model
echo ⬇️  Downloading Wav2Lip pretrained model...
curl -L "https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?share=EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxpQY6fQSlA" -o "Wav2Lip/checkpoints/wav2lip.pth" --progress-bar
if %errorlevel% equ 0 (
    echo ✅ Wav2Lip model downloaded successfully
) else (
    echo ❌ Failed to download Wav2Lip model
)

REM Wav2Lip GAN model (higher quality)
echo ⬇️  Downloading Wav2Lip GAN model...
curl -L "https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?share=Eb56pIgZgnRAqCwhKKWKoLwBnha3qLh3KdN7bfXb9XnXfQ" -o "Wav2Lip/checkpoints/wav2lip_gan.pth" --progress-bar
if %errorlevel% equ 0 (
    echo ✅ Wav2Lip GAN model downloaded successfully
) else (
    echo ❌ Failed to download Wav2Lip GAN model
)

REM Face detection model for Wav2Lip
echo ⬇️  Downloading face detection model...
curl -L "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" -o "Wav2Lip/face_detection/detection/sfd/s3fd.pth" --progress-bar
if %errorlevel% equ 0 (
    echo ✅ Face detection model downloaded successfully
) else (
    echo ❌ Failed to download face detection model
)

echo.
echo ==================================================
echo 🎉 Model download completed!
echo.
echo 📊 Downloaded Models Summary:
echo   ├── Stable Diffusion Models:
echo   │   ├── ToonYou (cartoon style): models/toonyou_beta6.safetensors
echo   │   └── MeinaMix (anime style): models/meina_mix.safetensors
echo   ├── AnimateDiff Models:
echo   │   ├── Main model: models/animatediff_v1-5-pruned.ckpt
echo   │   └── Motion module: models/mm_sd_v15_v2.ckpt
echo   ├── LoRA Models:
echo   │   └── Cartoon LoRA: loras/animov.safetensors
echo   └── Wav2Lip Models:
echo       ├── Main model: Wav2Lip/checkpoints/wav2lip.pth
echo       ├── GAN model: Wav2Lip/checkpoints/wav2lip_gan.pth
echo       └── Face detection: Wav2Lip/face_detection/detection/sfd/s3fd.pth
echo.
echo 💡 Next steps:
echo   1. Make sure you have cloned Wav2Lip repository
echo   2. Edit .env file and add your API keys
echo   3. Run: python generate_cartoon_short.py --prompt "Your story prompt"
echo.
pause
