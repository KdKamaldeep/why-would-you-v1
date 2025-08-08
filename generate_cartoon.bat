@echo off
REM Simple Cartoon Generator - Windows Batch Script

echo 🎨 Cartoon Generator
echo ==================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist .env (
    echo ❌ .env file not found
    echo Please copy config.env to .env and add your API keys
    pause
    exit /b 1
)

REM Check if models directory exists
if not exist models (
    echo ❌ Models not downloaded
    echo Please run download_models.bat first
    pause
    exit /b 1
)

echo ✅ Setup looks good!
echo.

REM Get user input
set /p prompt="📝 Enter your story idea: "

if "%prompt%"=="" (
    echo ❌ Please provide a story prompt!
    pause
    exit /b 1
)

echo.
echo 🎬 Generating cartoon for: "%prompt%"
echo ⏳ This may take several minutes...
echo.

REM Run the generator
python simple_cartoon_generator.py --prompt "%prompt%"

echo.
echo ✨ Generation complete! Check the output folder.
pause
