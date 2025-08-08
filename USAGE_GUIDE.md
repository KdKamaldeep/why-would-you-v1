# 🎬 Cartoon Generator Usage Guide

## 🚀 Quick Start (Easiest)

### Windows Users:
1. **Double-click** `generate_cartoon.bat`
2. **Enter your story idea** when prompted
3. **Wait** for the magic to happen! ✨

### All Platforms:
```bash
python quick_start.py
```

## 🎯 Command Line Usage

### Basic Generation:
```bash
python simple_cartoon_generator.py --prompt "A baby lion opens a smoothie shop"
```

### With Options:
```bash
# Anime style, 45 seconds long
python simple_cartoon_generator.py --prompt "A robot learns to dance" --style anime --duration 45

# Check if everything is set up correctly
python simple_cartoon_generator.py --check-only
```

## 📚 Multiple Videos at Once

### Using Sample Prompts:
```bash
python batch_generate.py --samples
```

### Using Your Own Prompts:
1. Create a file called `prompts.txt`
2. Add one story idea per line:
   ```
   A cat becomes a superhero
   A tree that grows candy
   A friendly monster's first day at school
   ```
3. Run: `python batch_generate.py prompts.txt`

## 🎨 Available Options

### Styles:
- `cartoon` - Colorful, fun cartoon style (default)
- `anime` - Japanese anime/manga style

### Duration:
- Default: 30 seconds
- Range: 15-60 seconds recommended

## 📁 Output Structure

Your generated videos will be saved in:
```
output/
├── final_short.mp4          # Your finished cartoon!
├── script.json              # Generated story script
├── narration.wav            # AI-generated voice
├── scene_1.png             # Generated images
├── scene_2.png
├── scene_3.png
└── subtitles.srt           # Subtitle file
```

## 💡 Story Prompt Tips

### ✅ Good Prompts:
- "A baby elephant learns to paint with its trunk"
- "A friendly dragon opens a bakery in a village"
- "A young wizard's first day at magic school goes wrong"
- "A robot discovers the joy of gardening"

### ❌ Avoid:
- Too vague: "Something funny"
- Too complex: "A 10-character epic saga across multiple dimensions"
- Adult themes: Keep it family-friendly!

## 🔧 Troubleshooting

### "❌ OpenAI API key not configured"
1. Copy `config.env` to `.env`
2. Edit `.env` and add your OpenAI API key
3. Get key from: https://platform.openai.com/api-keys

### "❌ Models not found"
Run the model download script:
```bash
# Windows
download_models.bat

# Linux/Mac
bash download_models.sh
```

### "❌ Import error"
Install required packages:
```bash
pip install -r requirements.txt
```

### "Generation takes too long"
- First run is always slower (downloads/setup)
- Reduce duration: `--duration 20`
- Check your internet connection
- GPU will speed up image generation significantly

## 🎊 Example Commands

```bash
# Quick test
python simple_cartoon_generator.py --prompt "A cat learns to fly" --duration 20

# High-quality anime
python simple_cartoon_generator.py --prompt "Magical forest adventure" --style anime --duration 45

# Check setup
python simple_cartoon_generator.py --check-only

# Generate 5 sample videos
python batch_generate.py --samples
```

## 🆘 Need Help?

1. **Check requirements**: `python simple_cartoon_generator.py --check-only`
2. **Verify setup**: Make sure `.env` has your API keys
3. **Download models**: Run `download_models.bat`
4. **Check logs**: Look for error messages in the terminal

Happy cartoon making! 🎨✨
