# Wav2Lip Integration Guide

This guide explains how to set up and use Wav2Lip for optional per-scene lip-syncing in the video generation pipeline.

## Overview

Wav2Lip is integrated as an **optional** post-process step that can be enabled per-scene. When enabled, it performs lip-syncing on the generated video using the scene's narration audio.

## Prerequisites

1. **Clone Wav2Lip repository**
   ```bash
   cd /workspace
   git clone https://github.com/Rudrabha/Wav2Lip.git
   cd Wav2Lip
   ```

2. **Install Wav2Lip dependencies in a virtual environment**
   ```bash
   # Create a virtual environment (REQUIRED - same pattern as LatentSync)
   cd /workspace/Wav2Lip
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

   **Note**: The pipeline expects the virtual environment at `/workspace/Wav2Lip/venv/bin/python` (same pattern as LatentSync uses `/workspace/LatentSync/venv/bin/python`)

3. **Download Wav2Lip checkpoint**
   ```bash
   # Create checkpoints directory
   mkdir -p checkpoints
   
   # Download the checkpoint (use wget or curl)
   cd checkpoints
   wget https://huggingface.co/Nekochu/Wav2Lip/blob/main/wav2lip_gan.pth
   # Or download manually from: https://github.com/Rudrabha/Wav2Lip/releases
   ```

4. **Verify installation**
   ```bash
   # Verify inference script exists
   ls -la /workspace/Wav2Lip/inference.py
   
   # Verify checkpoint exists
   ls -la /workspace/Wav2Lip/checkpoints/wav2lip_gan.pth
   
   # Verify virtual environment Python exists (REQUIRED)
   ls -la /workspace/Wav2Lip/venv/bin/python
   ```

## Configuration

Set environment variables to enable Wav2Lip:

```bash
# Enable Wav2Lip
export WAV2LIP_ENABLED=true

# Optional: Custom paths (defaults shown)
export WAV2LIP_DIR=/workspace/Wav2Lip
export WAV2LIP_CHECKPOINT=/workspace/Wav2Lip/checkpoints/wav2lip_gan.pth
export WAV2LIP_PYTHON=/workspace/Wav2Lip/venv/bin/python  # Virtual environment Python (default)
```

Or add to your `.env` or `config.env` file:
```env
WAV2LIP_ENABLED=true
WAV2LIP_DIR=/workspace/Wav2Lip
WAV2LIP_CHECKPOINT=/workspace/Wav2Lip/checkpoints/wav2lip_gan.pth
WAV2LIP_PYTHON=/workspace/Wav2Lip/venv/bin/python
```

**Important**: Wav2Lip runs in its own virtual environment (same pattern as LatentSync). The pipeline uses `/workspace/Wav2Lip/venv/bin/python` by default.

## Storyboard Configuration

Enable lip-syncing per-scene by adding `"lip_sync": true` to scene objects:

```json
{
  "stories": [
    {
      "title": "My Story",
      "scenes": [
        {
          "id": "scene_01",
          "duration": 7,
          "narration": "Did you know that...",
          "lip_sync": true,  // Enable Wav2Lip for this scene
          "visual_prompt": "..."
        },
        {
          "id": "scene_02",
          "duration": 7,
          "narration": "Another scene...",
          "lip_sync": false,  // Skip lip-syncing for this scene
          "visual_prompt": "..."
        }
      ]
    }
  ]
}
```

## Usage

1. **Enable Wav2Lip in environment**
   ```bash
   export WAV2LIP_ENABLED=true
   ```

2. **Run pipeline with scene(s) that have `"lip_sync": true`**
   ```bash
   python -m src.interfaces.batch_generate --storyboard storyboards/my_story.json
   ```

3. **Check logs for status**
   ```
   Scene 03: lip_sync=ON → SUCCESS (scene_03_lipsync.mp4)
   Scene 04: lip_sync=ON → FAILED (fallback to scene_04.mp4): Wav2Lip inference failed
   Scene 05: lip_sync=OFF → skipped
   ```

## How It Works

1. **Video Normalization**: Input video is normalized to:
   - Constant frame rate (matches scene FPS)
   - Even dimensions (required by Wav2Lip)
   - YUV420P pixel format

2. **Audio Normalization**: Input audio is normalized to:
   - Mono channel
   - 16kHz sample rate
   - PCM 16-bit format

3. **Wav2Lip Inference**: Normalized inputs are passed to Wav2Lip which:
   - Detects faces in the video
   - Generates lip movements synchronized with audio
   - Outputs lip-synced video

4. **Integration**: 
   - If successful: lip-synced video replaces original in concatenation
   - If failed: original video is used (fallback)
   - If disabled: original video is used (skipped)

## Output Files

- **Lip-synced video**: `{scene_dir}/scene_{N}_lipsync.mp4`
- **Original video**: `{scene_dir}/scene_{N}.mp4` (used if lip-sync fails)
- **Logs**: Check console output for per-scene status

## Troubleshooting

### Wav2Lip not running
- Check `WAV2LIP_ENABLED=true` is set
- Verify paths: `WAV2LIP_DIR` and `WAV2LIP_CHECKPOINT`
- **Verify virtual environment Python exists**: `/workspace/Wav2Lip/venv/bin/python`
- Check file permissions on checkpoint and Python executable
- If using custom paths, set `WAV2LIP_PYTHON` environment variable

### Lip-sync fails
- Verify video contains a face
- Check video/audio durations match
- Review logs for specific error messages
- Verify Wav2Lip dependencies are installed correctly

### Performance
- Wav2Lip can be slower than LatentSync
- Consider enabling only for scenes that need it
- GPU acceleration helps significantly

## Notes

- Wav2Lip is **Route A** (alternative to LatentSync)
- If `WAV2LIP_ENABLED=false` but `lip_sync=true`, the pipeline falls back to LatentSync (if available)
- Original videos are preserved; lip-synced videos are additional outputs
- All temporary files are cleaned up automatically

