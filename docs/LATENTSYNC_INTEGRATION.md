# LatentSync Lip Sync Integration

This document describes the LatentSync lip synchronization integration into the video generation pipeline.

## Overview

LatentSync has been integrated as an optional post-processing step that runs after WAN video generation and Coqui audio generation, but before final FFmpeg stitching. It enables realistic lip synchronization for scenes with characters speaking.

## Configuration

Add these environment variables to your `.env` file:

```bash
# Enable/disable LatentSync (default: false)
LATENTSYNC_ENABLED=true

# Path to LatentSync model directory
LATENTSYNC_MODEL_PATH=/path/to/latentsync/models

# Device to use (cuda or cpu)
LATENTSYNC_DEVICE=cuda

# Face detection mode (auto, manual, none)
LATENTSYNC_FACE_MODE=auto

# Minimum face size in pixels for detection
LATENTSYNC_MIN_FACE_SIZE=64

# Use half precision (faster, less memory)
LATENTSYNC_FP16=true

# Save debug frames during processing
LATENTSYNC_DEBUG_FRAMES=false
```

## Scene JSON Schema

To enable lip sync for a scene, add the following fields to your scene object:

```json
{
  "id": "scene_01",
  "duration": 8,
  "num_frames": 96,
  "visual_prompt": "Close-up portrait of a person speaking to camera, realistic lighting",
  "narration": "Hello, this is a test.",
  "lip_sync": true,
  "talking_head": true,
  "speaker": {
    "id": "char_01",
    "name": "Narrator",
    "gender": "female",
    "voice_id": "coqui_voice_female_01"
  },
  "dialogue": "Hello, this is a test.",
  "character_reference_image": "path/to/character_face.png"
}
```

### Field Descriptions

- **`lip_sync`** (boolean, required): Set to `true` to enable lip sync for this scene
- **`talking_head`** (boolean, optional): Indicates the scene is a talking head shot (helps with validation)
- **`speaker`** (object, optional): Speaker information
  - `id`: Unique character identifier
  - `name`: Character name
  - `gender`: "male" or "female"
  - `voice_id`: Coqui voice identifier
- **`dialogue`** (string, optional): Text used for speaking (defaults to `narration` if not provided)
- **`character_reference_image`** (string, optional): Path to reference image for character face

## Pipeline Integration

The LatentSync step is integrated into the main pipeline as follows:

1. **WAN generates scene video** → `scene_1.mp4`
2. **Coqui generates narration audio** → `audio_scene_1.wav`
3. **Video is synced to audio duration** (if needed)
4. **LatentSync processes lip sync** (if `lip_sync: true`) → `scene_1_lipsync.mp4`
5. **Final video is used for stitching** → Uses lip-synced version if available, otherwise original

### Validation

Before video generation, the pipeline validates:
- If `lip_sync: true`, checks if `visual_prompt` suggests face-visible framing
- Warns if face keywords are missing and `talking_head` is not set
- Continues with generation but logs warnings

### Face Detection

During LatentSync processing:
- If `face_mode: auto`, attempts to detect faces in the video
- Warns if no face is detected but continues processing
- Falls back to original video if LatentSync fails

## Usage

### In Storyboard JSON

See `storyboards/example_lipsync.json` for a complete example.

### Standalone Testing

Use the CLI utility to test LatentSync on individual video/audio pairs:

```bash
python scripts/run_latentsync.py \
  --video_in scene.mp4 \
  --audio_in narration.wav \
  --out output_lipsync.mp4 \
  --face_mode auto \
  --target_fps 24
```

### Command Line Options

- `--video_in`: Input video file (required)
- `--audio_in`: Input audio file (required)
- `--out`: Output video file (required)
- `--face_mode`: Face detection mode (auto/manual/none, default: auto)
- `--target_fps`: Target FPS (default: 24)
- `--keep_fps`: Keep original video FPS instead of target_fps
- `--character_ref`: Optional character reference image path
- `--model_path`: Override LATENTSYNC_MODEL_PATH
- `--device`: Override LATENTSYNC_DEVICE

## Implementation Details

### Module Structure

- **`src/core/latentsync.py`**: Main LatentSync wrapper class
  - `LatentSyncRunner`: Handles video/audio processing
  - `get_latentsync_runner()`: Factory function for creating runner from config

### Integration Points

1. **`src/core/generate_cartoon_short.py`**:
   - Validates lip_sync scenes before video generation
   - Applies LatentSync after video/audio generation
   - Falls back to original video on errors

2. **`src/core/video_processor.py`**:
   - Normalizes FPS across clips if needed
   - Handles concatenation of mixed lip-synced and non-lip-synced videos

### File Naming Convention

- Original WAN video: `scene_1.mp4`
- Lip-synced video: `scene_1_lipsync.mp4`
- Final video used: Lip-synced version if available, otherwise original

## Safeguards

1. **Face Detection**: Warns if no face detected in video when `lip_sync: true`
2. **Duration Matching**: Validates audio/video duration alignment
3. **Error Handling**: Falls back to original video if LatentSync fails
4. **FPS Normalization**: Automatically normalizes FPS if clips differ
5. **Validation**: Checks visual_prompt for face-visible framing keywords

## Troubleshooting

### LatentSync Not Running

- Check `LATENTSYNC_ENABLED=true` in `.env`
- Verify `LATENTSYNC_MODEL_PATH` points to valid model directory
- Check logs for availability warnings

### No Face Detected

- Ensure `visual_prompt` includes face-visible keywords (close-up, medium shot, portrait, face, head)
- Set `talking_head: true` in scene
- Check that video actually contains a face (may need to adjust WAN prompt)

### Lip Sync Quality Issues

- Ensure video has clear face visibility
- Use `character_reference_image` for better character consistency
- Adjust `LATENTSYNC_FACE_MODE` if needed
- Check audio quality and duration matching

### FPS Mismatch

- Pipeline automatically normalizes FPS if clips differ
- Set `keep_fps: true` in LatentSync to maintain original FPS
- Target FPS defaults to 24fps (configurable)

## Future Enhancements

- Support for multiple characters in one scene
- Automatic face detection and cropping
- Integration with other lip sync models (modular design allows swapping)
- Batch processing optimizations

## Notes

- LatentSync is optional - pipeline works normally when disabled
- Non-lip-sync scenes are unaffected
- Audio synchronization is maintained throughout
- Original videos are preserved (lip-synced versions are additional outputs)

