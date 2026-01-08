# MuseTalk Setup Guide

This guide explains how to set up MuseTalk as an additional lip-sync option alongside Wav2Lip.

## Prerequisites

- Python 3.10
- CUDA 11.8 (for GPU support)
- FFmpeg installed and in PATH
- Conda (recommended) or virtual environment

## Installation Steps

### 1. Clone MuseTalk Repository

```bash
cd /workspace
git clone https://github.com/TMElyralab/MuseTalk.git
cd MuseTalk
```

### 2. Create Conda Environment

```bash
conda create -n MuseTalk python=3.10
conda activate MuseTalk
```

### 3. Install PyTorch 2.0.1

**Using pip (recommended):**
```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

**Using conda:**
```bash
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Install MMLab Packages

MuseTalk requires MMLab packages. Install them using `mim`:

```bash
pip install --no-cache-dir -U openmim
mim install mmengine
mim install "mmcv==2.0.1"
mim install "mmdet==3.1.0"
mim install "mmpose==1.1.0"
```

### 6. Set Up FFmpeg

FFmpeg should already be installed. Verify it's in your PATH:

```bash
which ffmpeg
ffmpeg -version
```

If FFmpeg is not in PATH, set the `FFMPEG_PATH` environment variable:

```bash
export FFMPEG_PATH=/path/to/ffmpeg
```

### 7. Download Model Weights

**For Linux:**
```bash
cd /workspace/MuseTalk
sh ./download_weights.sh
```

**For Windows:**
```bash
cd /workspace/MuseTalk
download_weights.bat
```

Alternatively, manually download weights from the MuseTalk repository and place them in the `models` directory.

### 8. Create Virtual Environment (Optional but Recommended)

If you prefer using a virtual environment instead of conda:

```bash
cd /workspace/MuseTalk
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install --no-cache-dir -U openmim
mim install mmengine
mim install "mmcv==2.0.1"
mim install "mmdet==3.1.0"
mim install "mmpose==1.1.0"
```

## Configuration

### Environment Variables

Set the following environment variables to configure MuseTalk paths:

```bash
export MUSETALK_DIR=/workspace/MuseTalk
export MUSETALK_PYTHON=/workspace/MuseTalk/venv/bin/python
```

Or add them to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.):

```bash
echo 'export MUSETALK_DIR=/workspace/MuseTalk' >> ~/.bashrc
echo 'export MUSETALK_PYTHON=/workspace/MuseTalk/venv/bin/python' >> ~/.bashrc
source ~/.bashrc
```

### Verify Installation

Test MuseTalk installation:

```bash
cd /workspace/MuseTalk
python scripts/inference.py --help
```

## Usage in Code

MuseTalk can be used as an alternative to Wav2Lip. The code will automatically detect which lip-sync method to use based on configuration.

### Example Usage

```python
from src.core.sync_musetalk import lipsync_musetalk

success = lipsync_musetalk(
    in_video_mp4="input_video.mp4",
    in_audio_wav="input_audio.wav",
    out_video_mp4="output_lipsync.mp4",
    fps=24,
    musetalk_dir="/workspace/MuseTalk",  # Optional, uses env var if not provided
    python_cmd="/workspace/MuseTalk/venv/bin/python",  # Optional
    bbox_shift=0,  # Optional, controls mask region (affects mouth openness)
    device="cuda"  # Optional, "cuda" or "cpu"
)
```

## Parameters

- **bbox_shift**: Controls the bounding box shift parameter. Higher values can affect mouth openness. Default: 0
- **device**: Device to use for inference. Options: "cuda" (GPU) or "cpu". Default: "cuda"

## Troubleshooting

### Issue: "MuseTalk Python not found"

**Solution:** Ensure the virtual environment is created and the Python path is correct:
```bash
ls -la /workspace/MuseTalk/venv/bin/python
```

### Issue: "MuseTalk inference script not found"

**Solution:** Verify the inference script exists:
```bash
ls -la /workspace/MuseTalk/scripts/inference.py
```

### Issue: Model weights not found

**Solution:** Run the download script or manually download weights:
```bash
cd /workspace/MuseTalk
sh ./download_weights.sh
```

### Issue: CUDA out of memory

**Solution:** Use CPU mode or reduce video resolution:
```python
lipsync_musetalk(..., device="cpu")
```

## Differences from Wav2Lip

- MuseTalk is generally faster and produces higher quality results
- MuseTalk requires more dependencies (MMLab packages)
- MuseTalk has different parameter options (bbox_shift vs pads)

## References

- MuseTalk GitHub: https://github.com/TMElyralab/MuseTalk
- MuseTalk Paper: Check the repository for paper links

