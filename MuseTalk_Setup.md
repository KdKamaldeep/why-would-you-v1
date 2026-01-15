# 1. Clone MuseTalk
cd /workspace
git clone https://github.com/TMElyralab/MuseTalk.git
cd MuseTalk

# 2. Create conda environment
conda create -n MuseTalk python=3.10
conda activate MuseTalk

# 3. Install PyTorch
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install MMLab packages
pip install --no-cache-dir -U openmim
mim install mmengine
mim install "mmcv==2.0.1"
mim install "mmdet==3.1.0"
mim install "mmpose==1.1.0"

# 6. Download model weights
sh ./download_weights.sh

# 7. Create virtual environment (optional but recommended)
python3.10 -m venv venv
source venv/bin/activate
# Reinstall dependencies in venv if needed

# 8. Set environment variables
export MUSETALK_DIR=/workspace/MuseTalk
export MUSETALK_PYTHON=/workspace/MuseTalk/venv/bin/python
export MUSETALK_ENABLED=true