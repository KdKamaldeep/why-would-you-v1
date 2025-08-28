# 🎬 FPS Adjustment Guide

This guide explains how to adjust the frame rate (FPS) to control video playback speed.

## 📊 **Current FPS Settings**

The system has been updated with a **default FPS of 10** (reduced from 15) for slower, more natural video playback.

## ⚙️ **FPS Configuration Options**

### **1. Default FPS Values**
- **Current Default**: 10 FPS (slower, more natural)
- **Previous Default**: 15 FPS (faster)
- **Recommended Range**: 8-12 FPS for most content

### **2. FPS Effects on Video**
- **Lower FPS (8-10)**: Slower, more cinematic, better for storytelling
- **Higher FPS (12-15)**: Faster, more dynamic, better for action scenes
- **Very Low FPS (5-7)**: Very slow, dramatic effect
- **Very High FPS (15+)**: Fast, energetic, but may feel rushed

## 🔧 **How to Adjust FPS**

### **Method 1: Change Default in Code**
```python
# In VideoConfig class (src/core/generate_cartoon_short.py)
fps: int = 8  # Set to your preferred FPS
```

### **Method 2: Pass FPS Parameter**
```python
from src.core.generate_cartoon_short import VideoConfig

config = VideoConfig(
    prompt="Your story prompt",
    fps=8,  # Custom FPS setting
    duration=30
)
```

### **Method 3: Runtime FPS Override**
```python
# When calling video generation
generator = CartoonShortsGenerator(config)
# The FPS will be automatically applied to all video generation
```

## 🎯 **Recommended FPS Settings by Content Type**

### **Storytelling/Narrative Content**
- **FPS**: 8-10
- **Effect**: Slower, more contemplative, better for dialogue

### **Action/Dynamic Content**
- **FPS**: 10-12
- **Effect**: Balanced speed, good for movement

### **Fast-paced Content**
- **FPS**: 12-15
- **Effect**: Energetic, dynamic, good for action scenes

### **Cinematic/Dramatic Content**
- **FPS**: 6-8
- **Effect**: Very slow, dramatic, artistic

## 📈 **FPS Impact on File Size**

- **Lower FPS**: Smaller file sizes, faster processing
- **Higher FPS**: Larger file sizes, more processing time

## 🔄 **FPS and SVD Animation**

For SVD (Stable Video Diffusion) animations:
- **FPS affects motion perception**: Lower FPS makes motion appear smoother
- **SVD generates 24 frames**: Lower FPS means longer video duration from same frame count
- **Recommended SVD FPS**: 8-10 for natural motion

## ⚡ **Quick FPS Adjustment Examples**

```python
# Very slow, dramatic
config = VideoConfig(prompt="...", fps=6)

# Slow, natural
config = VideoConfig(prompt="...", fps=8)

# Balanced (current default)
config = VideoConfig(prompt="...", fps=10)

# Faster, dynamic
config = VideoConfig(prompt="...", fps=12)

# Fast, energetic
config = VideoConfig(prompt="...", fps=15)
```

## 🎬 **FPS and Audio Sync**

The FPS setting affects:
- **Video duration**: Lower FPS = longer video from same frame count
- **Audio sync**: System automatically adjusts to maintain audio-video synchronization
- **Scene timing**: Each scene's duration is calculated based on FPS

## 📝 **Notes**

- **FPS changes affect all video generation**: Both FFmpeg and SVD animations
- **Audio remains unchanged**: Only video speed is affected
- **Processing time**: Lower FPS = faster processing
- **Quality**: FPS doesn't affect visual quality, only playback speed

## 🔧 **Troubleshooting**

### **Video Too Fast**
- Reduce FPS to 8-10
- Check if SVD fps_id setting is too high

### **Video Too Slow**
- Increase FPS to 12-15
- Check if motion_bucket_id is too low

### **Audio-Video Sync Issues**
- Ensure FPS is consistent across all components
- Check that audio duration matches video duration
