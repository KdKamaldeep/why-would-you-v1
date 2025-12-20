# 🎬 SVD Overlapping Chunks Guide

This guide explains the new SVD (Stable Video Diffusion) overlapping chunk generation system that extends video generation beyond the 24-frame limit while maintaining visual consistency.

## 🎯 Problem Solved

### **Original Issue:**
- SVD has a hard limit of 24 frames per generation
- Previous system used simple chunking without overlap
- Visual inconsistencies between chunks
- Abrupt transitions between generated segments

### **New Solution:**
- **Overlapping chunk generation** with 4-8 frame overlap
- **Fixed parameters** across all chunks for consistency
- **Smooth transitions** between chunks
- **Reproducible results** with fixed seeds

## 🚀 How It Works

### **1. Overlapping Chunk Strategy**

```
Chunk 1: [Frame 1-24] (24 frames)
Chunk 2: [Frame 19-42] (24 frames, overlaps frames 19-24)
Chunk 3: [Frame 37-60] (24 frames, overlaps frames 37-42)
...
```

**Key Benefits:**
- **6-frame overlap** (configurable 4-8 range)
- **18 new frames** per chunk (24 - 6 overlap)
- **Smooth transitions** at chunk boundaries
- **Visual continuity** throughout the sequence

### **2. Fixed Parameters for Consistency**

All chunks use identical parameters:
- **Fixed seed** (base seed + chunk index)
- **Fixed motion_bucket_id** (motion intensity)
- **Fixed fps_id** (frame rate)
- **Fixed cond_aug** (conditioning augmentation)
- **Fixed resolution** and precision

### **3. Chunk Generation Process**

1. **Initial SVD Generation**: Generate first 24 frames
2. **Overlap Calculation**: Use last 6 frames as starting point
3. **Chunk Generation**: Generate next 24 frames with overlap
4. **Frame Selection**: Use frames 7-24 from each new chunk
5. **Repeat**: Continue until target frame count reached

## ⚙️ Configuration

### **VideoConfig Parameters**

```python
from src.core.generate_cartoon_short import VideoConfig

config = VideoConfig(
    prompt="Your story prompt",
    animator_type="svd",  # Enable SVD animation
    svd_chunked_generation=True,  # Enable overlapping chunks
    svd_overlap_frames=6,  # 4-8 frame overlap (default: 6)
    # ... other parameters
)
```

### **AnimationGenerator Parameters**

```python
from src.core.animation_generator import AnimationGenerator

animator = AnimationGenerator(
    width=768,
    height=1024,
    animator_type="svd",
    svd_chunked_generation=True,
    svd_overlap_frames=6  # Configurable overlap
)
```

## 📊 Performance Characteristics

### **Frame Generation Efficiency**

| Target Frames | Chunks Needed | Overlap Frames | New Frames/Chunk | Total Generated |
|---------------|---------------|----------------|------------------|-----------------|
| 60            | 3             | 6              | 18               | 54              |
| 120           | 6             | 6              | 18               | 108             |
| 240           | 12            | 6              | 18               | 216             |

### **Memory Usage**

- **Per chunk**: ~2-4GB VRAM (depending on resolution)
- **Processing**: Sequential chunk generation
- **Storage**: Temporary chunk directories cleaned up automatically

## 🎨 Visual Consistency Features

### **1. Parameter Consistency**

```python
# Fixed parameters across all chunks
fixed_seed = seed if seed is not None else 42
fixed_motion_bucket_id = motion_bucket_id
fixed_fps_id = fps_id
fixed_cond_aug = cond_aug

# Each chunk uses: fixed_seed + chunk_index
chunk_seed = fixed_seed + chunk_idx
```

### **2. Smooth Transitions**

- **Overlap frames** provide continuity
- **Gradual blending** between chunks
- **No abrupt visual jumps**
- **Consistent motion patterns**

### **3. Reproducible Results**

- **Fixed seeds** ensure identical results
- **Deterministic generation** across runs
- **Consistent quality** throughout sequence

## 🔧 Usage Examples

### **Basic Usage**

```python
from src.core.generate_cartoon_short import CartoonShortsGenerator, VideoConfig

# Configure for SVD with overlapping chunks
config = VideoConfig(
    prompt="A magical adventure story",
    duration=60,  # 60-second video
    animator_type="svd",
    svd_chunked_generation=True,
    svd_overlap_frames=6
)

# Generate video
generator = CartoonShortsGenerator(config)
output_path = generator.generate()
```

### **Advanced Configuration**

```python
# Custom overlap and parameters
config = VideoConfig(
    prompt="Epic battle scene",
    duration=120,  # 2-minute video
    animator_type="svd",
    svd_chunked_generation=True,
    svd_overlap_frames=8,  # Maximum overlap for smooth transitions
    # SVD parameters (passed to animation generator)
    motion_bucket_id=191,  # High motion
    fps_id=4,  # Fast motion
    cond_aug=0.05  # Strong conditioning
)
```

### **Direct Animation Generator Usage**

```python
from src.core.animation_generator import AnimationGenerator

# Create animator with overlapping chunks
animator = AnimationGenerator(
    width=768,
    height=1024,
    animator_type="svd",
    svd_chunked_generation=True,
    svd_overlap_frames=6
)

# Generate extended animation
result_dir = animator.animate_image(
    image_path="input_image.png",
    output_dir="output_frames",
    num_frames=300,  # 300 frames (12+ chunks)
    motion_bucket_id=127,
    fps_id=6,
    cond_aug=0.02,
    seed=42
)
```

## 🧪 Testing

### **Test Script**

Run the test script to verify the system:

```bash
python test_svd_overlapping_chunks.py
```

This will test:
- Different overlap configurations (4, 6, 8 frames)
- Visual consistency across multiple runs
- Frame count accuracy
- Parameter reproducibility

### **Manual Testing**

```python
# Test different overlap amounts
for overlap in [4, 6, 8]:
    animator = AnimationGenerator(
        svd_overlap_frames=overlap,
        svd_chunked_generation=True
    )
    # Generate test animation...
```

## 🔍 Troubleshooting

### **Common Issues**

#### **1. Inconsistent Visual Quality**
- **Solution**: Ensure fixed parameters across chunks
- **Check**: Verify seed, motion_bucket_id, fps_id are consistent

#### **2. Abrupt Transitions**
- **Solution**: Increase overlap_frames (try 8 instead of 6)
- **Check**: Verify overlap frames are being used correctly

#### **3. Memory Issues**
- **Solution**: Reduce resolution or use smaller overlap
- **Check**: Monitor VRAM usage during generation

#### **4. Frame Count Mismatch**
- **Solution**: Check effective_frames_per_chunk calculation
- **Check**: Verify chunk generation logic

### **Debug Information**

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Check chunk generation logs
# Look for: "Overlapping chunk X completed: Y frames added"
```

## 📈 Performance Optimization

### **1. Overlap Tuning**

- **4 frames**: Faster generation, minimal overlap
- **6 frames**: Balanced performance and quality (recommended)
- **8 frames**: Maximum smoothness, slower generation

### **2. Chunk Size Optimization**

- **24 frames**: Standard SVD limit
- **Effective 18 frames**: New content per chunk (with 6-frame overlap)
- **Memory efficient**: Sequential processing

### **3. Parameter Optimization**

- **motion_bucket_id**: 127 (medium) for balanced motion
- **fps_id**: 6 (normal) for smooth playback
- **cond_aug**: 0.02 (standard) for good quality

## 🎉 Benefits Summary

### **✅ Visual Consistency**
- Smooth transitions between chunks
- Consistent motion patterns
- No abrupt visual jumps

### **✅ Extended Duration**
- Generate videos beyond 24-frame limit
- Unlimited length capability
- Maintains quality throughout

### **✅ Reproducible Results**
- Fixed seeds ensure consistency
- Deterministic generation
- Reliable output quality

### **✅ Configurable System**
- Adjustable overlap amount (4-8 frames)
- Flexible parameter control
- Easy integration with existing pipeline

### **✅ Memory Efficient**
- Sequential chunk processing
- Automatic cleanup of temporary files
- Optimized VRAM usage

## 🔮 Future Enhancements

### **Planned Improvements**

1. **Adaptive Overlap**: Dynamic overlap based on motion intensity
2. **Quality Assessment**: Automatic quality checking between chunks
3. **Parallel Processing**: Multi-GPU chunk generation
4. **Advanced Blending**: AI-powered frame interpolation at boundaries

### **Integration Opportunities**

1. **ComfyUI Integration**: Direct workflow integration
2. **Batch Processing**: Multiple video generation
3. **Real-time Preview**: Live chunk generation preview
4. **Quality Metrics**: Automatic consistency scoring

---

This overlapping chunk system transforms SVD from a 24-frame limited tool into a powerful unlimited-length video generation system while maintaining the high quality and visual consistency that SVD is known for.
