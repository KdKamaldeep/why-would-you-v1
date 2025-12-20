# SVD Performance Optimization Guide

## Overview
This guide helps optimize SVD (Stable Video Diffusion) performance for different GPU configurations, especially for the A4000 GPU.

## Expected Performance by GPU

### A40 (48GB VRAM) - **YOUR GPU**
- **Expected**: 8-15 seconds for 25 frames
- **Optimized**: 5-10 seconds
- **Memory usage**: 12-20GB VRAM (plenty of headroom)

### A4000 (16GB VRAM)
- **Expected**: 15-30 seconds for 25 frames
- **Current**: ~60 seconds (needs optimization)
- **Target**: 15-20 seconds with optimizations

### RTX 4090 (24GB VRAM)
- **Expected**: 8-15 seconds for 25 frames
- **Optimized**: 5-10 seconds

### RTX 3080/3090 (10-24GB VRAM)
- **Expected**: 20-40 seconds for 25 frames
- **Optimized**: 15-25 seconds

## Performance Optimizations Applied

### 1. Memory Management
```python
# Clear GPU cache before and after generation
torch.cuda.empty_cache()
gc.collect()
```

### 2. Model Optimizations
- **FP16 Precision**: Reduces memory usage and increases speed
- **Attention Slicing**: Reduces memory usage for large models
- **XFormers**: Significant speed boost (requires `pip install xformers`)
- **Model Compilation**: PyTorch 2.0+ compilation for additional speed

### 3. Generation Parameters
```python
# Optimized for A4000
decode_chunk_size = 4  # Reduced from 8 for faster generation
torch.no_grad()  # Disable gradient computation
```

## Installation Requirements

### Required Packages
```bash
pip install diffusers transformers accelerate
pip install xformers  # For significant speed boost
```

### Optional but Recommended
```bash
pip install torch==2.0+  # For model compilation
```

## Configuration Options

### For A40 (48GB VRAM) - **YOUR CONFIGURATION**
```python
# Aggressive settings for A40 (48GB VRAM)
enable_memory_efficient_attention = False  # Not needed with 48GB VRAM
enable_xformers = True
use_fp16 = True
enable_model_cpu_offload = False  # A40 has massive VRAM
decode_chunk_size = 8  # Higher for faster generation
```

### For A4000 (16GB VRAM)
```python
# Optimal settings for A4000
enable_memory_efficient_attention = True
enable_xformers = True
use_fp16 = True
enable_model_cpu_offload = False  # A4000 has enough VRAM
decode_chunk_size = 4
```

### For Lower VRAM GPUs (8GB or less)
```python
# Conservative settings for lower VRAM
enable_memory_efficient_attention = True
enable_xformers = True
use_fp16 = True
enable_model_cpu_offload = True  # Enable for memory-constrained systems
decode_chunk_size = 2
```

### For High VRAM GPUs (24GB+)
```python
# Aggressive settings for high VRAM
enable_memory_efficient_attention = False  # Not needed
enable_xformers = True
use_fp16 = True
enable_model_cpu_offload = False
decode_chunk_size = 8  # Can use higher values
```

## Troubleshooting Performance Issues

### 1. Slow Generation (>60 seconds for 25 frames)
**Causes:**
- Missing xformers installation
- Using CPU instead of GPU
- Memory fragmentation
- Suboptimal decode_chunk_size

**Solutions:**
```bash
# Install xformers
pip install xformers

# Check GPU usage
nvidia-smi

# Restart Python to clear memory
```

### 2. Out of Memory Errors
**Causes:**
- Too high decode_chunk_size
- FP32 instead of FP16
- No attention slicing

**Solutions:**
```python
# Reduce decode_chunk_size
decode_chunk_size = 2

# Enable memory optimizations
enable_model_cpu_offload = True
enable_memory_efficient_attention = True
```

### 3. Model Loading Issues
**Causes:**
- Missing model files
- Corrupted downloads
- Insufficient disk space

**Solutions:**
```bash
# Clear and redownload models
rm -rf models/svd_xt_1_1
# Restart the application to trigger download
```

## Performance Monitoring

### GPU Usage Monitoring
```bash
# Monitor GPU usage during generation
watch -n 1 nvidia-smi

# Check memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### Logging Performance Metrics
The optimized SVD animator now logs:
- Generation time
- Frames per second
- Memory usage
- Optimization status

## Advanced Optimizations

### 1. Model Quantization (Experimental)
```python
# For extreme memory constraints
from diffusers import StableVideoDiffusionPipeline
import torch

pipeline = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16,
    variant="fp16",
    load_in_8bit=True  # Requires bitsandbytes
)
```

### 2. Custom CUDA Kernels
```python
# For maximum performance (requires custom compilation)
pipeline.enable_custom_kernels()
```

### 3. Batch Processing
```python
# Process multiple images simultaneously
# (Requires sufficient VRAM)
```

## Expected Results After Optimization

### A40 Performance Targets - **YOUR TARGETS**
- **First run**: 15-25 seconds (model loading overhead)
- **Subsequent runs**: 5-10 seconds
- **Memory usage**: 12-20GB VRAM
- **Throughput**: 2.5-5.0 frames/second

### A4000 Performance Targets
- **First run**: 25-35 seconds (model loading overhead)
- **Subsequent runs**: 15-20 seconds
- **Memory usage**: 8-12GB VRAM
- **Throughput**: 1.25-1.67 frames/second

### Performance Checklist
- [ ] XFormers installed and working
- [ ] FP16 precision enabled
- [ ] Attention slicing enabled
- [ ] Model compilation working
- [ ] GPU memory sufficient
- [ ] No CPU offloading needed

## Monitoring Commands

### Real-time Performance
```bash
# Monitor during generation
nvidia-smi dmon -s pucvmet -d 1

# Check if xformers is working
python -c "import xformers; print('XFormers available')"
```

### Memory Analysis
```bash
# Check memory fragmentation
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv

# Monitor memory during generation
watch -n 0.5 'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits'
```

## Conclusion

With these optimizations, your A40 should achieve:
- **5-10 seconds** for 25 frames (instead of 60+ seconds)
- **2.5-5.0 frames/second** generation speed
- **Stable memory usage** without OOM errors
- **Consistent performance** across multiple generations

The key optimizations are:
1. **XFormers installation** (biggest impact)
2. **FP16 precision** (memory and speed)
3. **Optimized decode_chunk_size** (balanced for A4000)
4. **Memory management** (cache clearing)
5. **Model compilation** (PyTorch 2.0+)
