# 🎭 Face-Based Image Generation Guide

This guide explains how to use the face-based image generation feature that allows you to generate images using faces from existing images with diffusion models.

## 🚀 Overview

The face-based image generation system supports multiple approaches:

1. **ControlNet with Face Detection** - Uses face landmarks to control generation
2. **IP-Adapter** - Uses reference face images as prompts
3. **Enhanced Prompting** - Uses face-aware prompts with the base diffusion model

## 📋 Prerequisites

### Required Dependencies
```bash
pip install -r requirements.txt
```

### Optional Dependencies (for advanced features)
- **ControlNet models** - For precise face control
- **IP-Adapter models** - For image-based prompting
- **MediaPipe** - For advanced face detection (included in requirements)

## 🎯 Quick Start

### 1. Basic Usage (Interactive Mode)
```bash
python -m src.interfaces.face_generator_interface
```

This will start an interactive interface where you can:
- Generate single images with faces
- Create videos with face swapping
- Configure generation parameters

### 2. Command Line Usage

#### Generate Single Image
```bash
python -m src.interfaces.face_generator_interface \
    --mode single \
    --prompt "A cartoon character in a magical forest" \
    --face-image "path/to/face_reference.jpg" \
    --output "my_generated_image.png"
```

#### Generate Video
```bash
python -m src.interfaces.face_generator_interface \
    --mode video \
    --prompt "A cartoon character exploring a magical world" \
    --face-image "path/to/face_reference.jpg" \
    --frames 60 \
    --output "my_face_video.mp4"
```

## 🔧 Advanced Configuration

### Using ControlNet for Better Face Control

1. **Download ControlNet Model**
   ```bash
   # Example: Download face control model
   git clone https://huggingface.co/lllyasviel/control_v11p_sd15_openpose models/controlnet_openpose
   ```

2. **Use with ControlNet**
   ```python
   from src.core.face_image_generator import FaceImageGenerator
   
   generator = FaceImageGenerator(
       controlnet_path="models/controlnet_openpose"
   )
   ```

### Using IP-Adapter for Image-Based Prompting

1. **Download IP-Adapter**
   ```bash
   # Download IP-Adapter model
   git clone https://huggingface.co/h94/IP-Adapter models/ip_adapter
   ```

2. **Use with IP-Adapter**
   ```python
   from src.core.face_image_generator import FaceImageGenerator
   
   generator = FaceImageGenerator(
       ip_adapter_path="models/ip_adapter"
   )
   ```

## 🎨 Generation Parameters

### Core Parameters
- **`prompt`** - Text description of what to generate
- **`face_image_path`** - Path to the reference face image
- **`negative_prompt`** - What to avoid in generation
- **`num_inference_steps`** - Number of denoising steps (default: 30)
- **`guidance_scale`** - How closely to follow the prompt (default: 7.5)
- **`strength`** - How much the face influences generation (0.0-1.0, default: 0.8)

### Example with Custom Parameters
```python
result = generator.generate_with_face(
    prompt="A superhero character in a futuristic city",
    face_image_path="my_face.jpg",
    negative_prompt="blurry, low quality, distorted",
    num_inference_steps=50,
    guidance_scale=8.0,
    strength=0.9
)
```

## 🎬 Video Generation

### Creating Face Swap Videos
```python
success = generator.generate_face_swap_video(
    prompt="A character going on an adventure",
    face_image_path="reference_face.jpg",
    background_prompt="blurry background",
    num_frames=30,
    output_path="adventure_video.mp4"
)
```

### Video Parameters
- **`num_frames`** - Number of frames to generate (default: 30)
- **`fps`** - Frames per second (default: 15)
- **`output_path`** - Where to save the video

## 🔍 Face Detection

The system supports multiple face detection methods:

### MediaPipe (Recommended)
- More accurate face detection
- Provides facial landmarks
- Better for ControlNet integration

### OpenCV (Fallback)
- Basic face detection
- Works without additional dependencies
- Suitable for simple use cases

### Face Detection Example
```python
# Detect face in an image
face_data = generator.detect_face("my_image.jpg")
if face_data:
    face_region, landmarks = face_data
    print(f"Face detected with {len(landmarks)} landmarks")
```

## 🎯 Best Practices

### 1. Face Reference Images
- Use clear, well-lit face photos
- Ensure the face is clearly visible
- Avoid extreme angles or expressions
- Recommended size: 512x512 pixels or larger

### 2. Prompts
- Be specific about the desired style and setting
- Include facial feature descriptions
- Use negative prompts to avoid unwanted elements
- Example: "A cartoon character with detailed facial features in a magical forest, high quality, detailed"

### 3. Generation Settings
- Start with default parameters
- Adjust `strength` to control face influence
- Use higher `guidance_scale` for more prompt adherence
- Increase `num_inference_steps` for better quality (slower)

### 4. Model Selection
- Use cartoon/anime models for cartoon-style output
- Use realistic models for photorealistic results
- Consider using LoRA models for specific styles

## 🛠️ Troubleshooting

### Common Issues

#### 1. No Face Detected
```
❌ No face detected in the input image
```
**Solutions:**
- Ensure the face is clearly visible
- Try a different face image
- Check if the image format is supported

#### 2. Diffusion Pipeline Not Initialized
```
❌ Diffusion pipeline not initialized
```
**Solutions:**
- Check if diffusion model is available
- Install required dependencies
- Verify model path is correct

#### 3. Poor Quality Results
**Solutions:**
- Increase `num_inference_steps`
- Adjust `guidance_scale`
- Use better quality face reference
- Try different prompts

#### 4. Memory Issues
**Solutions:**
- Reduce image resolution
- Use CPU instead of GPU
- Close other applications
- Use smaller models

### Performance Optimization

#### For GPU Users
```python
# Enable memory efficient attention
generator = FaceImageGenerator(device='cuda')
```

#### For CPU Users
```python
# Use CPU for generation
generator = FaceImageGenerator(device='cpu')
```

## 📁 Output Structure

Generated files are saved in:
```
output/face_generated/
├── face_generated_[original_name].png
├── face_video_[original_name].mp4
└── [custom_output_names].png/mp4
```

## 🔗 Integration with Cartoon Generation System

The face-based image generator is now fully integrated with your cartoon generation system through storyboards:

### Using Storyboard with Character Faces

Create a storyboard JSON file with character face specifications:

```json
{
  "title": "The Lion's Smoothie Shop Adventure",
  "description": "A brave lion opens a smoothie shop in the jungle",
  "scene_duration": 8,
  "cast": [
    {
      "name": "Lion",
      "role": "main character",
      "face": "source-face-images/sardar.png"
    },
    {
      "name": "Robot",
      "role": "helper",
      "face": "source-face-images/robot_face.jpg"
    },
    {
      "name": "Princess",
      "role": "customer"
    }
  ],
  "scenes": [
    {
      "title": "Lion's Dream",
      "visual_prompt": "A lion standing in front of an empty shop space",
      "subtitle": "Once upon a time, a brave lion had a dream...",
      "characters": ["Lion"]
    }
  ]
}
```

### Generate Cartoon with Character Faces

```bash
# Generate cartoon using storyboard with character faces
python -m src.interfaces.simple_cartoon_generator \
    --prompt "A brave lion opens a smoothie shop" \
    --storyboard storyboards/example_with_faces.json
```

### How It Works

1. **Cast Array**: Characters with a `face` field use face-based generation
2. **Auto-Generation**: Characters without a `face` field use standard generation
3. **Mixed Approach**: You can mix custom faces and auto-generated faces
4. **Automatic Detection**: The system automatically detects faces in reference images

## 🎨 Creative Examples

### Example 1: Character Creation
```bash
python -m src.interfaces.face_generator_interface \
    --mode single \
    --prompt "A brave warrior with detailed armor, fantasy setting, epic lighting" \
    --face-image "my_face.jpg" \
    --negative-prompt "modern, casual, blurry"
```

### Example 2: Animation Sequence
```bash
python -m src.interfaces.face_generator_interface \
    --mode video \
    --prompt "A character discovering a magical portal, dramatic lighting, fantasy world" \
    --face-image "character_face.jpg" \
    --frames 45 \
    --negative-prompt "realistic, modern, urban"
```

### Example 3: Style Transfer
```bash
python -m src.interfaces.face_generator_interface \
    --mode single \
    --prompt "Anime style character, vibrant colors, detailed eyes, fantasy background" \
    --face-image "realistic_face.jpg" \
    --strength 0.7
```

## 📚 Additional Resources

- [Stable Diffusion Documentation](https://huggingface.co/docs/diffusers/index)
- [ControlNet Guide](https://github.com/lllyasviel/ControlNet)
- [IP-Adapter Documentation](https://github.com/tencent-ailab/IP-Adapter)
- [MediaPipe Face Detection](https://google.github.io/mediapipe/solutions/face_detection)

## 🤝 Contributing

To improve the face-based image generation:

1. Test with different face detection methods
2. Experiment with various ControlNet models
3. Optimize generation parameters
4. Add support for multiple faces
5. Implement face expression control

---

**Happy face-based image generation! 🎭✨**
