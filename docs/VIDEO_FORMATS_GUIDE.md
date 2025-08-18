# 🎬 Video Formats Guide

This guide explains how to use the new video format functionality in the Cartoon Shorts Generator.

## 📐 Supported Video Formats

### 1. YouTube Shorts (9:16 Aspect Ratio)
- **Dimensions**: 768x1024 pixels
- **Aspect Ratio**: 0.75:1 (9:16)
- **Perfect for**: TikTok, Instagram Reels, YouTube Shorts, Snapchat
- **Default format**: Yes

### 2. Normal Video (16:9 Aspect Ratio)
- **Dimensions**: 1920x1080 pixels
- **Aspect Ratio**: 1.78:1 (16:9)
- **Perfect for**: YouTube, Vimeo, general video platforms, presentations

## 🚀 How to Use

### Command Line Interface

#### Basic Usage (YouTube Shorts - Default)
```bash
# Create a YouTube Shorts video (default)
python main.py "A dragon learns to bake cookies"

# Explicitly specify YouTube Shorts format
python main.py "A dragon learns to bake cookies" --video-format shorts
```

#### Normal Video Format
```bash
# Create a normal 16:9 video
python main.py "A dragon learns to bake cookies" --video-format normal
```

#### Advanced Options
```bash
# Create a 60-second normal video
python main.py "A dragon learns to bake cookies" --video-format normal --duration 60

# Specify custom output directory
python main.py "A dragon learns to bake cookies" --video-format shorts --output my_videos
```

### Programmatic Usage

```python
from src.core.generate_cartoon_short import CartoonShortsGenerator, VideoConfig

# YouTube Shorts configuration
shorts_config = VideoConfig(
    prompt="A dragon learns to bake cookies",
    duration=30,
    video_format="shorts",  # 768x1024
    output_path="output/shorts"
)

# Normal video configuration
normal_config = VideoConfig(
    prompt="A dragon learns to bake cookies",
    duration=30,
    video_format="normal",  # 1920x1080
    output_path="output/normal"
)

# Generate videos
shorts_generator = CartoonShortsGenerator(shorts_config)
normal_generator = CartoonShortsGenerator(normal_config)

shorts_output = shorts_generator.generate()
normal_output = normal_generator.generate()
```

### Direct Module Usage

```bash
# Using the core module directly
python -m src.core.generate_cartoon_short --prompt "A dragon learns to bake cookies" --video-format shorts
python -m src.core.generate_cartoon_short --prompt "A dragon learns to bake cookies" --video-format normal
```

## 🎨 Technical Details

### Image Generation
- **YouTube Shorts**: Images generated at 768x1024 pixels
- **Normal Video**: Images generated at 1920x1080 pixels
- **Animation**: All animation effects automatically adapt to the chosen dimensions
- **Quality**: Both formats maintain the same high-quality cartoon style

### Video Processing
- **Encoding**: HEVC (H.265) for optimal file size
- **FPS**: 15 fps (configurable)
- **Audio**: AAC encoding with narration and background music
- **Subtitles**: Automatically positioned for optimal readability in both formats

### File Output
- **YouTube Shorts**: `final_short.mp4`
- **Normal Video**: `final_video.mp4`
- **Location**: Specified output directory

## 📱 Platform Recommendations

### YouTube Shorts (9:16)
- **TikTok**: Perfect fit for vertical scrolling
- **Instagram Reels**: Native support for 9:16
- **YouTube Shorts**: Optimized for mobile viewing
- **Snapchat**: Ideal for Stories format
- **Facebook Stories**: Compatible format

### Normal Video (16:9)
- **YouTube**: Standard format for regular videos
- **Vimeo**: Professional video platform
- **LinkedIn**: Business presentations
- **Twitter**: Video tweets
- **General web**: Universal compatibility

## 🔧 Customization

### Custom Dimensions
You can modify the dimensions by editing the `VideoConfig` class:

```python
@dataclass
class VideoConfig:
    def __post_init__(self):
        if self.video_format.lower() == "shorts":
            self.width = 768
            self.height = 1024
        elif self.video_format.lower() == "normal":
            self.width = 1920
            self.height = 1080
        # Add custom formats here
        elif self.video_format.lower() == "square":
            self.width = 1080
            self.height = 1080
```

### Adding New Formats
To add a new video format:

1. Update the `VideoConfig.__post_init__()` method
2. Add the format to the CLI argument choices
3. Update the output filename logic
4. Test with the new dimensions

## 🧪 Testing

Use the provided test script to verify both formats work correctly:

```bash
python test_video_formats.py
```

This will create test videos in both formats and verify the functionality.

## 📊 Performance Considerations

### File Sizes
- **YouTube Shorts**: Smaller files due to lower resolution
- **Normal Video**: Larger files but higher quality
- **Processing Time**: Normal videos may take longer due to higher resolution

### Memory Usage
- **YouTube Shorts**: Lower memory requirements
- **Normal Video**: Higher memory usage during processing
- **Recommendation**: Use YouTube Shorts for faster processing, Normal for higher quality

## 🎯 Best Practices

1. **Choose the right format** for your target platform
2. **Test both formats** to see which works better for your content
3. **Consider your audience** - mobile users prefer vertical, desktop users prefer horizontal
4. **Optimize for platform** - each platform has specific requirements
5. **Batch processing** - you can create both formats from the same prompt for maximum reach

## 🐛 Troubleshooting

### Common Issues

1. **Memory errors with normal format**: Reduce batch size or use YouTube Shorts format
2. **Slow processing**: Normal format takes longer due to higher resolution
3. **File size too large**: Use YouTube Shorts format or reduce duration
4. **Aspect ratio issues**: Ensure you're using the correct format for your platform

### Solutions

- Use YouTube Shorts format for faster processing
- Reduce video duration for smaller file sizes
- Check available system memory before processing normal format videos
- Use the test script to verify functionality before batch processing
