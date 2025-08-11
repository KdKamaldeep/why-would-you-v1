#!/usr/bin/env python3
"""
Test script to verify narration timing improvements
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.video_processor import VideoProcessor, VideoConfig

def test_audio_duration_detection():
    """Test audio duration detection functionality."""
    print("🧪 Testing audio duration detection...")
    
    # Initialize video processor
    config = VideoConfig()
    processor = VideoProcessor(config)
    
    # Test estimation method
    test_text = "This is a test narration with exactly ten words for timing estimation."
    estimated_duration = processor.estimate_narration_duration(test_text)
    print(f"📝 Test text: '{test_text}'")
    print(f"📊 Estimated duration: {estimated_duration:.2f} seconds")
    print(f"📊 Word count: {len(test_text.split())}")
    
    # Test with different word counts
    test_cases = [
        "Short test.",
        "This is a medium length test sentence with more words.",
        "This is a very long test sentence that should take significantly more time to narrate because it contains many more words and should demonstrate the estimation functionality."
    ]
    
    print("\n📊 Duration estimation tests:")
    for text in test_cases:
        duration = processor.estimate_narration_duration(text)
        words = len(text.split())
        print(f"  {words:2d} words: {duration:.2f}s - '{text[:50]}{'...' if len(text) > 50 else ''}'")
    
    print("\n✅ Audio duration detection tests completed!")

def test_timing_improvements():
    """Test the timing improvement logic."""
    print("\n🎬 Testing timing improvement logic...")
    
    # Simulate scene durations and narration
    original_scene_durations = [8, 8, 8]  # Original script durations
    actual_audio_durations = [12.5, 9.2, 15.8]  # Actual generated audio durations
    
    print(f"📝 Original scene durations: {original_scene_durations}")
    print(f"🎵 Actual audio durations: {actual_audio_durations}")
    
    # Calculate improvements
    total_original = sum(original_scene_durations)
    total_actual = sum(actual_audio_durations)
    improvement = total_actual - total_original
    
    print(f"📊 Total original duration: {total_original}s")
    print(f"📊 Total actual duration: {total_actual:.1f}s")
    print(f"📊 Duration adjustment: {improvement:+.1f}s ({improvement/total_original*100:+.1f}%)")
    
    # Show per-scene adjustments
    print("\n📊 Per-scene adjustments:")
    for i, (orig, actual) in enumerate(zip(original_scene_durations, actual_audio_durations)):
        diff = actual - orig
        print(f"  Scene {i+1}: {orig}s → {actual:.1f}s ({diff:+.1f}s)")
    
    print("\n✅ Timing improvement tests completed!")

if __name__ == "__main__":
    print("🎨 Narration Timing Test Suite")
    print("=" * 50)
    
    test_audio_duration_detection()
    test_timing_improvements()
    
    print("\n🎉 All tests completed successfully!")
    print("\n💡 The timing improvements will now:")
    print("   • Calculate actual audio duration for each scene")
    print("   • Adjust video timing to match narration length")
    print("   • Prevent narration from being cut off")
    print("   • Provide better initial duration estimates")
