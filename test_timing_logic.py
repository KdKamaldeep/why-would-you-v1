#!/usr/bin/env python3
"""
Test the timing logic to ensure it works correctly
"""

def test_timing_logic():
    """Test the timing adjustment logic."""
    print("🧪 Testing timing logic...")
    
    # Simulate the script scenes with original and updated durations
    scenes = [
        {
            'original_duration': 8.0,  # Original script duration
            'duration': 12.5,          # Updated to match actual audio
            'narration': 'This is a longer narration that takes more time to speak.'
        },
        {
            'original_duration': 8.0,
            'duration': 9.2,
            'narration': 'Medium length narration.'
        },
        {
            'original_duration': 8.0,
            'duration': 15.8,
            'narration': 'This is a very long narration that takes significantly more time to speak because it contains many more words.'
        }
    ]
    
    print("📊 Scene durations:")
    for i, scene in enumerate(scenes):
        orig = scene['original_duration']
        curr = scene['duration']
        diff = curr - orig
        print(f"  Scene {i+1}: {orig:.1f}s → {curr:.1f}s ({diff:+.1f}s)")
    
    # Test the regeneration logic
    need_regeneration = False
    for i, scene in enumerate(scenes):
        original_duration = scene.get('original_duration', 8)
        current_duration = scene.get('duration', 8)
        if abs(current_duration - original_duration) > 0.5:
            need_regeneration = True
            print(f"  ✅ Scene {i+1} needs adjustment: {original_duration:.1f}s → {current_duration:.1f}s")
    
    if need_regeneration:
        print("✅ Timing fix would trigger video adjustment")
    else:
        print("❌ No timing adjustment needed (this would be wrong!)")
    
    # Calculate total improvements
    total_original = sum(scene['original_duration'] for scene in scenes)
    total_actual = sum(scene['duration'] for scene in scenes)
    improvement = total_actual - total_original
    
    print(f"\n📊 Summary:")
    print(f"  Total original: {total_original:.1f}s")
    print(f"  Total actual: {total_actual:.1f}s")
    print(f"  Improvement: {improvement:+.1f}s ({improvement/total_original*100:+.1f}%)")
    
    print("\n🎉 Timing logic test completed!")

if __name__ == "__main__":
    test_timing_logic()
