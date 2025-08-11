#!/usr/bin/env python3
"""
Test the duration detection and script rewriting logic
"""

def test_duration_detection():
    """Test the duration detection and script rewriting logic."""
    print("🧪 Testing duration detection and script rewriting...")
    
    # Simulate original script with estimated durations
    original_script = {
        'scenes': [
            {
                'duration': 8.0,  # Original estimated duration
                'narration': 'This is a longer narration that takes more time to speak.'
            },
            {
                'duration': 8.0,
                'narration': 'Medium length narration.'
            },
            {
                'duration': 8.0,
                'narration': 'This is a very long narration that takes significantly more time to speak because it contains many more words.'
            }
        ]
    }
    
    print("📝 Original script durations:")
    for i, scene in enumerate(original_script['scenes']):
        print(f"  Scene {i+1}: {scene['duration']:.1f}s")
    
    # Simulate detected actual audio durations
    detected_durations = [12.5, 9.2, 15.8]
    
    print(f"\n🎵 Detected actual audio durations: {detected_durations}")
    
    # Simulate the rewriting logic
    print("\n🔄 Rewriting script durations to match actual audio...")
    for i, scene in enumerate(original_script['scenes']):
        original_duration = scene['duration']
        actual_duration = detected_durations[i]
        
        # Store original for comparison
        scene['original_duration'] = original_duration
        
        # Rewrite scene duration to match actual audio
        scene['duration'] = actual_duration
        
        print(f"  Scene {i+1}: {original_duration:.1f}s → {actual_duration:.1f}s")
    
    # Calculate total improvements
    total_original = sum(scene['original_duration'] for scene in original_script['scenes'])
    total_actual = sum(scene['duration'] for scene in original_script['scenes'])
    improvement = total_actual - total_original
    
    print(f"\n📊 Summary:")
    print(f"  Total original: {total_original:.1f}s")
    print(f"  Total actual: {total_actual:.1f}s")
    print(f"  Improvement: {improvement:+.1f}s ({improvement/total_original*100:+.1f}%)")
    
    # Verify the script was updated correctly
    print(f"\n✅ Updated script:")
    for i, scene in enumerate(original_script['scenes']):
        print(f"  Scene {i+1}: {scene['duration']:.1f}s (was {scene['original_duration']:.1f}s)")
    
    print("\n🎉 Duration detection and script rewriting test completed!")

if __name__ == "__main__":
    test_duration_detection()
