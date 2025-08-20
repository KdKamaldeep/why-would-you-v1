#!/usr/bin/env python3
"""
Test script for the Narration Converter

This script demonstrates how to use the narration converter to transform
storyboard narration text with inline cues to SSML format.
"""

import json
import sys
import os

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.narration_converter import NarrationConverter

def test_individual_conversions():
    """Test individual narration text conversions"""
    converter = NarrationConverter()
    
    test_cases = [
        # Basic examples from the requirements
        {
            "input": "Hut [nervous]: Uh-oh! The lightning is coming.",
            "expected_contains": "<prosody pitch=\"+3st\" rate=\"95%\">"
        },
        {
            "input": "The hero [excited] shouted: Victory is ours!",
            "expected_contains": "<prosody pitch=\"+2st\" rate=\"105%\">"
        },
        {
            "input": "The wise old man [slow, wise] spoke: Patience is a virtue.",
            "expected_contains": "<prosody pitch=\"-3st\" rate=\"88%\">"
        },
        {
            "input": "The king [low pitch, confident] declared: I am the ruler.",
            "expected_contains": "<prosody pitch=\"-2st\" rate=\"92%\">"
        },
        {
            "input": "There was a [pause] moment of silence.",
            "expected_contains": "<break time=\"600ms\"/>"
        },
        {
            "input": "This is [emphasis strong] very important!",
            "expected_contains": "<emphasis level=\"strong\">"
        },
        
        # Additional test cases
        {
            "input": "The [baby] giggled with joy.",
            "expected_contains": "<prosody pitch=\"+4st\" rate=\"120%\">"
        },
        {
            "input": "The [giant] roared loudly.",
            "expected_contains": "<prosody pitch=\"-3st\" rate=\"80%\">"
        },
        {
            "input": "The [robot] said: Beep boop.",
            "expected_contains": "<prosody pitch=\"-1st\" rate=\"90%\" volume=\"-10%\">"
        },
        {
            "input": "The [magical] fairy whispered secrets.",
            "expected_contains": "<prosody pitch=\"+1st\" rate=\"95%\">"
        }
    ]
    
    print("Testing individual narration conversions:")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        result = converter.convert_narration(test_case["input"])
        print(f"\nTest {i}:")
        print(f"Input:  {test_case['input']}")
        print(f"Output: {result}")
        
        if test_case["expected_contains"] in result:
            print("✓ PASS")
        else:
            print("✗ FAIL - Expected content not found")
            print(f"Expected: {test_case['expected_contains']}")

def test_storyboard_conversion():
    """Test converting a complete storyboard"""
    converter = NarrationConverter()
    
    # Create a sample storyboard with narration cues
    sample_storyboard = {
        "title": "Test Story",
        "description": "A test story with narration cues",
        "scenes": [
            {
                "title": "Scene 1",
                "narration": "The hero [excited] ran into battle.",
                "characters": ["Hero"]
            },
            {
                "title": "Scene 2", 
                "narration": "The wise wizard [slow, wise] cast a spell.",
                "characters": ["Wizard"]
            },
            {
                "title": "Scene 3",
                "narration": "There was a [pause] moment of tension.",
                "characters": ["All"]
            }
        ]
    }
    
    print("\n\nTesting storyboard conversion:")
    print("=" * 50)
    
    # Convert the storyboard
    converted = converter.convert_storyboard(sample_storyboard)
    
    print("Original storyboard:")
    print(json.dumps(sample_storyboard, indent=2))
    
    print("\nConverted storyboard:")
    print(json.dumps(converted, indent=2))
    
    # Verify the conversion worked
    original_narrations = [scene["narration"] for scene in sample_storyboard["scenes"]]
    converted_narrations = [scene["narration"] for scene in converted["scenes"]]
    
    print("\nNarration comparison:")
    for i, (orig, conv) in enumerate(zip(original_narrations, converted_narrations)):
        print(f"\nScene {i+1}:")
        print(f"Original:  {orig}")
        print(f"Converted: {conv}")

def test_file_conversion():
    """Test converting an actual storyboard file"""
    converter = NarrationConverter()
    
    # Use one of the existing storyboard files
    input_file = "storyboards/exaexample_with_cues.json"
    
    if os.path.exists(input_file):
        print(f"\n\nTesting file conversion with {input_file}:")
        print("=" * 50)
        
        try:
            # Read original file
            with open(input_file, 'r', encoding='utf-8') as f:
                original_data = json.load(f)
            
            # Convert it
            converted_data = converter.convert_storyboard(original_data)
            
            # Show a sample of the conversion
            print("Sample narration conversions:")
            for i, scene in enumerate(converted_data["scenes"][:3]):  # Show first 3 scenes
                print(f"\nScene {i+1}:")
                print(f"Original:  {original_data['scenes'][i]['narration']}")
                print(f"Converted: {scene['narration']}")
            
            # Save the converted file
            output_file = "storyboards/aarav_ssml.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(converted_data, f, ensure_ascii=False, indent=2)
            
            print(f"\n✓ Converted file saved as: {output_file}")
            
        except Exception as e:
            print(f"✗ Error converting file: {e}")
    else:
        print(f"\nFile {input_file} not found, skipping file conversion test")

def main():
    """Run all tests"""
    print("Narration Converter Test Suite")
    print("=" * 60)
    
    test_individual_conversions()
    test_storyboard_conversion()
    test_file_conversion()
    
    print("\n" + "=" * 60)
    print("Test suite completed!")

if __name__ == "__main__":
    main()
