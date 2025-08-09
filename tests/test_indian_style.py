#!/usr/bin/env python3
"""Lightweight test for Indian style prompt composition.

This test avoids heavy model initialization by calling the instance method
`_compose_image_prompt` as an unbound function with a tiny dummy `self` that
only provides the `config.style` attribute.
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace


# Ensure 'src' is on the import path first so 'core' resolves locally
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.generate_cartoon_short import CartoonShortsGenerator  # type: ignore
from core.image_generator import ImageGenerator  # type: ignore


def build_dummy_self(style: str):
    return SimpleNamespace(config=SimpleNamespace(style=style))


def test_compose_prompt_indian_style():
    """Validate that Indian style preset is applied to image prompt."""
    self_obj = build_dummy_self('indian')
    scene = {
        'visual_prompt': 'A festive village square with kites and lanterns',
        'description': 'Village celebration',
        'characters': [
            {'name': 'Aarav', 'role': 'boy', 'appearance': 'cheerful', 'clothing': 'kurta', 'emotion': 'happy', 'action': 'running'},
            {'name': 'Meera', 'role': 'girl', 'appearance': 'playful', 'clothing': 'sari', 'emotion': 'excited', 'action': 'waving'},
        ],
    }

    prompt = CartoonShortsGenerator._compose_image_prompt(self_obj, scene)

    print("Composed prompt:\n", prompt)

    assert 'Indian cartoon style' in prompt
    assert 'vibrant festive palette' in prompt
    assert 'avoid anime/manga' in prompt
    assert 'A festive village square' in prompt  # visual_prompt carried over


def test_compose_prompt_desi_alias():
    """Aliases like 'desi' should map to the Indian preset as well."""
    self_obj = build_dummy_self('desi')
    scene = {
        'visual_prompt': 'Bazaar with colorful stalls',
        'characters': []
    }
    prompt = CartoonShortsGenerator._compose_image_prompt(self_obj, scene)
    print("Composed prompt (desi):\n", prompt)
    assert 'Indian cartoon style' in prompt


def test_generate_image_with_indian_style():
    """Compose an Indian style prompt and ensure an image is produced by ImageGenerator."""
    self_obj = build_dummy_self('indian')
    scene = {
        'visual_prompt': 'A festive village square with kites and lanterns',
        'description': 'Village celebration with bazaar stalls and colorful flags',
        'characters': [
            {'name': 'Aarav', 'role': 'boy', 'appearance': 'cheerful', 'clothing': 'kurta', 'emotion': 'happy', 'action': 'running'},
            {'name': 'Meera', 'role': 'girl', 'appearance': 'playful', 'clothing': 'sari', 'emotion': 'excited', 'action': 'waving'},
        ],
    }

    prompt = CartoonShortsGenerator._compose_image_prompt(self_obj, scene)

    out_dir = Path('test_output')
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / 'indian_style_scene.png'

    img_gen = ImageGenerator()
    result_path = img_gen.generate_cartoon_image(prompt, str(out_path))

    assert Path(result_path).exists(), "Expected generated image file to exist"
    size_bytes = Path(result_path).stat().st_size
    assert size_bytes > 0, "Generated image file should not be empty"
    print(f"\n✅ Generated image: {result_path} ({size_bytes} bytes)")


if __name__ == '__main__':
    # Run tests directly
    try:
        test_compose_prompt_indian_style()
        test_compose_prompt_desi_alias()
        test_generate_image_with_indian_style()
        print("\n✅ Indian style prompt tests passed")
    except AssertionError as e:
        print("\n❌ Indian style prompt tests failed:", e)
        sys.exit(1)


