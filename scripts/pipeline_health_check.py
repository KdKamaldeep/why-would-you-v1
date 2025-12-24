#!/usr/bin/env python3
"""
Quick health check for the reel generator pipeline.

This script verifies whether the WAN text-to-video, Coqui TTS, and FFmpeg
pieces are available without triggering large model downloads. It prints a
clear status report so you can spot missing dependencies before running a
full generation job.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple


@dataclass
class CheckResult:
    """Outcome of a single health check."""

    label: str
    ok: bool
    message: str
    hint: str | None = None


CheckFn = Callable[[], CheckResult]


def _run_command(command: list[str]) -> Tuple[bool, str]:
    """Run a shell command and capture the first line of output."""
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True
        )
        output = result.stdout.strip() or result.stderr.strip()
        return True, output.splitlines()[0] if output else ""
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return False, str(exc)


def check_ffmpeg() -> CheckResult:
    """Verify FFmpeg is installed and reachable in PATH."""
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return CheckResult(
            "FFmpeg",
            False,
            "ffmpeg binary not found in PATH",
            "Install ffmpeg (e.g. `sudo apt install ffmpeg`) and ensure it's on PATH.",
        )

    ok, output = _run_command(["ffmpeg", "-version"])
    if ok:
        return CheckResult("FFmpeg", True, f"Found ffmpeg at {ffmpeg_path}: {output}")
    return CheckResult("FFmpeg", False, output, "Try reinstalling ffmpeg or updating PATH.")


def check_wan() -> CheckResult:
    """Verify WAN T2V dependencies and local model cache."""
    try:
        from diffusers import AutoencoderKLWan, WanPipeline  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            "WAN Text-to-Video",
            False,
            f"diffusers WAN components missing: {exc}",
            "Install WAN support: `pip install diffusers==0.30.2 accelerate torch torchvision`.",
        )

    model_dir = Path("models/wan-2.2-t2v")
    if model_dir.exists():
        return CheckResult("WAN Text-to-Video", True, f"WAN model directory present: {model_dir}")
    return CheckResult(
        "WAN Text-to-Video",
        True,
        "WAN diffusers components importable (no local model directory found)",
        "Run your model download step if you haven't cached WAN weights yet.",
    )


def check_coqui() -> CheckResult:
    """Verify Coqui TTS can be imported and local XTTS assets if present."""
    try:
        from TTS.api import TTS  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            "Coqui TTS",
            False,
            f"Coqui TTS not available: {exc}",
            "Install Coqui XTTS: `pip install TTS==0.22.0` and download the XTTS-v2 assets.",
        )

    xtts_dir = Path("models/tts/XTTS-v2")
    if xtts_dir.exists():
        return CheckResult("Coqui TTS", True, f"Coqui TTS importable with local XTTS assets at {xtts_dir}")
    return CheckResult(
        "Coqui TTS",
        True,
        "Coqui TTS importable (no local XTTS assets detected)",
        "Place XTTS-v2 weights under models/tts/XTTS-v2 for offline synthesis.",
    )


def main() -> int:
    checks: tuple[CheckFn, ...] = (check_wan, check_coqui, check_ffmpeg)

    print("\n🎬 Reel Generator Pipeline Health Check")
    print("=" * 48)

    failed = []
    for fn in checks:
        result = fn()
        status = "✅" if result.ok else "❌"
        print(f"{status} {result.label}: {result.message}")
        if not result.ok and result.hint:
            print(f"   ↳ {result.hint}")
        if not result.ok:
            failed.append(result.label)

    if failed:
        print("\n⚠️  Some checks failed: " + ", ".join(failed))
        print("Please install the missing dependencies or models before running the pipeline.")
        return 1

    print("\nAll core components look ready for generation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
