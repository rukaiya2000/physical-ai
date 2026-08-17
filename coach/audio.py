"""Non-blocking playback of short coaching cues."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np


DECODE_SAMPLE_RATE = 48_000


def _read_audio(path: Path) -> tuple[int, np.ndarray]:
    """Read WAV directly and decode other formats to stereo PCM with ffmpeg."""

    if path.suffix.lower() == ".wav":
        from scipy.io import wavfile

        return wavfile.read(path)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            f"ffmpeg is required to play {path.suffix or 'this audio format'} cues"
        )
    result = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-vn",
            "-ac",
            "2",
            "-ar",
            str(DECODE_SAMPLE_RATE),
            "-f",
            "s16le",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(detail or f"ffmpeg exited with status {result.returncode}")
    samples = np.frombuffer(result.stdout, dtype="<i2")
    if samples.size == 0 or samples.size % 2:
        raise RuntimeError(f"ffmpeg returned invalid audio data for {path}")
    return DECODE_SAMPLE_RATE, samples.reshape(-1, 2)


class AudioCuePlayer:
    def play(self, path: Path | None) -> str | None:
        if path is None:
            return None
        if not path.exists():
            return f"Audio cue not found: {path}"
        try:
            import sounddevice as sd

            sample_rate, samples = _read_audio(path)
            sd.stop()
            sd.play(samples, sample_rate, blocking=False)
        except Exception as error:  # audio devices vary by host
            return f"Audio unavailable: {error}"
        return None

    def stop(self) -> None:
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:
            pass
