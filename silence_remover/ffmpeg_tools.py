"""Thin helpers around the ffmpeg / ffprobe executables."""

from __future__ import annotations

import functools
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Hide console windows spawned from the GUI on Windows.
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
# `-/filter_complex <file>` (used for export) arrived in FFmpeg 7.
MIN_FFMPEG_MAJOR = 7


class FFmpegError(RuntimeError):
    """Raised when ffmpeg/ffprobe is missing or a command fails."""


@dataclass(frozen=True)
class MediaInfo:
    duration_s: float
    has_video: bool
    has_audio: bool
    audio_channels: int = 0  # of the first audio track


def find_binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise FFmpegError(
            f"'{name}' not found on PATH. Install FFmpeg and make sure "
            f"'{name}' is available from a terminal."
        )
    return path


@functools.cache
def require_ffmpeg_version(minimum_major: int = MIN_FFMPEG_MAJOR) -> None:
    """Fail early with a clear message on ffmpeg releases that are too old.

    Git/nightly builds report versions like "N-11234-g…"; those are recent,
    so an unparseable version is accepted.
    """
    result = subprocess.run(
        [find_binary("ffmpeg"), "-version"], capture_output=True, text=True,
        creationflags=CREATE_NO_WINDOW,
    )
    match = re.match(r"ffmpeg version n?(\d+)\.", result.stdout)
    if match and int(match.group(1)) < minimum_major:
        raise FFmpegError(
            f"FFmpeg {match.group(1)} is too old; version {minimum_major} or newer "
            "is required for export. Please update FFmpeg."
        )


def probe(media_path: str | Path) -> MediaInfo:
    """Read duration and stream types of a media file."""
    path = Path(media_path)
    if not path.is_file():
        raise FFmpegError(f"File not found: {path}")

    cmd = [
        find_binary("ffprobe"), "-v", "error",
        "-show_entries", "format=duration:stream=codec_type,channels",
        "-of", "json", str(path),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, creationflags=CREATE_NO_WINDOW
    )
    if result.returncode != 0:
        raise FFmpegError(f"ffprobe failed: {result.stderr.strip()}")

    try:
        data = json.loads(result.stdout)
        duration_s = float(data["format"]["duration"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise FFmpegError(f"Could not read media duration of {path.name}") from exc

    streams = data.get("streams", [])
    stream_types = {s.get("codec_type") for s in streams}
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    return MediaInfo(
        duration_s=duration_s,
        has_video="video" in stream_types,
        has_audio=bool(audio),
        audio_channels=int(audio[0].get("channels") or 0) if audio else 0,
    )
