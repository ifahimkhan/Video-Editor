"""Extract a file's audio track to MP3 (full length, no silence removal)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .ffmpeg_tools import FFmpegError, find_binary, probe
from .render import run_with_progress

MP3_BITRATES_KBPS = (64, 96, 128, 160, 192, 256, 320)
DEFAULT_MP3_KBPS = 192


def extract_mp3(
    input_path: str | Path,
    output_path: str | Path,
    bitrate_kbps: int = DEFAULT_MP3_KBPS,
    on_progress: Callable[[float], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> Path:
    """Encode the first audio track of `input_path` to an MP3 file.

    Title/artist and other tags from the source are copied. Raises
    FFmpegError when the input has no audio, and RenderCancelled (from
    `render`) when `is_cancelled` returns True mid-export.
    """
    input_path, output_path = Path(input_path), Path(output_path)
    if bitrate_kbps not in MP3_BITRATES_KBPS:
        raise ValueError(f"MP3 bitrate must be one of {MP3_BITRATES_KBPS} kbps")
    if output_path.suffix.lower() != ".mp3":
        raise ValueError("Output file must have the .mp3 extension.")
    if input_path.resolve() == output_path.resolve():
        raise FFmpegError("Output file must be different from the input file.")

    info = probe(input_path)
    if not info.has_audio:
        raise FFmpegError("The selected file has no audio track to extract.")

    cmd = [
        find_binary("ffmpeg"), "-nostdin", "-y", "-v", "error",
        "-i", str(input_path),
        "-map", "0:a:0", "-vn", "-sn", "-dn",
        "-c:a", "libmp3lame", "-b:a", f"{bitrate_kbps}k",
        "-map_metadata", "0", "-id3v2_version", "3",
        "-progress", "pipe:1", "-nostats", str(output_path),
    ]
    run_with_progress(cmd, int(info.duration_s * 1000), output_path,
                      on_progress, is_cancelled)
    return output_path
