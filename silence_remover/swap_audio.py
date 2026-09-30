"""Replace a video's soundtrack with another audio file (e.g. a cleaned-up
recording from a noise-removal tool).

The video stream is copied untouched (no re-encode, no quality loss); only
the new audio is encoded. The output always has the video's duration: a
shorter replacement is padded with silence, a longer one is trimmed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .ffmpeg_tools import FFmpegError, find_binary, probe
from .render import MP4_LIKE, run_with_progress

MAX_OFFSET_MS = 60_000
# Replacement audio this much longer/shorter than the video gets a warning.
LENGTH_WARNING_S = 0.5
DEFAULT_AUDIO_BITRATE = "192k"


@dataclass(frozen=True)
class SwapResult:
    output: Path
    video_s: float
    audio_s: float

    @property
    def length_difference_s(self) -> float:
        """Replacement audio minus video duration (negative = audio shorter)."""
        return self.audio_s - self.video_s

    @property
    def length_warning(self) -> str | None:
        diff = self.length_difference_s
        if abs(diff) <= LENGTH_WARNING_S:
            return None
        if diff < 0:
            return (f"The new audio is {-diff:.1f}s shorter than the video; "
                    "the end was filled with silence.")
        return f"The new audio is {diff:.1f}s longer than the video; the extra was cut."


def audio_filter(offset_ms: int) -> str:
    """Shift the audio by `offset_ms` (positive = later), then pad with silence.

    The output duration is capped at the video length separately (`-t`), so
    padding here only matters when the new audio runs out early.
    """
    if not -MAX_OFFSET_MS <= offset_ms <= MAX_OFFSET_MS:
        raise ValueError(f"Audio offset must be within ±{MAX_OFFSET_MS} ms")
    steps = []
    if offset_ms > 0:
        steps.append(f"adelay={offset_ms}:all=1")
    elif offset_ms < 0:
        steps.append(f"atrim=start={-offset_ms / 1000:.3f},asetpts=PTS-STARTPTS")
    steps.append("apad")
    return ",".join(steps)


def audio_codec_for(output_path: Path) -> list[str]:
    """Encoder arguments that fit the output container."""
    if output_path.suffix.lower() == ".webm":
        return ["-c:a", "libopus", "-b:a", "128k"]
    return ["-c:a", "aac", "-b:a", DEFAULT_AUDIO_BITRATE]


def swap_audio(
    video_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
    offset_ms: int = 0,
    on_progress: Callable[[float], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> SwapResult:
    """Write `output_path`: the video of `video_path` with the audio of `audio_path`."""
    video_path, audio_path = Path(video_path), Path(audio_path)
    output_path = Path(output_path)
    if output_path.resolve() in (video_path.resolve(), audio_path.resolve()):
        raise FFmpegError("Output file must be different from both input files.")

    video_info = probe(video_path)
    if not video_info.has_video:
        raise FFmpegError("The selected video file has no video track.")
    audio_info = probe(audio_path)
    if not audio_info.has_audio:
        raise FFmpegError("The selected audio file has no audio track.")

    cmd = [
        find_binary("ffmpeg"), "-nostdin", "-y", "-v", "error",
        "-i", str(video_path), "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy",
        "-af", audio_filter(offset_ms),
        *audio_codec_for(output_path),
        "-t", f"{video_info.duration_s:.3f}",
        "-map_metadata", "0",
    ]
    if output_path.suffix.lower() in MP4_LIKE:
        cmd += ["-movflags", "+faststart"]
    cmd += ["-progress", "pipe:1", "-nostats", str(output_path)]

    run_with_progress(cmd, int(video_info.duration_s * 1000), output_path,
                      on_progress, is_cancelled)
    return SwapResult(output=output_path, video_s=video_info.duration_s,
                      audio_s=audio_info.duration_s + offset_ms / 1000)
