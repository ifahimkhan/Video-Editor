"""Render the kept segments into a new file with a single ffmpeg pass.

The filter graph is written to a file and loaded with ``-/filter_complex``
so hundreds of cuts never hit the OS command-line length limit.

Timestamps are rewritten as ``PTS - (silence removed before this point)``.
That keeps audio and video on the same exact timeline, including
variable-frame-rate phone footage; ``aresample`` then fills or trims the
sub-frame gaps left at each audio cut so the audio stays sample-continuous.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .ffmpeg_tools import (
    CREATE_NO_WINDOW,
    FFmpegError,
    find_binary,
    probe,
    require_ffmpeg_version,
)
from .segments import Segment, invert

MP4_LIKE = {".mp4", ".m4v", ".mov"}
AUDIO_FRAME_SAMPLES = 256


class RenderCancelled(Exception):
    pass


@dataclass(frozen=True)
class EncoderSettings:
    video_codec: str = "libx264"
    crf: int = 18
    preset: str = "fast"
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"


DEFAULT_ENCODER = EncoderSettings()


def build_filter_graph(
    keep: tuple[Segment, ...], duration_ms: int, has_video: bool, has_audio: bool
) -> str:
    """Filter graph producing [v] and/or [a] containing only `keep`."""
    if not keep:
        raise ValueError("Nothing to keep: every part of the file is silent.")
    if not (has_video or has_audio):
        raise ValueError("Input has neither video nor audio.")

    select = sum_expr([f"between(t,{_sec(s.start_ms)},{_sec(s.end_ms)})" for s in keep])
    offset = _removed_before_expr(invert(keep, duration_ms))

    chains = []
    if has_video:
        chains.append(
            f"[0:v:0]select='{select}',setpts='PTS-({offset})/TB'[v]"
        )
    if has_audio:
        chains.append(
            # Small audio frames make aselect cut within ~5 ms of the target.
            f"[0:a:0]asetnsamples=n={AUDIO_FRAME_SAMPLES}:p=0,"
            f"aselect='{select}',asetpts='PTS-({offset})/TB',"
            "aresample=async=1:min_hard_comp=0.001:first_pts=0[a]"
        )
    return ";\n".join(chains)


def render(
    input_path: str | Path,
    output_path: str | Path,
    keep: tuple[Segment, ...],
    encoder: EncoderSettings = DEFAULT_ENCODER,
    on_progress: Callable[[float], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> Path:
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise FFmpegError("Output file must be different from the input file.")

    require_ffmpeg_version()
    info = probe(input_path)
    duration_ms = int(info.duration_s * 1000)
    graph = build_filter_graph(keep, duration_ms, info.has_video, info.has_audio)
    kept_ms = sum(s.length_ms for s in keep)

    fd, script_path = tempfile.mkstemp(suffix=".ffgraph", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(graph)
        cmd = _build_command(input_path, output_path, script_path, info, encoder)
        run_with_progress(cmd, kept_ms, output_path, on_progress, is_cancelled)
    finally:
        Path(script_path).unlink(missing_ok=True)
    return output_path


def _build_command(input_path, output_path, script_path, info, encoder) -> list[str]:
    cmd = [
        find_binary("ffmpeg"), "-nostdin", "-y", "-v", "error",
        "-i", str(input_path), "-/filter_complex", script_path,
    ]
    if info.has_video:
        cmd += ["-map", "[v]", "-c:v", encoder.video_codec,
                "-crf", str(encoder.crf), "-preset", encoder.preset,
                "-pix_fmt", "yuv420p"]
    if info.has_audio:
        cmd += ["-map", "[a]", "-c:a", encoder.audio_codec,
                "-b:a", encoder.audio_bitrate]
    if output_path.suffix.lower() in MP4_LIKE:
        cmd += ["-movflags", "+faststart"]
    cmd += ["-progress", "pipe:1", "-nostats", str(output_path)]
    return cmd


def run_with_progress(cmd, total_ms, output_path, on_progress, is_cancelled) -> None:
    with tempfile.TemporaryFile() as err_file:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=err_file, text=True,
            creationflags=CREATE_NO_WINDOW,
        )
        assert proc.stdout is not None
        cancelled = False
        for line in proc.stdout:
            if is_cancelled is not None and is_cancelled():
                cancelled = True
                proc.kill()
                break
            if on_progress is not None and line.startswith("out_time_us="):
                on_progress(_progress_fraction(line, total_ms))
        proc.stdout.close()
        proc.wait()

        if cancelled:
            output_path.unlink(missing_ok=True)
            raise RenderCancelled()
        if proc.returncode != 0:
            err_file.seek(0)
            detail = err_file.read().decode(errors="replace").strip()
            output_path.unlink(missing_ok=True)
            raise FFmpegError(f"Export failed: {detail[-800:]}")
    if on_progress is not None:
        on_progress(1.0)


def _progress_fraction(line: str, total_ms: int) -> float:
    try:
        out_us = int(line.split("=", 1)[1])
    except ValueError:  # ffmpeg prints "N/A" before the first frame
        return 0.0
    return max(0.0, min(1.0, out_us / 1000 / max(1, total_ms)))


def _removed_before_expr(cuts: tuple[Segment, ...]) -> str:
    """Seconds of silence removed before timestamp T.

    `gt` (not `gte`) keeps a frame sitting exactly on a keep/cut boundary
    attached to the segment before it.
    """
    if not cuts:
        return "0"
    return sum_expr([f"gt(T,{_sec(c.start_ms)})*{_sec(c.length_ms)}" for c in cuts])


def sum_expr(terms: list[str]) -> str:
    """Sum of `terms` as a balanced tree of parenthesised pairs.

    FFmpeg's expression parser fails with "Cannot allocate memory" once a
    single `a+b+c+...` chain exceeds 100 terms, which a real video with
    100+ cuts easily does. Nesting pairs keeps every chain at 2 terms and
    the depth at log2(n).
    """
    if not terms:
        raise ValueError("sum_expr needs at least one term")
    if len(terms) == 1:
        return terms[0]
    mid = len(terms) // 2
    return f"({sum_expr(terms[:mid])}+{sum_expr(terms[mid:])})"


def _sec(ms: int) -> str:
    return f"{ms / 1000:.3f}"
