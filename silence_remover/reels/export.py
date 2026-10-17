"""Export one time range of a video as a reel with a single ffmpeg pass."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Callable

from ..ffmpeg_tools import FFmpegError, MediaInfo, find_binary, probe, require_ffmpeg_version
from ..render import DEFAULT_ENCODER, MP4_LIKE, EncoderSettings, run_with_progress
from .analyze import analyze_clip, resolve_webcam
from .layout import build_layout_graph
from .motion import build_tracked_graph
from .planner import make_plan
from .settings import Layout, ReelSettings

# ffprobe's duration and the last frame's timestamp can differ slightly.
END_TOLERANCE_S = 0.05
# Share of the progress bar used by face / typing analysis of tracked layouts.
ANALYSIS_SHARE = 0.3


def render_reel(
    input_path: str | Path,
    output_path: str | Path,
    start_s: float,
    end_s: float,
    settings: ReelSettings,
    encoder: EncoderSettings = DEFAULT_ENCODER,
    on_progress: Callable[[float], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> Path:
    """Cut `start_s`..`end_s` out of the input and frame it for the reel canvas."""
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise FFmpegError("Output file must be different from the input file.")

    require_ffmpeg_version()
    info = probe(input_path)
    if not info.has_video:
        raise FFmpegError(f"{input_path.name} has no video to make a reel from.")
    check_time_range(start_s, end_s, info.duration_s)
    if settings.layout.tracked:
        graph = _tracked_graph(input_path, info, start_s, end_s, settings,
                               _progress_part(on_progress, 0.0, ANALYSIS_SHARE),
                               is_cancelled)
        render_progress = _progress_part(on_progress, ANALYSIS_SHARE, 1.0)
    else:
        graph = build_layout_graph(info.width, info.height, settings)
        render_progress = on_progress

    fd, script_path = tempfile.mkstemp(suffix=".ffgraph", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(graph)
        cmd = _build_command(input_path, output_path, script_path, start_s, end_s,
                             info, encoder)
        length_ms = int((end_s - start_s) * 1000)
        run_with_progress(cmd, length_ms, output_path, render_progress, is_cancelled)
    finally:
        Path(script_path).unlink(missing_ok=True)
    return output_path


def _tracked_graph(input_path: Path, info: MediaInfo, start_s: float, end_s: float,
                   settings: ReelSettings, on_progress: Callable[[float], None],
                   is_cancelled: Callable[[], bool] | None) -> str:
    webcam = resolve_webcam(input_path, info, settings)
    need_faces = settings.layout is not Layout.CODE
    analysis = analyze_clip(input_path, info, start_s, end_s, webcam, need_faces,
                            on_progress, is_cancelled)
    plan = make_plan(analysis, settings, (info.width, info.height))
    return build_tracked_graph(plan)


def _progress_part(on_progress: Callable[[float], None] | None,
                   start: float, end: float) -> Callable[[float], None] | None:
    """Map a step's 0..1 progress onto `start`..`end` of the whole job.

    Never goes backwards: ffmpeg can report "N/A" (read as 0) mid-export.
    """
    if on_progress is None:
        return None
    best = [0.0]

    def report(fraction: float) -> None:
        best[0] = max(best[0], fraction)
        on_progress(start + best[0] * (end - start))
    return report


def check_time_range(start_s: float, end_s: float, duration_s: float) -> None:
    if start_s < 0:
        raise ValueError("Reel start must not be negative.")
    if end_s <= start_s:
        raise ValueError("Reel end must be after its start.")
    if end_s > duration_s + END_TOLERANCE_S:
        raise ValueError(f"Reel end {end_s:.1f}s is past the end of the video "
                         f"({duration_s:.1f}s).")


def _build_command(input_path: Path, output_path: Path, script_path: str,
                   start_s: float, end_s: float, info: MediaInfo,
                   encoder: EncoderSettings) -> list[str]:
    # `-ss` before `-i` seeks fast and, when re-encoding, still frame-accurately.
    cmd = [
        find_binary("ffmpeg"), "-nostdin", "-y", "-v", "error",
        "-ss", f"{start_s:.3f}", "-t", f"{end_s - start_s:.3f}",
        "-i", str(input_path), "-/filter_complex", script_path,
        "-map", "[v]", "-c:v", encoder.video_codec, "-crf", str(encoder.crf),
        "-preset", encoder.preset, "-pix_fmt", "yuv420p",
    ]
    if info.has_audio:
        cmd += ["-map", "0:a:0", "-c:a", encoder.audio_codec,
                "-b:a", encoder.audio_bitrate]
    if output_path.suffix.lower() in MP4_LIKE:
        cmd += ["-movflags", "+faststart"]
    cmd += ["-progress", "pipe:1", "-nostats", str(output_path)]
    return cmd
