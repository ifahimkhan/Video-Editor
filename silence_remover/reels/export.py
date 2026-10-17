"""Export one time range of a video as a reel with a single ffmpeg pass.

The filter graph is built in stages, each feeding the next:
layout (fit / crop / tracked) -> optional pause cuts -> optional captions.
ffmpeg runs inside a temporary folder holding the graph script and the
``captions.ass`` file, so no Windows path has to be escaped in the graph.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable, Sequence

from ..ffmpeg_tools import FFmpegError, MediaInfo, find_binary, probe, require_ffmpeg_version
from ..render import (
    DEFAULT_ENCODER,
    MP4_LIKE,
    EncoderSettings,
    build_filter_graph,
    run_with_progress,
)
from ..segments import Segment
from ..transcript import Word
from .analyze import analyze_clip, resolve_webcam
from .captions import build_ass, remap_words
from .layout import build_layout_graph
from .motion import build_tracked_graph
from .planner import make_plan
from .settings import Layout, ReelSettings

# ffprobe's duration and the last frame's timestamp can differ slightly.
END_TOLERANCE_S = 0.05
# Share of the progress bar used by face / typing analysis of tracked layouts.
ANALYSIS_SHARE = 0.3
GRAPH_NAME = "graph.ffgraph"
CAPTIONS_NAME = "captions.ass"


def render_reel(
    input_path: str | Path,
    output_path: str | Path,
    start_s: float,
    end_s: float,
    settings: ReelSettings,
    encoder: EncoderSettings = DEFAULT_ENCODER,
    on_progress: Callable[[float], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    words: Sequence[Word] = (),
    title: str | None = None,
    keep: tuple[Segment, ...] | None = None,
) -> Path:
    """Cut `start_s`..`end_s` out of the input and frame it for the reel canvas.

    words: transcript words (file times) for burned-in captions.
    title: shown at the top when `settings.title_overlay` is on.
    keep:  clip-relative parts to keep when pauses are cut (None = keep all).
    """
    input_path, output_path = Path(input_path).resolve(), Path(output_path).resolve()
    if input_path == output_path:
        raise FFmpegError("Output file must be different from the input file.")
    if keep is not None and not keep:
        raise ValueError("The whole clip is silent; nothing would be left.")

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
        render_progress = _progress_part(on_progress, 0.0, 1.0)

    clip_ms = round((end_s - start_s) * 1000)
    if keep is not None:
        graph = _add_cuts(graph, keep, clip_ms, info.has_audio)
    ass = _captions_file(settings, words, title, start_s, end_s, keep)
    if ass is not None:
        graph = _relabel(graph, "[pre]") + f";\n[pre]subtitles={CAPTIONS_NAME}[v]"

    with tempfile.TemporaryDirectory(prefix="reel-") as work:
        Path(work, GRAPH_NAME).write_text(graph, encoding="utf-8")
        if ass is not None:
            Path(work, CAPTIONS_NAME).write_text(ass, encoding="utf-8")
        cmd = _build_command(input_path, output_path, start_s, end_s, info, encoder,
                             cut_audio=keep is not None)
        kept_ms = sum(s.length_ms for s in keep) if keep is not None else clip_ms
        run_with_progress(cmd, kept_ms, output_path, render_progress, is_cancelled,
                          cwd=work)
    return output_path


def _tracked_graph(input_path: Path, info: MediaInfo, start_s: float, end_s: float,
                   settings: ReelSettings, on_progress: Callable[[float], None] | None,
                   is_cancelled: Callable[[], bool] | None) -> str:
    webcam = resolve_webcam(input_path, info, settings)
    need_faces = settings.layout is not Layout.CODE
    analysis = analyze_clip(input_path, info, start_s, end_s, webcam, need_faces,
                            on_progress, is_cancelled)
    plan = make_plan(analysis, settings, (info.width, info.height))
    return build_tracked_graph(plan)


def _add_cuts(graph: str, keep: tuple[Segment, ...], clip_ms: int, has_audio: bool) -> str:
    """Cut pauses after the layout, so tracking keeps the original timeline."""
    cuts = build_filter_graph(keep, clip_ms, has_video=True, has_audio=has_audio,
                              video_in="[lay]", audio_in="[0:a:0]")
    return f"{_relabel(graph, '[lay]')};\n{cuts}"


def _captions_file(settings: ReelSettings, words: Sequence[Word], title: str | None,
                   start_s: float, end_s: float,
                   keep: tuple[Segment, ...] | None) -> str | None:
    clip_words = remap_words(words, start_s, end_s, keep) if settings.captions else []
    show_title = settings.title_overlay and bool(title)
    if not clip_words and not show_title:
        return None
    length_s = (sum(s.length_ms for s in keep) / 1000) if keep is not None else end_s - start_s
    return build_ass(clip_words, settings, length_s, title)


def _relabel(graph: str, label: str) -> str:
    """Rename the graph's `[v]` output pad."""
    if graph.count("[v]") != 1:
        raise ValueError("Filter graph must have exactly one [v] output.")
    return graph.replace("[v]", label)


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


def _build_command(input_path: Path, output_path: Path, start_s: float, end_s: float,
                   info: MediaInfo, encoder: EncoderSettings, cut_audio: bool) -> list[str]:
    # `-ss` before `-i` seeks fast and, when re-encoding, still frame-accurately.
    cmd = [
        find_binary("ffmpeg"), "-nostdin", "-y", "-v", "error",
        "-ss", f"{start_s:.3f}", "-t", f"{end_s - start_s:.3f}",
        "-i", str(input_path), "-/filter_complex", GRAPH_NAME,
        "-map", "[v]", "-c:v", encoder.video_codec, "-crf", str(encoder.crf),
        "-preset", encoder.preset, "-pix_fmt", "yuv420p",
    ]
    if info.has_audio:
        cmd += ["-map", "[a]" if cut_audio else "0:a:0", "-c:a", encoder.audio_codec,
                "-b:a", encoder.audio_bitrate]
    if output_path.suffix.lower() in MP4_LIKE:
        cmd += ["-movflags", "+faststart"]
    cmd += ["-progress", "pipe:1", "-nostats", str(output_path)]
    return cmd
