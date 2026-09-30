"""Fast lossless export: stream copy, cuts snapped to keyframes.

Without re-encoding, a video segment can only begin on a keyframe. Each kept
segment's start is therefore moved back to the keyframe at or before it,
which keeps a little extra silence but never clips speech. Ends can stay
where they are. Each segment is stream-copied to a temp file and the pieces
are joined with the concat demuxer.

Typical phone/camera footage has a keyframe every 0.5-2 s; screen recordings
can have much longer gaps, which makes lossless cuts correspondingly coarse.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import numpy as np

from .ffmpeg_tools import (
    CREATE_NO_WINDOW,
    FFmpegError,
    find_binary,
    probe,
    require_ffmpeg_version,
)
from .render import MP4_LIKE, RenderCancelled, run_with_progress
from .segments import DetectionResult, Segment

# Seek target sits just after the keyframe so rounding in ffprobe's printed
# time can't make the demuxer seek back to the previous keyframe.
KEYFRAME_EPSILON_S = 0.0005
PARALLEL_JOBS = max(2, min(8, os.cpu_count() or 2))
SEGMENT_PHASE = 0.8   # share of the progress bar spent cutting segments


class KeyframeScanCancelled(Exception):
    pass


def read_keyframes(
    media_path: str | Path, is_cancelled: Callable[[], bool] | None = None
) -> np.ndarray:
    """Keyframe times (ms, sorted) of the first video stream.

    Reads packet flags only, no decoding. Returns an empty array for files
    without video.
    """
    if not probe(media_path).has_video:
        return np.empty(0, dtype=np.float64)
    cmd = [
        find_binary("ffprobe"), "-v", "error", "-select_streams", "v:0",
        "-show_entries", "packet=pts_time,flags", "-of", "csv=p=0", str(media_path),
    ]
    times: list[float] = []
    with tempfile.TemporaryFile() as err_file:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err_file, text=True,
                                creationflags=CREATE_NO_WINDOW)
        assert proc.stdout is not None
        try:
            for i, line in enumerate(proc.stdout):
                if is_cancelled is not None and i % 2000 == 0 and is_cancelled():
                    raise KeyframeScanCancelled()
                parsed = _parse_packet_line(line)
                if parsed is not None:
                    times.append(parsed)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.stdout.close()
            proc.wait()
        if proc.returncode != 0:
            err_file.seek(0)
            detail = err_file.read().decode(errors="replace").strip()
            raise FFmpegError(f"Could not read keyframes: {detail[-500:]}")
    return np.unique(np.asarray(times, dtype=np.float64) * 1000.0)


def _parse_packet_line(line: str) -> float | None:
    """'12.345000,K__' -> 12.345; non-keyframes and N/A timestamps -> None."""
    parts = line.strip().split(",")
    if len(parts) < 2 or not parts[1].startswith("K"):
        return None
    try:
        return float(parts[0])
    except ValueError:
        return None


def snap_to_keyframes(
    keep: tuple[Segment, ...], keyframes_ms: np.ndarray, duration_ms: int
) -> tuple[Segment, ...]:
    """Move each segment start back to the nearest keyframe; merge overlaps.

    With no keyframe information (audio-only input) segments are unchanged.
    """
    if keyframes_ms.size == 0:
        return keep
    snapped: list[Segment] = []
    for seg in keep:
        idx = int(np.searchsorted(keyframes_ms, seg.start_ms, side="right")) - 1
        start = int(keyframes_ms[idx]) if idx >= 0 else 0
        end = min(seg.end_ms, duration_ms)
        if snapped and start <= snapped[-1].end_ms:
            snapped[-1] = Segment(snapped[-1].start_ms, max(end, snapped[-1].end_ms))
        else:
            snapped.append(Segment(start, end))
    return tuple(s for s in snapped if s.length_ms > 0)


def build_concat_list(segment_files: list[Path]) -> str:
    """ffconcat script joining already-cut segment files."""
    if not segment_files:
        raise ValueError("Nothing to keep: every part of the file is silent.")
    lines = ["ffconcat version 1.0"]
    lines += [f"file {_ffconcat_quote(f.resolve().as_posix())}" for f in segment_files]
    return "\n".join(lines) + "\n"


def segment_command(
    input_path: Path, seg: Segment, out_file: Path, has_video: bool, has_audio: bool
) -> list[str]:
    """Stream-copy one segment. `-ss` before `-i` starts on the keyframe."""
    start_s = seg.start_ms / 1000 + (KEYFRAME_EPSILON_S if seg.start_ms else 0.0)
    cmd = [find_binary("ffmpeg"), "-nostdin", "-y", "-v", "error",
           "-ss", f"{start_s:.6f}", "-t", f"{seg.length_ms / 1000:.6f}",
           "-i", str(input_path)]
    if has_video:
        cmd += ["-map", "0:v:0"]
    if has_audio:
        cmd += ["-map", "0:a:0"]
    return cmd + ["-c", "copy", "-avoid_negative_ts", "make_zero", str(out_file)]


def render_lossless(
    input_path: str | Path,
    output_path: str | Path,
    keep: tuple[Segment, ...],
    on_progress: Callable[[float], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> Path:
    """Stream-copy `keep` (already keyframe-snapped) into `output_path`.

    Each segment is copied to its own temp file (in parallel), then the files
    are joined with the concat demuxer. Cutting inside one concat script
    (`inpoint`/`outpoint`) is avoided: it trims by decode timestamp and leaks
    B-frames from the next GOP, giving out-of-order frames at every join.
    """
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise FFmpegError("Output file must be different from the input file.")
    if not keep:
        raise ValueError("Nothing to keep: every part of the file is silent.")
    require_ffmpeg_version()
    info = probe(input_path)
    suffix = output_path.suffix or ".mp4"

    with tempfile.TemporaryDirectory(prefix="silence_remover_") as tmp:
        files = [Path(tmp) / f"seg{i:05d}{suffix}" for i in range(len(keep))]
        commands = [segment_command(input_path, seg, f, info.has_video, info.has_audio)
                    for seg, f in zip(keep, files, strict=True)]
        _run_parallel(commands, on_progress, is_cancelled)

        list_path = Path(tmp) / "list.ffconcat"
        list_path.write_text(build_concat_list(files), encoding="utf-8")
        cmd = [find_binary("ffmpeg"), "-nostdin", "-y", "-v", "error",
               "-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy"]
        if output_path.suffix.lower() in MP4_LIKE:
            cmd += ["-movflags", "+faststart"]
        cmd += ["-progress", "pipe:1", "-nostats", str(output_path)]
        total_ms = sum(s.length_ms for s in keep)
        join_progress = None
        if on_progress is not None:
            join_progress = lambda f: on_progress(SEGMENT_PHASE + (1 - SEGMENT_PHASE) * f)  # noqa: E731
        run_with_progress(cmd, total_ms, output_path, join_progress, is_cancelled)
    return output_path


def _run_parallel(
    commands: list[list[str]],
    on_progress: Callable[[float], None] | None,
    is_cancelled: Callable[[], bool] | None,
) -> None:
    """Run the segment copies on a small thread pool; stop early on error."""
    done = 0
    with ThreadPoolExecutor(max_workers=PARALLEL_JOBS) as pool:
        futures = [pool.submit(_run_one, cmd, is_cancelled) for cmd in commands]
        try:
            for future in as_completed(futures):
                future.result()
                done += 1
                if on_progress is not None:
                    on_progress(SEGMENT_PHASE * done / len(commands))
        except BaseException:
            for f in futures:
                f.cancel()
            raise


def _run_one(cmd: list[str], is_cancelled: Callable[[], bool] | None) -> None:
    if is_cancelled is not None and is_cancelled():
        raise RenderCancelled()
    result = subprocess.run(cmd, capture_output=True, text=True,
                            creationflags=CREATE_NO_WINDOW)
    if result.returncode != 0:
        raise FFmpegError(f"Lossless cut failed: {result.stderr.strip()[-500:]}")


def _ffconcat_quote(path: str) -> str:
    return "'" + path.replace("'", "'\\''") + "'"


__all__ = [
    "KeyframeScanCancelled", "RenderCancelled", "build_concat_list", "read_keyframes",
    "render_lossless", "segment_command", "snap_result", "snap_to_keyframes",
]


def snap_result(result: DetectionResult, keyframes_ms: np.ndarray) -> DetectionResult:
    """Detection result as lossless export will actually cut it."""
    return DetectionResult(
        keep=snap_to_keyframes(result.keep, keyframes_ms, result.duration_ms),
        duration_ms=result.duration_ms,
    )
