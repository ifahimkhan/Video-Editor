"""Decode video frames into numpy arrays through an ffmpeg pipe."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np

from ..ffmpeg_tools import CREATE_NO_WINDOW, FFmpegError, find_binary
from ..render import RenderCancelled
from .settings import even

# Frames analysed for tracking are scaled to this width: enough detail to see
# a typed character, small enough to keep a 60 s clip fast.
ANALYSIS_WIDTH = 960
ANALYSIS_FPS = 5


def scaled_size(src_w: int, src_h: int, width: int = ANALYSIS_WIDTH) -> tuple[int, int]:
    """Even (width, height) with the source's aspect ratio."""
    width = min(width, src_w)
    return even(width), even(src_h * width / src_w)


def iter_frames(
    path: str | Path,
    start_s: float,
    end_s: float,
    fps: float,
    size: tuple[int, int],
    is_cancelled: Callable[[], bool] | None = None,
) -> Iterator[np.ndarray]:
    """Yield BGR frames (H x W x 3 uint8) of `start_s`..`end_s` at `fps`."""
    w, h = size
    cmd = [
        find_binary("ffmpeg"), "-nostdin", "-v", "error",
        "-ss", f"{start_s:.3f}", "-t", f"{end_s - start_s:.3f}", "-i", str(path),
        "-vf", f"fps={fps},scale={w}:{h}", "-an",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    frame_bytes = w * h * 3
    with tempfile.TemporaryFile() as err_file:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err_file,
                                creationflags=CREATE_NO_WINDOW)
        assert proc.stdout is not None
        try:
            while True:
                if is_cancelled is not None and is_cancelled():
                    raise RenderCancelled()
                data = proc.stdout.read(frame_bytes)
                if len(data) < frame_bytes:
                    break
                yield np.frombuffer(data, np.uint8).reshape(h, w, 3)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.stdout.close()
            proc.wait()
        if proc.returncode not in (0, None) and proc.returncode > 0:
            err_file.seek(0)
            detail = err_file.read().decode(errors="replace").strip()
            raise FFmpegError(f"Could not read video frames: {detail[-500:]}")


def grab_frames(path: str | Path, times: Sequence[float],
                size: tuple[int, int]) -> list[np.ndarray]:
    """One BGR frame at each time in `times`; times past the end are skipped."""
    w, h = size
    frames = []
    for t in times:
        cmd = [
            find_binary("ffmpeg"), "-nostdin", "-v", "error",
            "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
            "-vf", f"scale={w}:{h}", "-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
        ]
        result = subprocess.run(cmd, capture_output=True, creationflags=CREATE_NO_WINDOW)
        if result.returncode != 0:
            raise FFmpegError("Could not read a video frame: "
                              f"{result.stderr.decode(errors='replace').strip()[-500:]}")
        if len(result.stdout) >= w * h * 3:
            frames.append(np.frombuffer(result.stdout[:w * h * 3], np.uint8).reshape(h, w, 3))
    return frames
