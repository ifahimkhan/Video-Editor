"""Analyse one clip for tracking: face position and typing, in one decode pass."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ..ffmpeg_tools import FFmpegError, MediaInfo
from .activity import ActivityDetector
from .face_detect import FaceDetectionUnavailable, FaceDetector, face_detection_available
from .frames import ANALYSIS_FPS, ANALYSIS_WIDTH, iter_frames
from .layout import default_webcam
from .settings import Rect, ReelSettings
from .webcam import find_webcam

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClipAnalysis:
    """Per-sample tracking data at `fps`, in source pixels; NaN = not found."""

    fps: float
    duration_s: float
    faces: np.ndarray    # (n, 2) face centres
    focus: np.ndarray    # (n, 2) typing focus points
    typing: np.ndarray   # (n,) bool
    jumps: np.ndarray    # (n,) bool: scroll, window switch or cut
    webcam: Rect
    screen_color: str    # "0xRRGGBB": typical screen colour, used to hide the webcam


def resolve_webcam(path: str | Path, info: MediaInfo, settings: ReelSettings) -> Rect:
    """The webcam overlay: from settings, found automatically, or assumed."""
    if settings.webcam is not None:
        if not settings.webcam.fits_inside(info.width, info.height):
            raise ValueError(f"Webcam area {settings.webcam} lies outside the "
                             f"{info.width}x{info.height} video.")
        return settings.webcam
    if face_detection_available():
        guess = find_webcam(path)
        if guess is not None:
            if not guess.confident:
                log.info("Webcam border not visible; estimated from the face: %s",
                         guess.rect)
            return guess.rect
        log.info("No face found in the bottom-right corner; assuming a webcam there.")
    return default_webcam(info.width, info.height)


def analyze_clip(
    path: str | Path,
    info: MediaInfo,
    start_s: float,
    end_s: float,
    webcam: Rect,
    need_faces: bool,
    on_progress: Callable[[float], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> ClipAnalysis:
    """Read the clip once at ANALYSIS_FPS; track the face and the typing."""
    detector = _face_detector() if need_faces else None
    # Frames come at full size (the webcam face is small); the activity
    # detector gets a cheap nearest-neighbour downscale.
    step = max(1, round(info.width / ANALYSIS_WIDTH))
    # Round outwards so the excluded area covers the whole webcam.
    small_x, small_y = webcam.x // step, webcam.y // step
    small_cam = Rect(small_x, small_y,
                     max(1, -(-webcam.right // step) - small_x),
                     max(1, -(-webcam.bottom // step) - small_y))
    activity = ActivityDetector(ANALYSIS_FPS, exclude=small_cam)
    expected = max(1, round((end_s - start_s) * ANALYSIS_FPS))

    faces: list[tuple[float, float]] = []
    screen_color = "0x000000"
    frames = iter_frames(path, start_s, end_s, ANALYSIS_FPS, (info.width, info.height),
                         is_cancelled)
    for i, frame in enumerate(frames):
        if i == 0:
            screen_color = _screen_color(frame, webcam)
        activity.update(frame[::step, ::step])
        faces.append(_face_center(detector, frame, webcam))
        if on_progress is not None:
            on_progress(min(1.0, (i + 1) / expected))

    track = activity.track()
    n = len(faces)
    if n == 0:
        raise FFmpegError(f"No video frames could be read between {start_s:.1f}s "
                          f"and {end_s:.1f}s for tracking.")
    return ClipAnalysis(
        fps=ANALYSIS_FPS, duration_s=end_s - start_s,
        faces=np.array(faces, float).reshape(n, 2),
        focus=track.focus * step, typing=track.typing, jumps=track.jumps,
        webcam=webcam, screen_color=screen_color,
    )


def _face_detector() -> FaceDetector | None:
    try:
        return FaceDetector()
    except FaceDetectionUnavailable as exc:
        log.warning("Face tracking off (%s); the webcam crop stays centred.", exc)
        return None


def _face_center(detector: FaceDetector | None, frame: np.ndarray,
                 webcam: Rect) -> tuple[float, float]:
    if detector is None:
        return float("nan"), float("nan")
    region = np.ascontiguousarray(frame[webcam.y:webcam.bottom, webcam.x:webcam.right])
    found = detector.detect(region)
    if not found:
        return float("nan"), float("nan")
    cx, cy = max(found, key=lambda f: f.score).center
    return cx + webcam.x, cy + webcam.y


def _screen_color(frame: np.ndarray, webcam: Rect) -> str:
    """Median colour of the screen outside the webcam, as 0xRRGGBB."""
    mask = np.ones(frame.shape[:2], bool)
    mask[webcam.y:webcam.bottom, webcam.x:webcam.right] = False
    pixels = frame[mask] if mask.any() else frame.reshape(-1, 3)
    b, g, r = (int(v) for v in np.median(pixels, axis=0))
    return f"0x{r:02x}{g:02x}{b:02x}"
