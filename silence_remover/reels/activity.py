"""Detect typing on a screen recording and where it happens. Pure numpy.

Consecutive frames are compared (the webcam overlay excluded):

* a small change (a few typed characters) counts as typing activity and is
  added to a fading heatmap; the focus point is the hottest area of it;
* a change over a large part of the screen (scroll, window switch, cut) is a
  *jump*: the heatmap restarts so the focus moves straight to new content;
* a pixel only counts as changed when it differs from both of the two
  previous frames, so content that flips back and forth (a blinking cursor)
  is ignored while newly typed text is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .settings import Rect

PIXEL_DIFF = 25             # grey-level change that counts as "changed"
MIN_CHANGED_SHARE = 4e-5    # below this share of the screen: noise (~20 px at 960x540)
JUMP_SHARE = 0.15           # above this share: scroll, window switch or cut
HEAT_WINDOW_S = 2.0         # heatmap fades by 1/e over this time
TYPING_WINDOW_S = 1.0       # typing = enough active frames within this time
TYPING_MIN_ACTIVE = 2
CELL_PX = 32                # heatmap peak search grid
PEAK_RADIUS_CELLS = 3       # focus = centroid of heat this close to the peak
MIN_HEAT = 1.0


@dataclass(frozen=True)
class ActivityTrack:
    """Per analysed frame, times relative to the first frame."""

    times: np.ndarray   # seconds
    typing: np.ndarray  # bool
    focus: np.ndarray   # (n, 2) x, y in frame pixels; NaN when nothing is happening
    jumps: np.ndarray   # bool: big screen change at this frame


def detect_activity(frames: Iterable[np.ndarray], fps: float,
                    exclude: Rect | None = None) -> ActivityTrack:
    """Analyse BGR (or grey) frames sampled at `fps`."""
    detector = ActivityDetector(fps, exclude)
    for frame in frames:
        detector.update(frame)
    return detector.track()


class ActivityDetector:
    """Streaming form of `detect_activity`: feed frames one at a time."""

    def __init__(self, fps: float, exclude: Rect | None = None) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive.")
        self._fps = fps
        self._exclude = exclude
        self._decay = float(np.exp(-1.0 / (HEAT_WINDOW_S * fps)))
        self._window = max(1, round(TYPING_WINDOW_S * fps))
        self._previous: np.ndarray | None = None
        self._before_previous: np.ndarray | None = None
        self._heat: np.ndarray | None = None
        self._mask: np.ndarray | None = None
        self._active: list[bool] = []
        self._typing: list[bool] = []
        self._focus: list[tuple[float, float]] = []
        self._jumps: list[bool] = []

    def update(self, frame: np.ndarray) -> None:
        gray = _gray(frame)
        if self._previous is None:
            self._heat = np.zeros(gray.shape, np.float32)
            self._mask = _screen_mask(gray.shape, self._exclude)
            is_active = is_jump = False
        else:
            changed = (np.abs(gray - self._previous) > PIXEL_DIFF) & self._mask
            if self._before_previous is not None:
                changed &= np.abs(gray - self._before_previous) > PIXEL_DIFF
            share = changed.sum() / max(1, self._mask.sum())
            is_jump = share > JUMP_SHARE
            is_active = not is_jump and share >= MIN_CHANGED_SHARE
            self._heat = np.zeros_like(self._heat) if is_jump else self._heat * self._decay
            if is_active:
                self._heat += changed
        self._before_previous, self._previous = self._previous, gray
        self._active.append(is_active)
        self._typing.append(sum(self._active[-self._window:]) >= TYPING_MIN_ACTIVE)
        self._focus.append(_hottest_point(self._heat))
        self._jumps.append(is_jump)

    def track(self) -> ActivityTrack:
        n = len(self._active)
        return ActivityTrack(
            times=np.arange(n) / self._fps,
            typing=np.array(self._typing, bool),
            focus=np.array(self._focus, float).reshape(n, 2),
            jumps=np.array(self._jumps, bool),
        )


def _gray(frame: np.ndarray) -> np.ndarray:
    return frame.astype(np.float32).mean(axis=2) if frame.ndim == 3 \
        else frame.astype(np.float32)


def _screen_mask(shape: tuple[int, ...], exclude: Rect | None) -> np.ndarray:
    mask = np.ones(shape[:2], bool)
    if exclude is not None:
        mask[exclude.y:exclude.bottom, exclude.x:exclude.right] = False
    return mask


def _hottest_point(heat: np.ndarray) -> tuple[float, float]:
    """Centroid of the heat around the hottest grid cell, or NaN if cold."""
    if heat.sum() < MIN_HEAT:
        return float("nan"), float("nan")
    h, w = heat.shape
    rows, cols = -(-h // CELL_PX), -(-w // CELL_PX)
    padded = np.zeros((rows * CELL_PX, cols * CELL_PX), np.float32)
    padded[:h, :w] = heat
    cells = padded.reshape(rows, CELL_PX, cols, CELL_PX).sum(axis=(1, 3))
    r, c = np.unravel_index(np.argmax(cells), cells.shape)
    y0 = max(0, (r - PEAK_RADIUS_CELLS) * CELL_PX)
    y1 = min(h, (r + PEAK_RADIUS_CELLS + 1) * CELL_PX)
    x0 = max(0, (c - PEAK_RADIUS_CELLS) * CELL_PX)
    x1 = min(w, (c + PEAK_RADIUS_CELLS + 1) * CELL_PX)
    local = heat[y0:y1, x0:x1]
    ys, xs = np.mgrid[y0:y1, x0:x1]
    total = local.sum()
    return float((xs * local).sum() / total), float((ys * local).sum() / total)
