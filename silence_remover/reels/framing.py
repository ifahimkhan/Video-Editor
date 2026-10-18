"""Plan the moving crops (speaker / code) and the Auto shot list. Pure numpy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .layout import cover_crop
from .settings import Rect, even
from .tracking import smooth_path

# Tuned as shares of the crop size so they work at any resolution.
SPEAKER_DEAD_ZONE = 0.05   # crop widths
SPEAKER_MAX_SPEED = 1.0    # crop widths per second
CODE_DEAD_ZONE = 0.15
CODE_MAX_SPEED = 1.5
# Auto shots.
MIN_SHOT_S = 3.0           # no shot shorter than this (except the last)
IDLE_TO_SPEAKER_S = 2.5    # no typing this long -> back to the speaker
MIN_LAST_SHOT_S = 1.0      # a shorter final shot is merged into the previous


@dataclass(frozen=True)
class CropPath:
    """A fixed-size crop whose top-left corner moves over time (source pixels)."""

    times: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    w: int
    h: int

    def x_at(self, t: float) -> float:
        return float(np.interp(t, self.times, self.xs))

    def y_at(self, t: float) -> float:
        return float(np.interp(t, self.times, self.ys))


@dataclass(frozen=True)
class Shot:
    start_s: float
    end_s: float
    kind: str  # "speaker" or "code"


def speaker_path(faces: np.ndarray, webcam: Rect, canvas: tuple[int, int],
                 fps: float) -> CropPath:
    """Crop of the webcam with the canvas aspect, following the face centre.

    `faces` is (n, 2) face centres in source pixels, NaN where no face was found.
    """
    crop = cover_crop(webcam, *canvas)
    return _follow(faces, crop, webcam, fps, SPEAKER_DEAD_ZONE, SPEAKER_MAX_SPEED)


def code_path(focus: np.ndarray, jumps: np.ndarray, frame: tuple[int, int],
              panel: tuple[int, int], fps: float, zoom: float) -> CropPath:
    """Crop of the screen with the panel's aspect, following typing activity.

    `zoom` 1.0 is the largest crop that fits the frame; 2.0 shows half of that.
    """
    frame_w, frame_h = frame
    base = cover_crop(Rect(0, 0, frame_w, frame_h), *panel)
    w, h = min(even(base.w / zoom), base.w), min(even(base.h / zoom), base.h)
    centred = Rect((frame_w - w) // 2, (frame_h - h) // 2, w, h)
    return _follow(focus, centred, Rect(0, 0, frame_w, frame_h), fps,
                   CODE_DEAD_ZONE, CODE_MAX_SPEED, jumps)


def plan_shots(times: np.ndarray, typing: np.ndarray, duration_s: float,
               min_shot_s: float = MIN_SHOT_S,
               idle_s: float = IDLE_TO_SPEAKER_S) -> list[Shot]:
    """Auto layout: code focus while typing, the speaker otherwise."""
    kind, start, idle_since = "speaker", 0.0, None
    shots: list[Shot] = []
    for t, is_typing in zip(times, typing, strict=True):
        t = float(t)
        idle_since = None if is_typing else (t if idle_since is None else idle_since)
        held = t - start
        wants = "code" if is_typing else (
            "speaker" if idle_since is not None and t - idle_since >= idle_s else kind)
        first = not shots and held == 0
        if wants != kind and (held >= min_shot_s or first):
            if held > 0:
                shots.append(Shot(start, t, kind))
            kind, start = wants, t
    shots.append(Shot(start, float(duration_s), kind))
    return _merge_short_tail(shots)


def _follow(targets: np.ndarray, crop: Rect, bounds: Rect, fps: float,
            dead_zone: float, max_speed: float,
            jumps: np.ndarray | None = None) -> CropPath:
    """Smoothly move `crop` so its centre follows `targets`, inside `bounds`."""
    x_range = (bounds.x, bounds.right - crop.w)
    y_range = (bounds.y, bounds.bottom - crop.h)
    tx = np.clip(targets[:, 0] - crop.w / 2, *x_range)
    ty = np.clip(targets[:, 1] - crop.h / 2, *y_range)
    xs = smooth_path(tx, fps, default=crop.x, dead_zone=dead_zone * crop.w,
                     max_speed=max_speed * crop.w, jumps=jumps)
    ys = smooth_path(ty, fps, default=crop.y, dead_zone=dead_zone * crop.h,
                     max_speed=max_speed * crop.h, jumps=jumps)
    return CropPath(
        times=np.arange(len(targets)) / fps,
        xs=np.clip(xs, *x_range), ys=np.clip(ys, *y_range), w=crop.w, h=crop.h,
    )


def _merge_short_tail(shots: list[Shot]) -> list[Shot]:
    if len(shots) < 2 or shots[-1].end_s - shots[-1].start_s >= MIN_LAST_SHOT_S:
        return shots
    previous = shots[-2]
    return [*shots[:-2], Shot(previous.start_s, shots[-1].end_s, previous.kind)]
