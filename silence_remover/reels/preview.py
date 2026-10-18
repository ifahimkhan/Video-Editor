"""Boxes to draw on a source frame to preview a reel layout. Pure logic.

Tracked crops move while the reel plays; the preview shows where they start
(centred), which is enough to judge size, zoom and the webcam position.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .framing import code_path, speaker_path
from .layout import cover_crop, default_webcam
from .settings import Layout, Rect, ReelSettings, split_heights


@dataclass(frozen=True)
class PreviewBox:
    name: str
    rect: Rect
    kind: str  # "frame", "webcam" or "crop": picks the colour


def preview_boxes(settings: ReelSettings, frame_w: int, frame_h: int,
                  webcam: Rect | None) -> list[PreviewBox]:
    frame = Rect(0, 0, frame_w, frame_h)
    canvas_w, canvas_h = settings.aspect.size
    layout = settings.layout
    if layout in (Layout.FIT_BLUR, Layout.FIT_BARS):
        return [PreviewBox("Whole video", frame, "frame")]
    if layout is Layout.STATIC_CROP:
        return [PreviewBox("Crop", cover_crop(frame, canvas_w, canvas_h), "crop")]

    cam = webcam or default_webcam(frame_w, frame_h)
    boxes = [PreviewBox("Webcam", cam, "webcam")]
    if layout is Layout.SPLIT:
        top, bottom = split_heights(canvas_h, settings.split_ratio)
        boxes.append(PreviewBox("Code panel (follows typing)",
                                _code_start(settings, frame_w, frame_h, (canvas_w, top)),
                                "crop"))
        boxes.append(PreviewBox("Speaker panel (follows face)",
                                _speaker_start(cam, (canvas_w, bottom)), "crop"))
        return boxes
    if layout in (Layout.CODE, Layout.AUTO):
        boxes.append(PreviewBox("Code crop (follows typing)",
                                _code_start(settings, frame_w, frame_h, (canvas_w, canvas_h)),
                                "crop"))
    if layout in (Layout.SPEAKER, Layout.AUTO):
        boxes.append(PreviewBox("Speaker crop (follows face)",
                                _speaker_start(cam, (canvas_w, canvas_h)), "crop"))
    return boxes


def _code_start(settings: ReelSettings, frame_w: int, frame_h: int,
                panel: tuple[int, int]) -> Rect:
    path = code_path(np.full((1, 2), np.nan), np.zeros(1, bool), (frame_w, frame_h),
                     panel, fps=1, zoom=settings.code_zoom)
    return Rect(int(path.xs[0]), int(path.ys[0]), path.w, path.h)


def _speaker_start(webcam: Rect, panel: tuple[int, int]) -> Rect:
    path = speaker_path(np.full((1, 2), np.nan), webcam, panel, fps=1)
    return Rect(int(path.xs[0]), int(path.ys[0]), path.w, path.h)
