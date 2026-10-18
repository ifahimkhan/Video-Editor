"""Turn a clip's analysis into the crops and shots for a tracked layout. Pure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .analyze import ClipAnalysis
from .framing import CropPath, Shot, code_path, plan_shots, speaker_path
from .settings import Layout, Rect, ReelSettings, split_heights


@dataclass(frozen=True)
class FramingPlan:
    layout: Layout
    canvas: tuple[int, int]
    webcam: Rect
    screen_color: str
    speaker: CropPath | None = None
    code: CropPath | None = None
    shots: tuple[Shot, ...] = ()
    panel_heights: tuple[int, int] | None = None  # SPLIT: (top, bottom)


def make_plan(analysis: ClipAnalysis, settings: ReelSettings,
              frame: tuple[int, int]) -> FramingPlan:
    if not settings.layout.tracked:
        raise ValueError(f"Layout '{settings.layout.value}' does not use tracking.")
    canvas = settings.aspect.size
    a = analysis
    base = {"layout": settings.layout, "canvas": canvas, "webcam": a.webcam,
            "screen_color": a.screen_color}

    def speaker(panel: tuple[int, int]) -> CropPath:
        return speaker_path(a.faces, a.webcam, panel, a.fps)

    def code(panel: tuple[int, int]) -> CropPath:
        return code_path(a.focus, a.jumps, frame, panel, a.fps, settings.code_zoom)

    if settings.layout is Layout.SPEAKER:
        return FramingPlan(**base, speaker=speaker(canvas))
    if settings.layout is Layout.CODE:
        return FramingPlan(**base, code=code(canvas))
    if settings.layout is Layout.SPLIT:
        top, bottom = split_heights(canvas[1], settings.split_ratio)
        return FramingPlan(**base, code=code((canvas[0], top)),
                           speaker=speaker((canvas[0], bottom)),
                           panel_heights=(top, bottom))
    times = np.arange(len(a.typing)) / a.fps
    shots = plan_shots(times, a.typing, a.duration_s)
    return FramingPlan(**base, speaker=speaker(canvas), code=code(canvas),
                       shots=tuple(shots))
