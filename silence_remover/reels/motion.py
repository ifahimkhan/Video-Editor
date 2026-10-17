"""FFmpeg filter graphs for tracked layouts (moving crops, Auto shots).

A crop's position is a piecewise-linear function of time, written as an
ffmpeg expression: a balanced tree of ``if(lt(t,T),…,…)`` over the
analysis samples, so even a 3-minute clip stays ~10 levels deep. The
graph goes into a filter-script file, so its length doesn't matter.
"""

from __future__ import annotations

import numpy as np

from ..render import sum_expr
from .framing import CropPath
from .planner import FramingPlan
from .settings import Layout, Rect


def build_tracked_graph(plan: FramingPlan) -> str:
    """Filter graph turning `[0:v:0]` into `[v]` for a tracked layout."""
    w, h = plan.canvas
    if plan.layout is Layout.SPEAKER:
        return f"[0:v:0]{_speaker_chain(plan, h)}[v]"
    if plan.layout is Layout.CODE:
        return f"[0:v:0]{_code_chain(plan, h)}[v]"
    if plan.layout is Layout.SPLIT:
        top, bottom = plan.panel_heights
        return ";\n".join([
            "[0:v:0]split=2[scr][cam]",
            f"[scr]{_code_chain(plan, top)}[top]",
            f"[cam]{_speaker_chain(plan, bottom)}[bot]",
            "[top][bot]vstack=inputs=2[v]",
        ])
    return _auto_graph(plan, h)


def _auto_graph(plan: FramingPlan, h: int) -> str:
    code_shots = [s for s in plan.shots if s.kind == "code"]
    if not code_shots:
        return f"[0:v:0]{_speaker_chain(plan, h)}[v]"
    if len(code_shots) == len(plan.shots):
        return f"[0:v:0]{_code_chain(plan, h)}[v]"
    enable = sum_expr([f"between(t,{s.start_s:.3f},{s.end_s:.3f})" for s in code_shots])
    return ";\n".join([
        "[0:v:0]split=2[a][b]",
        f"[a]{_speaker_chain(plan, h)}[spk]",
        f"[b]{_code_chain(plan, h)}[code]",
        f"[spk][code]overlay=x=0:y=0:enable='{enable}'[v]",
    ])


def _speaker_chain(plan: FramingPlan, panel_h: int) -> str:
    return f"{crop_filter(plan.speaker)},scale={plan.canvas[0]}:{panel_h},setsar=1"


def _code_chain(plan: FramingPlan, panel_h: int) -> str:
    # The webcam is shown separately (or not at all), so hide it on the screen.
    return (f"{cover_box(plan.webcam, plan.screen_color)},{crop_filter(plan.code)},"
            f"scale={plan.canvas[0]}:{panel_h},setsar=1")


def crop_filter(path: CropPath) -> str:
    x = piecewise_expr(path.times, path.xs)
    y = piecewise_expr(path.times, path.ys)
    return f"crop=w={path.w}:h={path.h}:x='{x}':y='{y}'"


def cover_box(rect: Rect, color: str) -> str:
    return f"drawbox=x={rect.x}:y={rect.y}:w={rect.w}:h={rect.h}:color={color}:t=fill"


def piecewise_expr(times: np.ndarray, values: np.ndarray) -> str:
    """ffmpeg expression in `t`: linear between samples, held after the last."""
    if len(times) == 0 or len(times) != len(values):
        raise ValueError("A crop path needs at least one sample per time.")
    t, v = _drop_flat_points(np.asarray(times, float), np.round(np.asarray(values, float)))
    if len(t) == 1:
        return _num(v[0])
    tree = _segment_tree(t, v, 0, len(t) - 1)
    return f"if(gte(t,{t[-1]:.3f}),{_num(v[-1])},{tree})"


def _segment_tree(t: np.ndarray, v: np.ndarray, lo: int, hi: int) -> str:
    if hi - lo == 1:
        if v[lo] == v[hi]:
            return _num(v[lo])
        slope = (v[hi] - v[lo]) / (t[hi] - t[lo])
        return f"({_num(v[lo])}+{slope:.4f}*(t-{t[lo]:.3f}))"
    mid = (lo + hi) // 2
    return (f"if(lt(t,{t[mid]:.3f}),{_segment_tree(t, v, lo, mid)},"
            f"{_segment_tree(t, v, mid, hi)})")


def _drop_flat_points(t: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove samples in the middle of a run of equal values (no change in shape)."""
    if len(v) <= 2:
        return t, v
    keep = np.ones(len(v), bool)
    keep[1:-1] = ~((v[1:-1] == v[:-2]) & (v[1:-1] == v[2:]))
    if np.all(v == v[0]):
        keep[1:] = False
    return t[keep], v[keep]


def _num(value: float) -> str:
    return str(int(value))
