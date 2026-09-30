"""Plot-ready arrays for the waveform timeline. numpy only, no Qt."""

from __future__ import annotations

import numpy as np

from ..segments import Segment

# The waveform is drawn on a dB scale: DISPLAY_FLOOR_DB maps to the centre
# line, 0 dBFS to the top/bottom edge. Below the lowest threshold setting.
DISPLAY_FLOOR_DB = -72.0


def db_to_height(db: np.ndarray | float) -> np.ndarray | float:
    """dBFS -> 0..1 height above the centre line."""
    return np.clip((np.asarray(db, dtype=np.float32) - DISPLAY_FLOOR_DB)
                   / -DISPLAY_FLOOR_DB, 0.0, 1.0)


def height_to_db(height: float) -> float:
    return DISPLAY_FLOOR_DB + float(np.clip(height, 0.0, 1.0)) * -DISPLAY_FLOOR_DB


def time_axis(n_frames: int, frame_ms: int) -> np.ndarray:
    """Centre time in seconds of each frame."""
    return (np.arange(n_frames, dtype=np.float64) + 0.5) * frame_ms / 1000.0


def cut_steps(cuts: tuple[Segment, ...], duration_ms: int) -> tuple[np.ndarray, np.ndarray]:
    """Step-curve arrays shading the cut regions in one plot item.

    Returns (x, y) for ``stepMode="center"``: len(x) == len(y) + 1. `y` is 1
    inside a cut and -1 elsewhere, so with fillLevel=-1 only cuts are filled.
    """
    if duration_ms <= 0:
        return np.asarray([0.0, 0.001]), np.asarray([-1.0])
    edges = [0.0]
    levels: list[float] = []
    for cut in cuts:
        start, end = min(cut.start_ms, duration_ms), min(cut.end_ms, duration_ms)
        if end <= start:
            continue
        if start > edges[-1] * 1000:
            edges.append(start / 1000)
            levels.append(-1.0)
        edges.append(end / 1000)
        levels.append(1.0)
    if not levels or edges[-1] * 1000 < duration_ms:
        edges.append(duration_ms / 1000)
        levels.append(-1.0)
    return np.asarray(edges), np.asarray(levels)


def decimate_max(
    t: np.ndarray, values: np.ndarray, start_s: float, end_s: float, max_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """Visible slice of (t, values), reduced to at most `max_points` bins by max.

    Keeping the per-bin maximum (rather than pyqtgraph's min/max "peak" pairs)
    gives filled envelopes a clean outline at every zoom level.
    """
    lo = max(0, int(np.searchsorted(t, start_s)) - 1)
    hi = min(len(t), int(np.searchsorted(t, end_s)) + 1)
    t_vis, v_vis = t[lo:hi], values[lo:hi]
    if len(t_vis) <= max_points or max_points < 1:
        return t_vis, v_vis
    step = -(-len(t_vis) // max_points)
    starts = np.arange(0, len(t_vis), step)
    return t_vis[starts], np.maximum.reduceat(v_vis, starts)


def format_time(seconds: float, spacing: float = 1.0) -> str:
    """Axis label; shows tenths when zoomed in below one second per tick."""
    sign = "-" if seconds < 0 else ""
    # Round first so 59.96 s shows as 1:00.0, not 0:60.0.
    seconds = round(abs(seconds), 1) if spacing < 1 else float(int(abs(seconds)))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    sec_text = f"{secs:04.1f}" if spacing < 1 else f"{int(secs):02d}"
    if hours:
        return f"{sign}{hours}:{minutes:02d}:{sec_text}"
    return f"{sign}{minutes}:{sec_text}"
