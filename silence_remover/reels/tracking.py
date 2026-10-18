"""Turn jittery per-frame targets into a calm camera move. Pure numpy.

Like a camera operator:

* small moves inside the *dead zone* are ignored, so the crop doesn't wobble
  with every head nod or typed character;
* bigger moves ease in (exponential smoothing) at a capped speed;
* on a *jump* (scroll, window switch, cut) the crop cuts straight to the
  next known target, which may arrive a few frames after the jump itself;
* when the target is lost it holds briefly, then eases back to `default`.
"""

from __future__ import annotations

import math

import numpy as np

SMOOTHING_S = 0.6   # time constant of the ease towards the target
HOLD_S = 1.5        # how long a lost target is held before returning to default


def smooth_path(
    targets: np.ndarray,
    fps: float,
    default: float,
    dead_zone: float = 0.0,
    max_speed: float = math.inf,
    smoothing_s: float = SMOOTHING_S,
    hold_s: float = HOLD_S,
    jumps: np.ndarray | None = None,
) -> np.ndarray:
    """Smoothed positions, one per target. NaN targets mean "lost"."""
    if fps <= 0:
        raise ValueError("fps must be positive.")
    dt = 1.0 / fps
    alpha = 1.0 - math.exp(-dt / smoothing_s) if smoothing_s > 0 else 1.0
    max_step = max_speed * dt

    known = targets[~np.isnan(targets)]
    pos = float(known[0]) if known.size else float(default)
    goal, lost_for, cut_pending = pos, 0.0, False
    out = np.empty(len(targets), float)
    for i, target in enumerate(targets):
        cut_pending = cut_pending or (jumps is not None and bool(jumps[i]))
        if math.isnan(target):
            lost_for += dt
            if lost_for > hold_s:
                goal, cut_pending = float(default), False
        else:
            goal, lost_for = float(target), 0.0
        if cut_pending and not math.isnan(target):
            pos, cut_pending = goal, False
        else:
            pos += _step(goal - pos, dead_zone, alpha, max_step)
        out[i] = pos
    return out


def _step(error: float, dead_zone: float, alpha: float, max_step: float) -> float:
    if abs(error) <= dead_zone:
        return 0.0
    error -= math.copysign(dead_zone, error)
    return max(-max_step, min(max_step, error * alpha))
