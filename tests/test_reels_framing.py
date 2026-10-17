"""Tests for crop-path smoothing, framing and Auto shot planning."""

import numpy as np
import pytest

from silence_remover.reels.framing import (
    Shot,
    code_path,
    plan_shots,
    speaker_path,
)
from silence_remover.reels.settings import Rect
from silence_remover.reels.tracking import smooth_path

FPS = 5
NAN = float("nan")


def test_smooth_path_holds_still_inside_dead_zone():
    targets = np.array([100, 104, 97, 103, 99, 101], float)
    out = smooth_path(targets, FPS, default=100, dead_zone=10)
    assert np.allclose(out, 100)


def test_smooth_path_moves_towards_target_without_overshoot():
    targets = np.array([0] + [500] * 40, float)
    out = smooth_path(targets, FPS, default=0, dead_zone=0)
    assert np.all(np.diff(out) >= 0)
    assert out[-1] == pytest.approx(500, abs=5)
    assert out[1] < 250  # eases in instead of snapping


def test_smooth_path_limits_speed():
    targets = np.array([0] + [5000] * 10, float)
    out = smooth_path(targets, FPS, default=0, dead_zone=0, max_speed=1000)
    assert np.max(np.abs(np.diff(out))) <= 1000 / FPS + 1e-6


def test_smooth_path_snaps_on_jump():
    targets = np.array([0, 0, 800, 800], float)
    jumps = np.array([False, False, True, False])
    out = smooth_path(targets, FPS, default=0, dead_zone=0, jumps=jumps)
    assert out[2] == 800


def test_smooth_path_holds_then_returns_to_default_when_lost():
    targets = np.array([600] * 5 + [NAN] * 40, float)
    out = smooth_path(targets, FPS, default=0, dead_zone=0, hold_s=1.0)
    assert out[6] == pytest.approx(out[4], abs=5)  # holds briefly
    assert out[-1] == pytest.approx(0, abs=10)    # then eases back


def test_smooth_path_all_missing_stays_at_default():
    out = smooth_path(np.full(5, NAN), FPS, default=42)
    assert np.allclose(out, 42)


def test_speaker_path_keeps_crop_inside_webcam_and_centres_face():
    cam = Rect(1440, 810, 480, 270)
    faces = np.tile([1700.0, 900.0], (20, 1))
    path = speaker_path(faces, cam, (1080, 1920), FPS)
    assert path.h == 270 and path.w == 152
    assert np.all(path.xs >= cam.x) and np.all(path.xs + path.w <= cam.right)
    assert path.xs[-1] + path.w / 2 == pytest.approx(1700, abs=20)
    assert np.allclose(path.ys, cam.y)


def test_speaker_path_without_face_uses_webcam_centre():
    cam = Rect(1440, 810, 480, 270)
    path = speaker_path(np.full((10, 2), NAN), cam, (1080, 1920), FPS)
    assert path.xs[0] + path.w / 2 == pytest.approx(cam.x + cam.w / 2, abs=2)


def test_code_path_follows_focus_and_stays_in_frame():
    focus = np.tile([1800.0, 100.0], (30, 1))
    path = code_path(focus, np.zeros(30, bool), (1920, 1080), (1080, 1920), FPS, zoom=1.5)
    assert path.h == pytest.approx(1080 / 1.5, abs=2)
    assert path.w / path.h == pytest.approx(1080 / 1920, abs=0.01)
    assert np.all(path.xs >= 0) and np.all(path.xs + path.w <= 1920)
    assert np.all(path.ys >= 0) and np.all(path.ys + path.h <= 1080)
    assert path.xs[-1] + path.w == pytest.approx(1920, abs=2)  # pushed to right edge
    assert path.ys[-1] == pytest.approx(0, abs=2)


def test_crop_path_interpolates_between_samples():
    focus = np.array([[400.0, 540.0]] * 5 + [[1400.0, 540.0]] * 5)
    jumps = np.array([False] * 5 + [True] + [False] * 4)
    path = code_path(focus, jumps, (1920, 1080), (1080, 1920), FPS, zoom=2.0)
    x_before, x_after = path.xs[4], path.xs[5]
    assert path.x_at(path.times[4] + 0.1) == pytest.approx((x_before + x_after) / 2)


def test_plan_shots_switches_with_minimum_length():
    times = np.arange(0, 20, 1 / FPS)
    typing = (times >= 5) & (times < 12)
    shots = plan_shots(times, typing, duration_s=20)
    assert [s.kind for s in shots] == ["speaker", "code", "speaker"]
    assert shots[1].start_s == pytest.approx(5, abs=0.3)
    assert shots[-1].end_s == 20
    assert all(s.end_s - s.start_s >= 3 for s in shots[:-1])


def test_plan_shots_ignores_short_typing_bursts():
    times = np.arange(0, 20, 1 / FPS)
    typing = (times >= 5) & (times < 5.4)
    typing |= (times >= 6.5) & (times < 6.9)
    shots = plan_shots(times, typing, duration_s=20)
    # Code shot holds through the short gap instead of flickering.
    kinds = [s.kind for s in shots]
    assert kinds.count("code") == 1


def test_plan_shots_all_idle_is_one_speaker_shot():
    times = np.arange(0, 10, 1 / FPS)
    assert plan_shots(times, np.zeros_like(times, bool), 10) == [Shot(0, 10, "speaker")]


def test_smooth_path_cuts_to_first_target_after_a_jump():
    # The jump frame itself has no target (heatmap just restarted).
    targets = np.array([0, 0, NAN, 800, 800], float)
    jumps = np.array([False, False, True, False, False])
    out = smooth_path(targets, FPS, default=0, dead_zone=0, jumps=jumps)
    assert out[3] == 800
