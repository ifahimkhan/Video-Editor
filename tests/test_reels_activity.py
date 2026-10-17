"""Tests for typing / screen-activity detection."""

import numpy as np
import pytest

from silence_remover.reels.activity import detect_activity
from silence_remover.reels.settings import Rect

W, H, FPS = 960, 540, 5


def blank(value: int = 30) -> np.ndarray:
    return np.full((H, W, 3), value, np.uint8)


def typing_frames(n: int, x0: int = 300, y: int = 200) -> list[np.ndarray]:
    """One more 6x10 'character' per frame along a line starting at x0."""
    frames, frame = [], blank()
    for i in range(n):
        frame = frame.copy()
        frame[y:y + 10, x0 + i * 8:x0 + i * 8 + 6] = 220
        frames.append(frame)
    return frames


def test_static_screen_is_idle():
    track = detect_activity([blank()] * 10, FPS)
    assert len(track.times) == 10
    assert not track.typing.any()
    assert np.isnan(track.focus).all()
    assert not track.jumps.any()


def test_times_follow_fps():
    track = detect_activity([blank()] * 4, FPS)
    assert track.times == pytest.approx([0, 0.2, 0.4, 0.6])


def test_typing_is_detected_and_focused():
    track = detect_activity(typing_frames(15), FPS)
    assert track.typing[5:].all()
    fx, fy = track.focus[-1]
    assert 300 <= fx <= 300 + 15 * 8
    assert fy == pytest.approx(205, abs=10)


def test_focus_follows_most_recent_typing_area():
    frames = typing_frames(10, x0=100, y=100)
    later = typing_frames(20, x0=600, y=420)
    # Keep the first line on screen while typing the second.
    frames += [np.maximum(frames[-1], f) for f in later]
    track = detect_activity(frames, FPS)
    fx, fy = track.focus[-1]
    assert fx > 500 and fy > 380


def test_cursor_blink_is_not_typing():
    on, off = blank(), blank()
    on[200:214, 300:302] = 220
    track = detect_activity([on, off] * 8, FPS)
    assert not track.typing.any()


def test_full_screen_change_is_a_jump_not_typing():
    frames = [blank(30)] * 3 + [blank(200)] * 3
    track = detect_activity(frames, FPS)
    assert track.jumps[3]
    assert not track.typing.any()


def test_jump_resets_focus_to_new_content():
    frames = typing_frames(10, x0=100, y=100)
    switched = [blank(60)] + [np.maximum(blank(60), f) for f in typing_frames(10, 600, 420)]
    track = detect_activity(frames + switched, FPS)
    assert track.jumps[10]
    fx, fy = track.focus[-1]
    assert fx > 500 and fy > 380


def test_changes_inside_webcam_are_ignored():
    rng = np.random.default_rng(0)
    cam = Rect(800, 400, 160, 140)
    frames = []
    for _ in range(10):
        f = blank()
        f[cam.y:cam.bottom, cam.x:cam.right] = rng.integers(0, 255, (cam.h, cam.w, 3))
        frames.append(f)
    track = detect_activity(frames, FPS, exclude=cam)
    assert not track.typing.any() and not track.jumps.any()


def test_rejects_bad_fps():
    with pytest.raises(ValueError):
        detect_activity([blank()], 0)


def test_exclude_covering_whole_screen_is_idle():
    track = detect_activity(typing_frames(6), FPS, exclude=Rect(0, 0, W, H))
    assert not track.typing.any() and not track.jumps.any()
