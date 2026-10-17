"""Tests for locating the webcam overlay and reading frames."""

import subprocess

import numpy as np
import pytest

from silence_remover.reels.face_detect import Face
from silence_remover.reels.frames import grab_frames, iter_frames, scaled_size
from silence_remover.reels.settings import Rect
from silence_remover.reels.webcam import find_webcam, locate_webcam
from silence_remover.render import RenderCancelled

from .conftest import requires_ffmpeg
from .test_reels_face_detect import FACE_CENTER, FACE_JPG, requires_face_model

W, H = 1920, 1080
CAM = Rect(1600, 760, 320, 320)


def dark_screen(rng: np.random.Generator) -> np.ndarray:
    """Dark editor with short light 'text' dashes on every other line."""
    frame = np.full((H, W, 3), 30, np.uint8)
    for y in range(40, H, 24):
        for x in rng.integers(0, W - 60, 25):
            frame[y:y + 8, x:x + rng.integers(10, 60)] = 200
    return frame


def frames_with_cam(n: int = 6) -> tuple[list[np.ndarray], list[list[Face]]]:
    rng = np.random.default_rng(1)
    frames, faces = [], []
    for i in range(n):
        frame = dark_screen(rng)
        cam = rng.integers(90, 160, (CAM.h, CAM.w, 3)).astype(np.uint8)
        frame[CAM.y:CAM.bottom, CAM.x:CAM.right] = cam
        frames.append(frame)
        fx = CAM.x + 110 + i
        faces.append([Face(fx, CAM.y + 40, 60, 70, 0.9)])
    return frames, faces


def test_locate_webcam_finds_overlay_edges():
    frames, faces = frames_with_cam()
    guess = locate_webcam(frames, faces)
    assert guess is not None and guess.confident
    r = guess.rect
    assert abs(r.x - CAM.x) <= 4 and abs(r.y - CAM.y) <= 4
    assert abs(r.right - CAM.right) <= 4 and abs(r.bottom - CAM.bottom) <= 4


def test_locate_webcam_with_margin_from_frame_edge():
    rng = np.random.default_rng(2)
    cam = Rect(1500, 700, 360, 300)
    frames, faces = [], []
    for _ in range(5):
        frame = dark_screen(rng)
        frame[cam.y:cam.bottom, cam.x:cam.right] = rng.integers(
            90, 160, (cam.h, cam.w, 3)).astype(np.uint8)
        frames.append(frame)
        faces.append([Face(cam.x + 150, cam.y + 60, 60, 70, 0.9)])
    r = locate_webcam(frames, faces).rect
    assert abs(r.x - cam.x) <= 4 and abs(r.y - cam.y) <= 4
    assert abs(r.right - cam.right) <= 4 and abs(r.bottom - cam.bottom) <= 4


def test_locate_webcam_without_faces_returns_none():
    frames, _ = frames_with_cam(3)
    assert locate_webcam(frames, [[], [], []]) is None


def test_locate_webcam_ignores_faces_outside_corner():
    frames, _ = frames_with_cam(3)
    faces = [[Face(100, 100, 60, 70, 0.9)]] * 3
    assert locate_webcam(frames, faces) is None


def test_locate_webcam_without_border_estimates_from_face():
    frame = np.full((H, W, 3), 30, np.uint8)  # no visible overlay edges
    faces = [[Face(1700, 850, 60, 70, 0.9)]]
    guess = locate_webcam([frame], faces)
    assert guess is not None and not guess.confident
    r = guess.rect
    assert r.x < 1700 and r.right > 1760 and r.y < 850 and r.bottom > 920
    assert r.fits_inside(W, H)


def test_scaled_size_is_even_and_keeps_aspect():
    assert scaled_size(1920, 1080, 960) == (960, 540)
    w, h = scaled_size(1366, 768, 640)
    assert w == 640 and h % 2 == 0


@pytest.fixture(scope="module")
def webcam_video(tmp_path_factory):
    """4 s 'screen recording' with the face picture overlaid bottom-right."""
    path = tmp_path_factory.mktemp("cam") / "tutorial.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", "color=c=0x1e1e1e:size=1920x1080:rate=10:duration=4",
        "-loop", "1", "-i", str(FACE_JPG),
        "-filter_complex",
        # White blocks stand in for code text (drawtext needs fonts).
        "[0:v]drawbox=x=100:y=200:w=420:h=24:color=white:t=fill,"
        "drawbox=x=100:y=260:w=260:h=24:color=white:t=fill[scr];"
        "[scr][1:v]overlay=1600:760:shortest=1,format=yuv420p[v]",
        "-map", "[v]", "-c:v", "libx264", "-preset", "ultrafast", "-t", "4", str(path),
    ], check=True)
    return path


@requires_ffmpeg
def test_iter_frames_reads_expected_count(webcam_video):
    frames = list(iter_frames(webcam_video, 1.0, 3.0, fps=5, size=(480, 270)))
    assert len(frames) == pytest.approx(10, abs=1)
    assert frames[0].shape == (270, 480, 3) and frames[0].dtype == np.uint8


@requires_ffmpeg
def test_iter_frames_can_be_cancelled(webcam_video):
    with pytest.raises(RenderCancelled):
        list(iter_frames(webcam_video, 0, 4, fps=5, size=(480, 270),
                         is_cancelled=lambda: True))


@requires_ffmpeg
def test_grab_frames_at_times(webcam_video):
    frames = grab_frames(webcam_video, [0.5, 2.0], size=(1920, 1080))
    assert len(frames) == 2 and frames[1].shape == (1080, 1920, 3)


@requires_ffmpeg
@requires_face_model
def test_find_webcam_in_real_video(webcam_video):
    guess = find_webcam(webcam_video)
    assert guess is not None
    r = guess.rect
    assert r.x <= 1600 + FACE_CENTER[0] <= r.right
    assert r.y <= 760 + FACE_CENTER[1] <= r.bottom
    assert guess.confident
    assert abs(r.x - 1600) <= 6 and abs(r.y - 760) <= 6
    assert r.right == 1920 and r.bottom == 1080
