"""Tests for the YuNet face detector wrapper."""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from silence_remover.reels.face_detect import (
    INPUT_SIZE,
    STRIDES,
    FaceDetectionUnavailable,
    FaceDetector,
    decode_outputs,
    face_detection_available,
    iou,
    letterbox,
    nms,
)

from .conftest import requires_ffmpeg

FACE_JPG = Path(__file__).parent / "fixtures" / "face.jpg"
# Face position in tests/fixtures/face.jpg (320x320), measured once.
FACE_CENTER = (140, 74)

requires_face_model = pytest.mark.skipif(
    not face_detection_available(), reason="onnxruntime or face model missing"
)


def load_face_bgr() -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(FACE_JPG), "-f", "rawvideo",
         "-pix_fmt", "bgr24", "-"], capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(320, 320, 3)


def empty_outputs() -> dict[str, np.ndarray]:
    out = {}
    for s in STRIDES:
        n = (INPUT_SIZE // s) ** 2
        out[f"cls_{s}"] = np.zeros((1, n, 1), np.float32)
        out[f"obj_{s}"] = np.zeros((1, n, 1), np.float32)
        out[f"bbox_{s}"] = np.zeros((1, n, 4), np.float32)
    return out


def test_letterbox_scales_long_side_to_input_size():
    tensor, scale = letterbox(np.full((180, 320, 3), 200, np.uint8))
    assert tensor.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)
    assert scale == pytest.approx(2.0)
    assert tensor[0, 0, 359, 639] == 200   # inside the resized image
    assert tensor[0, 0, 361, 0] == 0       # padding below it


def test_letterbox_rejects_non_colour_image():
    with pytest.raises(ValueError):
        letterbox(np.zeros((10, 10), np.uint8))


def test_decode_single_anchor():
    out = empty_outputs()
    # Anchor at row 10, col 20 of the stride-16 grid, box 4x5 strides.
    idx = 10 * (INPUT_SIZE // 16) + 20
    out["cls_16"][0, idx, 0] = 0.9
    out["obj_16"][0, idx, 0] = 0.9
    out["bbox_16"][0, idx] = [0.5, 0.5, np.log(4), np.log(5)]
    faces = decode_outputs(out, scale=2.0)
    assert len(faces) == 1
    face = faces[0]
    assert face.score == pytest.approx(0.9)
    cx, cy = face.center
    assert cx == pytest.approx((20.5 * 16) / 2)
    assert cy == pytest.approx((10.5 * 16) / 2)
    assert face.w == pytest.approx(4 * 16 / 2)
    assert face.h == pytest.approx(5 * 16 / 2)


def test_decode_ignores_low_scores():
    out = empty_outputs()
    out["cls_8"][0, 5, 0] = 0.3
    out["obj_8"][0, 5, 0] = 0.3
    assert decode_outputs(out, scale=1.0) == []


def test_nms_drops_overlapping_lower_score():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 10, 10]], float)
    scores = np.array([0.8, 0.9, 0.7])
    assert nms(boxes, scores, 0.3) == [1, 2]


def test_iou():
    assert iou(np.array([0, 0, 10, 10.]), np.array([0, 0, 10, 10.])) == pytest.approx(1)
    assert iou(np.array([0, 0, 10, 10.]), np.array([20, 20, 5, 5.])) == 0


def test_missing_model_is_reported(tmp_path):
    with pytest.raises(FaceDetectionUnavailable):
        FaceDetector(tmp_path / "missing.onnx")


@requires_ffmpeg
@requires_face_model
def test_detects_real_face():
    faces = FaceDetector().detect(load_face_bgr())
    assert len(faces) == 1
    cx, cy = faces[0].center
    assert cx == pytest.approx(FACE_CENTER[0], abs=10)
    assert cy == pytest.approx(FACE_CENTER[1], abs=10)


@requires_ffmpeg
@requires_face_model
def test_detects_small_face_in_webcam_corner():
    frame = np.zeros((1080, 1920, 3), np.uint8)
    frame[760:, 1600:] = load_face_bgr()
    faces = FaceDetector().detect(frame)
    assert faces
    cx, cy = faces[0].center
    assert cx == pytest.approx(1600 + FACE_CENTER[0], abs=15)
    assert cy == pytest.approx(760 + FACE_CENTER[1], abs=15)


def test_no_face_in_blank_frame():
    if not face_detection_available():
        pytest.skip("face model missing")
    assert FaceDetector().detect(np.zeros((360, 640, 3), np.uint8)) == []
