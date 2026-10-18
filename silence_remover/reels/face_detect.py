"""Face detection with YuNet (ONNX) run by onnxruntime.

Uses ``face_detection_yunet_2023mar.onnx`` from the OpenCV model zoo
(MIT licensed). The model takes a fixed 640x640 BGR image (raw 0-255
values) and returns, per anchor on 8/16/32-pixel grids, a class score,
an objectness score and a box. Decoding and NMS are done here in numpy
so OpenCV is not needed.

onnxruntime is optional: `FaceDetector()` raises `FaceDetectionUnavailable`
when it or the model file is missing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

INPUT_SIZE = 640
STRIDES = (8, 16, 32)
SCORE_THRESHOLD = 0.6
NMS_IOU = 0.3
MODEL_ENV_VAR = "SILENCE_REMOVER_FACE_MODEL"
DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / \
    "face_detection_yunet_2023mar.onnx"


class FaceDetectionUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class Face:
    """Face box in the pixels of the image passed to `detect`."""

    x: float
    y: float
    w: float
    h: float
    score: float

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2

    @property
    def area(self) -> float:
        return self.w * self.h


def model_path() -> Path:
    override = os.environ.get(MODEL_ENV_VAR)
    return Path(override) if override else DEFAULT_MODEL


def face_detection_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return model_path().is_file()


class FaceDetector:
    """Detects faces in BGR uint8 images of any size."""

    def __init__(self, path: Path | None = None) -> None:
        try:
            import onnxruntime
        except ImportError as exc:
            raise FaceDetectionUnavailable("onnxruntime is not installed") from exc

        path = path or model_path()
        if not path.is_file():
            raise FaceDetectionUnavailable(f"Face model not found: {path}")

        opts = onnxruntime.SessionOptions()
        opts.log_severity_level = 4
        try:
            self._session = onnxruntime.InferenceSession(
                str(path), sess_options=opts, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:  # onnxruntime raises its own exception types
            raise FaceDetectionUnavailable(f"Could not load face model: {exc}") from exc
        self._input = self._session.get_inputs()[0].name
        self._outputs = [o.name for o in self._session.get_outputs()]
        missing = {f"{k}_{s}" for k in ("cls", "obj", "bbox") for s in STRIDES} \
            - set(self._outputs)
        if missing:
            raise FaceDetectionUnavailable(
                f"Unsupported face model (missing outputs {sorted(missing)}); "
                "expected YuNet 2023mar."
            )

    def detect(self, bgr: np.ndarray, threshold: float = SCORE_THRESHOLD) -> list[Face]:
        """Faces in `bgr` (H x W x 3 uint8), best first."""
        tensor, scale = letterbox(bgr)
        values = self._session.run(self._outputs, {self._input: tensor})
        return decode_outputs(dict(zip(self._outputs, values, strict=True)), scale, threshold)


def letterbox(bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """Resize to fit 640x640 (padding bottom/right), as a 1x3x640x640 float tensor."""
    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.size == 0:
        raise ValueError(f"Expected an H x W x 3 image, got shape {bgr.shape}.")
    h, w = bgr.shape[:2]
    scale = INPUT_SIZE / max(h, w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    rows = np.minimum((np.arange(new_h) / scale).astype(int), h - 1)
    cols = np.minimum((np.arange(new_w) / scale).astype(int), w - 1)
    canvas = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32)
    canvas[:new_h, :new_w] = bgr[rows][:, cols]
    return canvas.transpose(2, 0, 1)[np.newaxis], scale


def decode_outputs(outputs: dict[str, np.ndarray], scale: float,
                   threshold: float = SCORE_THRESHOLD) -> list[Face]:
    """Turn raw YuNet outputs into faces in original-image pixels, best first."""
    boxes, scores = [], []
    for stride in STRIDES:
        cols = INPUT_SIZE // stride
        cls = np.clip(outputs[f"cls_{stride}"].reshape(-1), 0, 1)
        obj = np.clip(outputs[f"obj_{stride}"].reshape(-1), 0, 1)
        bbox = outputs[f"bbox_{stride}"].reshape(-1, 4)
        score = np.sqrt(cls * obj)
        keep = np.nonzero(score >= threshold)[0]
        if keep.size == 0:
            continue
        r, c = np.divmod(keep, cols)
        cx = (c + bbox[keep, 0]) * stride
        cy = (r + bbox[keep, 1]) * stride
        w = np.exp(bbox[keep, 2]) * stride
        h = np.exp(bbox[keep, 3]) * stride
        boxes.append(np.stack([cx - w / 2, cy - h / 2, w, h], axis=1))
        scores.append(score[keep])
    if not boxes:
        return []
    all_boxes, all_scores = np.concatenate(boxes), np.concatenate(scores)
    return [
        Face(*(float(v) / scale for v in all_boxes[i]), score=float(all_scores[i]))
        for i in nms(all_boxes, all_scores, NMS_IOU)
    ]


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Indices of boxes (x, y, w, h) kept by non-maximum suppression, best first."""
    order = list(np.argsort(-scores))
    kept: list[int] = []
    while order:
        best = order.pop(0)
        kept.append(int(best))
        order = [i for i in order if iou(boxes[best], boxes[i]) < iou_threshold]
    return kept


def iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return float(inter / union) if union > 0 else 0.0
