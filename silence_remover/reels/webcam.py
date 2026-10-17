"""Find the webcam overlay in a screen recording, once per video.

The webcam picture sits in a fixed corner (bottom-right), so it is located
from a handful of frames spread across the video:

1. Find the face in the bottom-right quarter of each sampled frame.
2. Average each frame's edge strength. The overlay's border is an edge that
   is in the same place in every frame and runs along the whole side.
3. Walk outwards from the face and take, on each side, the nearest line
   whose edge *coverage* (share of the span with a strong edge) is high.
   Code text has gaps between characters and lines, so it rarely qualifies.

If the left or top border can't be found the box is estimated from the
face size and marked as not confident.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..ffmpeg_tools import probe
from .face_detect import Face, FaceDetector
from .frames import grab_frames
from .settings import Rect

SAMPLE_COUNT = 8
EDGE_MIN = 12.0          # mean grey-level step that counts as an edge
COVERAGE_MIN = 0.85      # share of the side's span that must be an edge
MAX_REACH_FACES = 7.0    # borders searched up to this many face sizes away
MIN_GAP_FACES = 0.4      # ...and at least this far from the face
SPAN_SHARE = 0.75        # share of face-to-frame-edge distance a border is scored on
# Estimate when a border is not visible: overlay width in face widths.
CAM_WIDTH_FACES = 5.5
CAM_ASPECT = 16 / 9
FACE_HEIGHT_SHARE = 0.4  # face centre sits at ~40% of the overlay height


@dataclass(frozen=True)
class WebcamGuess:
    rect: Rect
    confident: bool  # False when the box was estimated from the face size


def find_webcam(path: str | Path, detector: FaceDetector | None = None,
                samples: int = SAMPLE_COUNT) -> WebcamGuess | None:
    """Locate the webcam overlay of the video at `path` (source pixels)."""
    info = probe(path)
    times = np.linspace(0.05, 0.95, samples) * info.duration_s
    frames = grab_frames(path, times, (info.width, info.height))
    detector = detector or FaceDetector()
    faces = [corner_faces(detector, f) for f in frames]
    return locate_webcam(frames, faces)


def corner_faces(detector: FaceDetector, frame: np.ndarray) -> list[Face]:
    """Faces in the bottom-right quarter, in full-frame pixels.

    Detecting on the quarter alone gives the small webcam face more pixels.
    """
    h, w = frame.shape[:2]
    oy, ox = h // 2, w // 2
    found = detector.detect(np.ascontiguousarray(frame[oy:, ox:]))
    return [Face(f.x + ox, f.y + oy, f.w, f.h, f.score) for f in found]


def locate_webcam(frames: Sequence[np.ndarray],
                  faces: Sequence[Sequence[Face]]) -> WebcamGuess | None:
    """Overlay box from frames and the faces found in each (same pixels)."""
    if not frames:
        return None
    h, w = frames[0].shape[:2]
    face = _typical_corner_face(faces, w, h)
    if face is None:
        return None
    edge_x, edge_y = _persistent_edges(frames)
    estimate = _estimate_from_face(face, w, h)
    fx, fy, fw, fh = face.x, face.y, face.w, face.h

    # Each border is scored along most of the overlay's side, starting at the
    # face: edges inside the webcam picture (a shoulder, a shelf) are shorter.
    left = _nearest_edge(edge_x, _span(fy, h), axis=1,
                         search=(fx - MIN_GAP_FACES * fw, fx - MAX_REACH_FACES * fw))
    x0 = left if left is not None else estimate.x
    top = _nearest_edge(edge_y, _span(x0, w), axis=0,
                        search=(fy - MIN_GAP_FACES * fh, fy - MAX_REACH_FACES * fh))
    y0 = top if top is not None else estimate.y
    right = _nearest_edge(edge_x, _span(y0, h), axis=1,
                          search=(fx + fw * (1 + MIN_GAP_FACES), fx + fw * (1 + MAX_REACH_FACES)))
    x1 = right if right is not None else w
    bottom = _nearest_edge(edge_y, _span(x0, x1), axis=0,
                           search=(fy + fh * (1 + MIN_GAP_FACES),
                                   fy + fh * (1 + MAX_REACH_FACES)))
    y1 = bottom if bottom is not None else h
    x0, y0 = min(x0, w - 2), min(y0, h - 2)
    rect = Rect(x0, y0, max(2, min(x1, w) - x0), max(2, min(y1, h) - y0))
    return WebcamGuess(rect, confident=left is not None and top is not None)


def _span(start: float, end: float) -> tuple[int, int]:
    """First SPAN_SHARE of start..end: inside the overlay even with a margin."""
    return int(start), int(start + SPAN_SHARE * (end - start))


def _typical_corner_face(faces: Sequence[Sequence[Face]], w: int, h: int) -> Face | None:
    best = []
    for found in faces:
        corner = [f for f in found if f.center[0] > w / 2 and f.center[1] > h / 2]
        if corner:
            best.append(max(corner, key=lambda f: f.score))
    if not best:
        return None
    med = np.median([[f.x, f.y, f.w, f.h] for f in best], axis=0)
    return Face(*(float(v) for v in med), score=float(np.median([f.score for f in best])))


def _persistent_edges(frames: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Mean |horizontal step| (h, w-1) and |vertical step| (h-1, w) over frames."""
    edge_x = np.zeros((frames[0].shape[0], frames[0].shape[1] - 1), np.float32)
    edge_y = np.zeros((frames[0].shape[0] - 1, frames[0].shape[1]), np.float32)
    for frame in frames:
        gray = frame.astype(np.float32).mean(axis=2)
        edge_x += np.abs(np.diff(gray, axis=1))
        edge_y += np.abs(np.diff(gray, axis=0))
    return edge_x / len(frames), edge_y / len(frames)


def _nearest_edge(edges: np.ndarray, span: tuple[int, int], axis: int,
                  search: tuple[float, float]) -> int | None:
    """Nearest border position walking from search[0] towards search[1].

    `axis=1`: vertical borders (columns of `edges`), scored over rows `span`.
    `axis=0`: horizontal borders (rows of `edges`), scored over columns `span`.
    Returns the pixel index where the overlay starts (step index + 1).
    """
    limit = edges.shape[axis] - 1
    start = int(np.clip(round(search[0]), 0, limit))
    stop = int(np.clip(round(search[1]), 0, limit))
    lo, hi = int(span[0]), int(span[1])
    if hi - lo < 2 or start == stop:
        return None
    step = 1 if stop > start else -1
    for k in range(start, stop + step, step):
        line = edges[lo:hi, k] if axis == 1 else edges[k, lo:hi]
        if np.mean(line > EDGE_MIN) >= COVERAGE_MIN:
            return k + 1
    return None


def _estimate_from_face(face: Face, w: int, h: int) -> Rect:
    cam_w = min(w, int(face.w * CAM_WIDTH_FACES))
    cam_h = min(h, int(cam_w / CAM_ASPECT))
    cx, cy = face.center
    x = int(np.clip(cx - cam_w / 2, 0, w - cam_w))
    y = int(np.clip(cy - FACE_HEIGHT_SHARE * cam_h, 0, h - cam_h))
    return Rect(x, y, cam_w, cam_h)
