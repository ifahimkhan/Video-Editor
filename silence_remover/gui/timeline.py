"""Fallback timeline (plain QPainter) used when pyqtgraph is not installed."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QWidget

from ..analysis import FLOOR_DB, LoudnessProfile
from ..segments import DetectionMode, DetectionResult, DetectionSettings

DB_TOP = 0.0
DB_BOTTOM = -70.0
KEEP_COLOR = QColor(76, 175, 80, 60)
CUT_COLOR = QColor(229, 57, 53, 70)
WAVE_COLOR = QColor(66, 133, 244)
THRESH_COLOR = QColor(255, 152, 0)
PROB_COLOR = QColor(255, 235, 59)
BG_COLOR = QColor(30, 30, 34)


class TimelineWidget(QWidget):
    # Same interface as WaveformTimeline; this widget never emits them.
    threshold_db_dragged = pyqtSignal(float)
    vad_threshold_dragged = pyqtSignal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(140)
        self._profile: LoudnessProfile | None = None
        self._result: DetectionResult | None = None
        self._settings = DetectionSettings()

    def set_profile(self, profile: LoudnessProfile | None) -> None:
        self._profile = profile
        self._result = None
        self.update()

    def set_detection(self, result: DetectionResult, settings: DetectionSettings) -> None:
        self._result = result
        self._settings = settings
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt API)
        painter = QPainter(self)
        painter.fillRect(self.rect(), BG_COLOR)
        if self._profile is None or self._profile.duration_ms == 0:
            painter.setPen(QColor(160, 160, 160))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "Open a video to see its loudness timeline")
            return
        self._paint_regions(painter)
        self._paint_envelope(painter)
        if self._settings.mode is DetectionMode.VOICE and self._profile.has_vad:
            self._paint_speech_prob(painter)
        else:
            self._paint_threshold(painter)

    def _x(self, ms: float) -> float:
        return ms / self._profile.duration_ms * self.width()

    def _y(self, db: float) -> float:
        clamped = min(DB_TOP, max(DB_BOTTOM, db))
        return (DB_TOP - clamped) / (DB_TOP - DB_BOTTOM) * self.height()

    def _paint_regions(self, painter: QPainter) -> None:
        if self._result is None:
            return
        h = float(self.height())
        for seg in self._result.keep:
            painter.fillRect(QRectF(self._x(seg.start_ms), 0,
                                    self._x(seg.length_ms), h), KEEP_COLOR)
        for seg in self._result.cuts:
            painter.fillRect(QRectF(self._x(seg.start_ms), 0,
                                    self._x(seg.length_ms), h), CUT_COLOR)

    def _paint_envelope(self, painter: QPainter) -> None:
        peaks = _downsample_max(self._profile.db, max(1, self.width()))
        if peaks.size == 0:
            return
        step = self.width() / peaks.size
        bottom = float(self.height())
        points = [QPointF(0, bottom)]
        points += [QPointF(i * step, self._y(v)) for i, v in enumerate(peaks)]
        points.append(QPointF(self.width(), bottom))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(WAVE_COLOR)
        painter.drawPolygon(QPolygonF(points))

    def _paint_threshold(self, painter: QPainter) -> None:
        self._dashed_line(painter, self._y(self._settings.threshold_db))

    def _paint_speech_prob(self, painter: QPainter) -> None:
        """Speech probability as a line (0 at bottom, 1 at top)."""
        h = float(self.height())
        probs = _downsample_max(self._profile.speech_prob, max(1, self.width()))
        if probs.size == 0:
            return
        # The VAD track may be a few ms longer than the loudness track.
        span_ms = len(self._profile.speech_prob) * self._profile.vad_frame_ms
        step = self._x(span_ms) / probs.size
        points = [QPointF(i * step, h - float(p) * h) for i, p in enumerate(probs)]
        painter.setPen(QPen(PROB_COLOR, 1.5))
        painter.drawPolyline(QPolygonF(points))
        self._dashed_line(painter, h - self._settings.vad_threshold * h)

    def _dashed_line(self, painter: QPainter, y: float) -> None:
        painter.setPen(QPen(THRESH_COLOR, 1, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(0, y), QPointF(self.width(), y))


def _downsample_max(db: np.ndarray, buckets: int) -> np.ndarray:
    """Loudest value per pixel column so short words stay visible."""
    if db.size <= buckets:
        return db
    usable = db.size - db.size % buckets
    peaks = db[:usable].reshape(buckets, -1).max(axis=1)
    return np.maximum(peaks, FLOOR_DB)
