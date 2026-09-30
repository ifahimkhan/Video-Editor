"""Interactive waveform timeline (pyqtgraph).

Top lane:    mirrored waveform on a dB scale (peak, with RMS loudness drawn
             darker on top), red shading over the parts that will be cut and
             a draggable loudness-threshold line.
Bottom lane: Silero speech probability with a draggable voice threshold;
             shown only in Voice mode.

Mouse wheel zooms the time axis, drag pans. Both lanes share the time axis.
Only the visible range is sent to pyqtgraph, reduced to ~2 points per pixel.
"""

from __future__ import annotations

import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from ..analysis import LoudnessProfile
from ..segments import (
    THRESHOLD_RANGE_DB,
    VAD_THRESHOLD_RANGE,
    DetectionMode,
    DetectionResult,
    DetectionSettings,
)
from .waveform_data import (
    cut_steps,
    db_to_height,
    decimate_max,
    format_time,
    height_to_db,
    time_axis,
)

BACKGROUND = (30, 30, 34)
PEAK_BRUSH = (66, 133, 244, 110)
RMS_BRUSH = (66, 133, 244, 255)
CUT_BRUSH = (229, 57, 53, 70)
PROB_BRUSH = (255, 235, 59, 90)
PROB_PEN = (255, 235, 59)
THRESH_PEN = (255, 152, 0)
LANE_HEIGHT_PX = 70
MIN_VISIBLE_S = 0.5


class TimeAxis(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):  # noqa: N802 (pyqtgraph API)
        return [format_time(v * scale, spacing * scale) for v in values]


def _make_plot(show_time_axis: bool) -> pg.PlotWidget:
    plot = pg.PlotWidget(background=BACKGROUND,
                         axisItems={"bottom": TimeAxis(orientation="bottom")})
    plot.setMenuEnabled(False)
    plot.hideButtons()
    plot.hideAxis("left")
    if not show_time_axis:
        plot.hideAxis("bottom")
    plot.setMouseEnabled(x=True, y=False)
    return plot


class WaveformTimeline(QWidget):
    """Drop-in replacement for the painted TimelineWidget."""

    threshold_db_dragged = pyqtSignal(float)
    vad_threshold_dragged = pyqtSignal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(160)
        self._profile: LoudnessProfile | None = None
        self._syncing = False
        # Full-resolution arrays; plot items only ever get the visible part.
        self._wave_t = self._peak_h = self._rms_h = None
        self._prob_t = self._prob_v = None

        self._wave = _make_plot(show_time_axis=True)
        self._wave.setYRange(-1.05, 1.05, padding=0)
        self._lane = _make_plot(show_time_axis=False)
        self._lane.setYRange(0, 1.02, padding=0)
        self._lane.setFixedHeight(LANE_HEIGHT_PX)
        self._lane.setXLink(self._wave.getPlotItem())
        self._lane.setVisible(False)

        self._build_items()

        self._wave.getViewBox().sigXRangeChanged.connect(self._redraw_visible)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self._wave, 1)
        layout.addWidget(self._lane)

    def _build_items(self) -> None:
        # Step mode needs len(x) == len(y) + 1 even when empty.
        self._cuts = pg.PlotCurveItem(x=[0.0, 0.001], y=[-1.05], pen=None,
                                      brush=pg.mkBrush(*CUT_BRUSH),
                                      fillLevel=-1.05, stepMode="center")
        self._cuts.setZValue(-10)
        self._peak_top = self._wave_curve(PEAK_BRUSH)
        self._peak_bottom = self._wave_curve(PEAK_BRUSH)
        self._rms_top = self._wave_curve(RMS_BRUSH)
        self._rms_bottom = self._wave_curve(RMS_BRUSH)
        self._wave.addItem(self._cuts)

        low, high = (float(db_to_height(v)) for v in THRESHOLD_RANGE_DB)
        self._thresh = self._threshold_line(movable=True, bounds=(low, high))
        self._thresh_mirror = self._threshold_line(movable=False)
        self._thresh.sigPositionChanged.connect(self._on_threshold_dragged)
        self._wave.addItem(self._thresh)
        self._wave.addItem(self._thresh_mirror)

        self._prob = pg.PlotDataItem(pen=pg.mkPen(PROB_PEN, width=1),
                                     brush=pg.mkBrush(*PROB_BRUSH), fillLevel=0)
        self._lane.addItem(self._prob)
        self._vad_line = self._threshold_line(movable=True, bounds=VAD_THRESHOLD_RANGE)
        self._vad_line.sigPositionChanged.connect(self._on_vad_dragged)
        self._lane.addItem(self._vad_line)

    def _wave_curve(self, brush) -> pg.PlotDataItem:
        item = pg.PlotDataItem(pen=None, brush=pg.mkBrush(*brush), fillLevel=0)
        self._wave.addItem(item)
        return item

    @staticmethod
    def _threshold_line(movable: bool, bounds=None) -> pg.InfiniteLine:
        pen = pg.mkPen(THRESH_PEN, width=1.5, style=Qt.PenStyle.DashLine)
        line = pg.InfiniteLine(angle=0, movable=movable, pen=pen, bounds=bounds,
                               hoverPen=pg.mkPen(THRESH_PEN, width=3))
        line.setZValue(10)
        return line

    # ---------- public API (same as TimelineWidget) ----------

    def set_profile(self, profile: LoudnessProfile | None) -> None:
        self._profile = profile
        # Loudness view until set_detection says otherwise.
        self._lane.setVisible(False)
        self._thresh.setVisible(True)
        self._thresh_mirror.setVisible(True)
        self._cuts.setData(x=[0.0, 0.001], y=[-1.05], stepMode="center")
        if profile is None or profile.duration_ms == 0:
            self._profile = None
            self._wave_t = self._prob_t = None
            for item in (self._peak_top, self._peak_bottom, self._rms_top,
                         self._rms_bottom, self._prob):
                item.clear()
            return

        self._wave_t = time_axis(len(profile.db), profile.frame_ms)
        peak_db = profile.peak_db if profile.peak_db is not None else profile.db
        self._peak_h = db_to_height(peak_db[: len(self._wave_t)])
        self._rms_h = db_to_height(profile.db)
        if profile.speech_prob is not None:
            self._prob_t = time_axis(len(profile.speech_prob), profile.vad_frame_ms)
            self._prob_v = profile.speech_prob
        else:
            self._prob_t = self._prob_v = None
            self._prob.clear()

        duration_s = profile.duration_ms / 1000
        for plot in (self._wave, self._lane):
            plot.setLimits(xMin=0, xMax=duration_s,
                           minXRange=min(MIN_VISIBLE_S, duration_s))
        self._wave.setXRange(0, duration_s, padding=0)
        self._redraw_visible()

    def _redraw_visible(self, *_args) -> None:
        """Re-decimate for the visible time range (runs on zoom/pan)."""
        if self._wave_t is None:
            return
        (x0, x1), _ = self._wave.getViewBox().viewRange()
        points = max(200, 2 * self._wave.width())
        t, peak = decimate_max(self._wave_t, self._peak_h, x0, x1, points)
        _, rms = decimate_max(self._wave_t, self._rms_h, x0, x1, points)
        self._peak_top.setData(t, peak)
        self._peak_bottom.setData(t, -peak)
        self._rms_top.setData(t, rms)
        self._rms_bottom.setData(t, -rms)
        if self._prob_t is not None:
            self._prob.setData(*decimate_max(self._prob_t, self._prob_v, x0, x1, points))

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        super().resizeEvent(event)
        self._redraw_visible()

    def set_detection(self, result: DetectionResult, settings: DetectionSettings) -> None:
        if self._profile is None:
            return
        x, y = cut_steps(result.cuts, result.duration_ms)
        self._cuts.setData(x=x, y=y * 1.05, stepMode="center")

        voice = settings.mode is DetectionMode.VOICE and self._profile.has_vad
        self._lane.setVisible(voice)
        self._thresh.setVisible(not voice)
        self._thresh_mirror.setVisible(not voice)

        height = float(db_to_height(settings.threshold_db))
        self._set_line(self._thresh, height)
        if not self._thresh.moving:  # while dragging, the drag handler mirrors it
            self._thresh_mirror.setValue(-height)
        self._set_line(self._vad_line, settings.vad_threshold)

    # ---------- dragging ----------

    def _set_line(self, line: pg.InfiniteLine, value: float) -> None:
        if line.moving:  # user is dragging it; don't fight the mouse
            return
        self._syncing = True
        try:
            line.setValue(value)
        finally:
            self._syncing = False

    def _on_threshold_dragged(self, line: pg.InfiniteLine) -> None:
        self._thresh_mirror.setValue(-line.value())
        if not self._syncing:
            self.threshold_db_dragged.emit(height_to_db(line.value()))

    def _on_vad_dragged(self, line: pg.InfiniteLine) -> None:
        if not self._syncing:
            self.vad_threshold_dragged.emit(float(line.value()))
