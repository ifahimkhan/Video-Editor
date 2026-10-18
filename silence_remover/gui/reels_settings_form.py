"""Settings panel of the reels dialog: every option, turned into ReelSettings."""

from __future__ import annotations

import os

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QWidget,
)

from ..reels.gemini_ranker import KEY_ENV_VAR
from ..reels.settings import (
    CLIP_COUNT_RANGE,
    CODE_ZOOM_RANGE,
    LENGTH_RANGE_S,
    SPLIT_RATIO_RANGE,
    Aspect,
    CaptionStyle,
    Layout,
    ReelSettings,
)

LAYOUTS = [
    (Layout.AUTO, "Auto: code while typing, speaker otherwise"),
    (Layout.SPLIT, "Split: typing on top, webcam below"),
    (Layout.SPEAKER, "Speaker: follow the face, full screen"),
    (Layout.CODE, "Code: follow the typing, full screen"),
    (Layout.FIT_BLUR, "Fit: whole video centred, blurred background"),
    (Layout.FIT_BARS, "Fit: whole video centred, black bars"),
    (Layout.STATIC_CROP, "Centre crop (no tracking)"),
]
ASPECTS = [(Aspect.VERTICAL, "9:16 (Shorts, Reels, TikTok)"), (Aspect.SQUARE, "1:1"),
           (Aspect.PORTRAIT, "4:5 (Instagram feed)")]
STYLES = [(CaptionStyle.BOLD, "Bold (capitals, yellow word)"),
          (CaptionStyle.BOX, "Box (white on dark box)"),
          (CaptionStyle.MINIMAL, "Minimal (plain white)")]
RANKERS = [("local", "Local scorer (offline)"),
           ("gemini", "Gemini AI (sends the transcript to Google)")]
CODE_LAYOUTS = {Layout.AUTO, Layout.SPLIT, Layout.CODE}


class ReelSettingsForm(QWidget):
    changed = pyqtSignal()  # any option changed

    def __init__(self) -> None:
        super().__init__()
        defaults = ReelSettings()
        form = QFormLayout(self)

        self.layout_box = _combo(LAYOUTS)
        self.aspect = _combo(ASPECTS)
        self.min_length = _spin(LENGTH_RANGE_S, defaults.min_length_s, " s")
        self.max_length = _spin(LENGTH_RANGE_S, defaults.max_length_s, " s")
        self.clips = _spin(CLIP_COUNT_RANGE, defaults.clip_count, "")
        self.ranker = _combo(RANKERS)
        self.gemini_note = QLabel()
        self.gemini_note.setWordWrap(True)
        self.code_zoom = _double(CODE_ZOOM_RANGE, defaults.code_zoom, 0.1, "×")
        self.split_ratio = _double(SPLIT_RATIO_RANGE, defaults.split_ratio, 0.05, "")
        self.captions = QCheckBox("Burn in word-by-word captions")
        self.captions.setChecked(True)
        self.caption_style = _combo(STYLES)
        self.title_overlay = QCheckBox("Show the clip title at the start")
        self.remove_silence = QCheckBox("Cut pauses inside each clip")
        self.remove_silence.setChecked(defaults.remove_silence)  # off: pauses are kept

        rows = [
            ("Layout", self.layout_box, "How the landscape video fills the vertical frame."),
            ("Shape", self.aspect, "Output frame shape."),
            ("Shortest clip", self.min_length, "Clips are whole sentences at least this long."),
            ("Longest clip", self.max_length, "...and at most this long."),
            ("Number of clips", self.clips, "How many of the best moments to find."),
            ("Pick clips with", self.ranker,
             f"Gemini needs the {KEY_ENV_VAR} environment variable; without it, or if it "
             "fails, the local scorer is used."),
            ("", self.gemini_note, ""),
            ("Code zoom", self.code_zoom, "Zoom into the typing area for bigger text."),
            ("Screen share (split)", self.split_ratio,
             "Share of the height given to the screen in the split layout."),
            ("Captions", self.captions, "Needs the transcript (faster-whisper)."),
            ("Caption style", self.caption_style, "Look of the captions."),
            ("Title", self.title_overlay, "Uses the title from the clip list (editable)."),
            ("Pauses", self.remove_silence,
             "Off: clips keep their natural pauses. On: pauses longer than 0.35 s are cut."),
        ]
        for name, control, tip in rows:
            control.setToolTip(tip)
            form.addRow(name, control)

        for combo in (self.layout_box, self.aspect, self.ranker, self.caption_style):
            combo.currentIndexChanged.connect(self._on_changed)
        for spin in (self.min_length, self.max_length, self.clips, self.code_zoom,
                     self.split_ratio):
            spin.valueChanged.connect(self._on_changed)
        for box in (self.captions, self.title_overlay, self.remove_silence):
            box.toggled.connect(self._on_changed)
        self._on_changed()

    def ranker_name(self) -> str:
        return self.ranker.currentData()

    def settings(self) -> ReelSettings:
        """The chosen options; raises ValueError when they don't fit together."""
        return ReelSettings(
            layout=self.layout_box.currentData(), aspect=self.aspect.currentData(),
            min_length_s=self.min_length.value(), max_length_s=self.max_length.value(),
            clip_count=self.clips.value(), code_zoom=round(self.code_zoom.value(), 2),
            split_ratio=round(self.split_ratio.value(), 2),
            captions=self.captions.isChecked(), caption_style=self.caption_style.currentData(),
            title_overlay=self.title_overlay.isChecked(),
            remove_silence=self.remove_silence.isChecked(),
        )

    def _on_changed(self) -> None:
        layout = self.layout_box.currentData()
        self.code_zoom.setEnabled(layout in CODE_LAYOUTS)
        self.split_ratio.setEnabled(layout is Layout.SPLIT)
        self.caption_style.setEnabled(self.captions.isChecked())
        gemini = self.ranker_name() == "gemini"
        has_key = bool(os.environ.get(KEY_ENV_VAR))
        self.gemini_note.setVisible(gemini)
        self.gemini_note.setText(
            "The transcript will be sent to Google Gemini." if has_key else
            f"{KEY_ENV_VAR} is not set, so the local scorer will be used.")
        self.changed.emit()


def _combo(items) -> QComboBox:
    box = QComboBox()
    for data, label in items:
        box.addItem(label, data)
    return box


def _spin(bounds: tuple[int, int], value: int, suffix: str) -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(*bounds)
    spin.setValue(value)
    spin.setSuffix(suffix)
    return spin


def _double(bounds: tuple[float, float], value: float, step: float,
            suffix: str) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(*bounds)
    spin.setSingleStep(step)
    spin.setDecimals(2)
    spin.setValue(value)
    spin.setSuffix(suffix)
    return spin
