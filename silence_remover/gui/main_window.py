"""Main window: open a video, tune the three controls, export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..audio_export import DEFAULT_MP3_KBPS
from ..lossless import snap_result
from ..segments import (
    MIN_SILENCE_RANGE_MS,
    PADDING_RANGE_MS,
    THRESHOLD_RANGE_DB,
    VAD_THRESHOLD_RANGE,
    DetectionMode,
    DetectionSettings,
    detect,
)
from ..transcript import DEFAULT_MODEL, transcript_available
from ..vad import vad_available

try:
    from .waveform_timeline import WaveformTimeline as Timeline
except ImportError:  # pyqtgraph missing: simpler painted timeline
    from .timeline import TimelineWidget as Timeline
from .workers import (
    AnalysisWorker,
    AudioExportWorker,
    MediaAnalysis,
    RemoveEchoWorker,
    RenderWorker,
    SwapAudioWorker,
    TranscriptWorker,
)

TRANSCRIPT_FILTER = "Subtitles (*.srt);;Web subtitles (*.vtt);;Plain text (*.txt)"
ECHO_OUTPUT_FILTER ="MP3 (*.mp3);;WAV (*.wav);;FLAC (*.flac);;M4A (*.m4a)"
AUDIO_FILTER = "Audio files (*.wav *.mp3 *.m4a *.aac *.flac *.ogg *.opus);;All files (*)"
VIDEO_FILTER = "Media files (*.mp4 *.mkv *.mov *.avi *.webm *.m4v *.mp3 *.wav *.m4a)"
DEFAULTS = DetectionSettings()


class SliderSpin(QWidget):
    """Slider and spin box kept in sync, one value."""

    def __init__(self, low: int, high: int, value: int, suffix: str, step: int = 1):
        super().__init__()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(low, high)
        self.slider.setSingleStep(step)
        self.spin = QSpinBox()
        self.spin.setRange(low, high)
        self.spin.setSingleStep(step)
        self.spin.setSuffix(suffix)
        self.spin.setMinimumWidth(90)
        self.slider.valueChanged.connect(self.spin.setValue)
        self.spin.valueChanged.connect(self.slider.setValue)
        self.slider.setValue(value)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.slider, 1)
        row.addWidget(self.spin)

    def value(self) -> int:
        return self.spin.value()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Silence Remover")
        self.resize(900, 520)
        self._input_path: str | None = None
        self._profile = None
        self._result = None
        self._keyframes_ms = np.empty(0)
        self._worker = None
        self._build_ui()
        self._set_busy(False)

    # ---------- UI construction ----------

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)

        top = QHBoxLayout()
        self.btn_open = QPushButton("Open video…")
        self.btn_open.clicked.connect(self._choose_input)
        self.lbl_file = QLabel("No file selected")
        top.addWidget(self.btn_open)
        top.addWidget(self.lbl_file, 1)
        layout.addLayout(top)

        self.timeline = Timeline()
        self.timeline.threshold_db_dragged.connect(
            lambda db: self.threshold.spin.setValue(round(db)))
        self.timeline.vad_threshold_dragged.connect(
            lambda p: self.vad_threshold.spin.setValue(round(p * 100)))
        layout.addWidget(self.timeline, 1)

        self.mode = QComboBox()
        self.mode.addItem("Loudness (dB threshold)", DetectionMode.LOUDNESS)
        self.mode.addItem("Voice (Silero VAD)", DetectionMode.VOICE)
        if not vad_available():
            self.mode.model().item(1).setEnabled(False)
            self.mode.setToolTip("Voice detection needs onnxruntime and the Silero model.")
        self.mode.currentIndexChanged.connect(self._on_mode_changed)
        vad_low, vad_high = (int(v * 100) for v in VAD_THRESHOLD_RANGE)
        self.vad_threshold = SliderSpin(
            vad_low, vad_high, int(DEFAULTS.vad_threshold * 100), " %", 5)
        self.threshold = SliderSpin(*THRESHOLD_RANGE_DB, int(DEFAULTS.threshold_db), " dB")
        self.min_silence = SliderSpin(*MIN_SILENCE_RANGE_MS, DEFAULTS.min_silence_ms, " ms", 50)
        self.padding = SliderSpin(*PADDING_RANGE_MS, DEFAULTS.padding_ms, " ms", 10)
        self.export_mode = QComboBox()
        self.export_mode.addItem("Precise (re-encode, frame-accurate)", False)
        self.export_mode.addItem("Fast lossless (no re-encode, cuts on keyframes)", True)
        self.export_mode.currentIndexChanged.connect(self._refresh_detection)
        grid = QGridLayout()
        self._row_labels: dict[QWidget, QLabel] = {}
        rows = [
            ("Detection", self.mode,
             "Loudness cuts anything quiet. Voice keeps only speech, so it "
             "also removes hum, music and keyboard noise."),
            ("Voice sensitivity", self.vad_threshold,
             "How sure the model must be that audio is speech. Lower keeps more."),
            ("Threshold", self.threshold, "Audio quieter than this is treated as silence."),
            ("Minimum silence", self.min_silence, "Only pauses at least this long are cut."),
            ("Softness (padding)", self.padding,
             "Extra audio kept around speech so words aren't clipped."),
            ("Export", self.export_mode,
             "Lossless copies the original video untouched and is much faster, but each "
             "kept part must start on a keyframe, so a little extra silence may remain."),
        ]
        for i, (name, control, tip) in enumerate(rows):
            label = QLabel(name)
            label.setToolTip(tip)
            control.setToolTip(tip)
            if isinstance(control, SliderSpin):
                control.spin.valueChanged.connect(self._refresh_detection)
            self._row_labels[control] = label
            grid.addWidget(label, i, 0)
            grid.addWidget(control, i, 1)
        layout.addLayout(grid)
        self._on_mode_changed()

        self.lbl_stats = QLabel("")
        layout.addWidget(self.lbl_stats)

        bottom = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self._cancel)
        self.btn_export = QPushButton("Export…")
        self.btn_export.clicked.connect(self._choose_output)
        self.btn_export_audio = QPushButton("Export audio (MP3)…")
        self.btn_export_audio.setToolTip(
            f"Save the full original audio track as MP3 ({DEFAULT_MP3_KBPS} kbps). "
            "Silence cuts are not applied.")
        self.btn_export_audio.clicked.connect(self._choose_audio_output)
        bottom.addWidget(self.progress, 1)
        bottom.addWidget(self.btn_cancel)
        self.btn_swap_audio = QPushButton("Swap audio…")
        self.btn_swap_audio.setToolTip(
            "Replace this video's soundtrack with another audio file (for example a "
            "cleaned-up recording). The picture is copied untouched.")
        self.btn_swap_audio.clicked.connect(self._choose_swap_audio)
        bottom.addWidget(self.btn_swap_audio)
        self.btn_remove_echo = QPushButton("Remove echo…")
        self.btn_remove_echo.setToolTip(
            "Remove the echo of a voice recorded by two microphones at once, and save "
            "the cleaned audio. Use Swap audio to put it back into the video.")
        self.btn_remove_echo.clicked.connect(self._choose_remove_echo)
        bottom.addWidget(self.btn_remove_echo)
        self._transcript_available = transcript_available()
        self.btn_transcript = QPushButton("Export transcript…")
        self.btn_transcript.setToolTip(
            "Write down what is said, as subtitles (.srt/.vtt) or plain text, "
            "in its own language or translated to English. Silence cuts are not "
            "applied." if self._transcript_available else
            "Transcripts need faster-whisper: pip install faster-whisper")
        self.btn_transcript.clicked.connect(self._choose_transcript)
        bottom.addWidget(self.btn_transcript)
        bottom.addWidget(self.btn_export_audio)
        bottom.addWidget(self.btn_export)
        layout.addLayout(bottom)

        self.setCentralWidget(root)

    # ---------- Actions ----------

    def _choose_input(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open video", "", VIDEO_FILTER)
        if not path:
            return
        self._input_path = path
        self._profile = None
        self._result = None
        self.lbl_file.setText(Path(path).name)
        self.timeline.set_profile(None)
        self.lbl_stats.setText("")
        self._start(AnalysisWorker(path), self._on_analysis_done, "Analyzing audio…")

    def _choose_output(self) -> None:
        if self._input_path is None or self._result is None:
            return
        if not self._result.keep:
            self._show_error("Everything is below the threshold. Lower the threshold.")
            return
        src = Path(self._input_path)
        lossless = self._lossless()
        # Stream copy keeps the source codecs, so the source container is safest.
        suffix = (src.suffix or ".mp4") if lossless else ".mp4"
        first_filter = f"Same as source (*{suffix})" if lossless else "MP4 (*.mp4)"
        filters = f"{first_filter};;MKV (*.mkv)"
        suggested = str(src.with_name(f"{src.stem}_trimmed{suffix}"))
        out, chosen = QFileDialog.getSaveFileName(self, "Export", suggested, filters)
        if not out:
            return
        if not Path(out).suffix:
            out += ".mkv" if "mkv" in chosen.lower() else suffix
        worker = RenderWorker(self._input_path, out, self._result.keep, lossless=lossless)
        self._start(worker, self._on_render_done, "Exporting…")

    def _choose_swap_audio(self) -> None:
        if self._input_path is None or not self._has_video():
            return
        audio, _ = QFileDialog.getOpenFileName(
            self, "Choose the clean audio", "", AUDIO_FILTER)
        if not audio:
            return
        trim = self._ask_swap_trim()
        if trim is None:
            return
        src = Path(self._input_path)
        suffix = src.suffix or ".mp4"
        name = f"{src.stem}_clean_audio{'_trimmed' if trim else ''}{suffix}"
        out, _ = QFileDialog.getSaveFileName(
            self, "Save video with new audio", str(src.with_name(name)),
            f"Same as source (*{suffix})")
        if not out:
            return
        if not Path(out).suffix:
            out += suffix
        keep = self._result.keep if trim else None
        worker = SwapAudioWorker(self._input_path, audio, out, keep=keep,
                                 lossless=self._lossless())
        self._start(worker, self._on_swap_done, "Swapping audio…")

    def _ask_swap_trim(self) -> bool | None:
        """Ask whether to also cut the silence: True/False, or None if cancelled."""
        result = self._result
        if result is None or not result.cuts or not result.keep:
            return False  # nothing to cut, so just swap
        box = QMessageBox(self)
        box.setWindowTitle("Swap audio")
        box.setText("How should the video be exported?")
        box.setInformativeText(
            "Keep full video: only the audio is replaced, nothing is cut.\n"
            f"Also remove silence: apply the {len(self._result.cuts)} cuts shown "
            "on the timeline as well.")
        full = box.addButton("Keep full video", QMessageBox.ButtonRole.AcceptRole)
        trim = box.addButton("Also remove silence", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(full)
        box.exec()
        clicked = box.clickedButton()
        if clicked is full:
            return False
        if clicked is trim:
            return True
        return None

    def _on_swap_done(self, result) -> None:
        self._finish(f"Saved {result.output}")
        message = f"Saved to:\n{result.output}"
        if result.length_warning:
            message += f"\n\n{result.length_warning}"
        QMessageBox.information(self, "Audio swapped", message)

    def _choose_remove_echo(self) -> None:
        if self._input_path is None or self._profile is None:
            return
        src = Path(self._input_path)
        suggested = str(src.with_name(f"{src.stem}_no_echo.mp3"))
        out, chosen = QFileDialog.getSaveFileName(
            self, "Save audio without echo", suggested, ECHO_OUTPUT_FILTER)
        if not out:
            return
        if not Path(out).suffix:
            out += "." + (chosen.split("(*.")[1].rstrip(")") if "(*." in chosen else "mp3")
        if Path(out).resolve() == src.resolve():
            self._show_error("Choose a different file name than the source.")
            return
        worker = RemoveEchoWorker(self._input_path, out)
        self._start(worker, self._on_echo_removed, "Removing echo…")

    def _on_echo_removed(self, result) -> None:
        self._finish(f"Saved {result.output}")
        message = f"{result.summary}\n\nSaved to:\n{result.output}"
        if self._has_video():
            message += "\n\nTo use it in the video, click Swap audio… and pick this file."
        QMessageBox.information(self, "Echo removed", message)

    def _choose_transcript(self) -> None:
        if self._input_path is None or self._profile is None:
            return
        translate = self._ask_transcript_language()
        if translate is None:
            return
        src = Path(self._input_path)
        suggested = str(src.with_name(f"{src.stem}{'_english' if translate else ''}.srt"))
        out, chosen = QFileDialog.getSaveFileName(
            self, "Save transcript", suggested, TRANSCRIPT_FILTER)
        if not out:
            return
        if not Path(out).suffix:
            out += "." + (chosen.split("(*.")[1].rstrip(")") if "(*." in chosen else "srt")
        worker = TranscriptWorker(self._input_path, out, translate)
        self._start(worker, self._on_transcript_done,
                    f"Transcribing… (the first run downloads the '{DEFAULT_MODEL}' "
                    "speech model, about 1.5 GB)")

    def _ask_transcript_language(self) -> bool | None:
        """True to translate to English, False to keep the language, None if cancelled."""
        box = QMessageBox(self)
        box.setWindowTitle("Export transcript")
        box.setText("Which language should the transcript be in?")
        box.setInformativeText(
            "Original language: write down the words as spoken.\n"
            "Translate to English: speech in any language is written in English.")
        original = box.addButton("Original language", QMessageBox.ButtonRole.AcceptRole)
        english = box.addButton("Translate to English", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(original)
        box.exec()
        clicked = box.clickedButton()
        if clicked is original:
            return False
        if clicked is english:
            return True
        return None

    def _on_transcript_done(self, result) -> None:
        self._finish(f"Saved {result.output}")
        QMessageBox.information(self, "Transcript saved",
                                f"{result.summary}\n\nSaved to:\n{result.output}")

    def _has_video(self) -> bool:
        # The keyframe scan finds keyframes only when there is a video track.
        return self._profile is not None and self._keyframes_ms.size > 0

    def _choose_audio_output(self) -> None:
        if self._input_path is None or self._profile is None:
            return
        src = Path(self._input_path)
        suggested = str(src.with_suffix(".mp3"))
        out, _ = QFileDialog.getSaveFileName(self, "Export audio", suggested, "MP3 audio (*.mp3)")
        if not out:
            return
        if Path(out).suffix.lower() != ".mp3":
            out += ".mp3"
        if Path(out).resolve() == src.resolve():
            self._show_error("Choose a different file name than the source.")
            return
        worker = AudioExportWorker(self._input_path, out, DEFAULT_MP3_KBPS)
        self._start(worker, self._on_render_done, "Exporting audio…")

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.statusBar().showMessage("Cancelling…")

    # ---------- Worker plumbing ----------

    def _start(self, worker, on_done, message: str) -> None:
        self._worker = worker
        worker.progress.connect(lambda f: self.progress.setValue(int(f * 1000)))
        worker.finished_ok.connect(on_done)
        worker.failed.connect(self._on_failed)
        worker.cancelled.connect(lambda: self._finish("Cancelled."))
        self.progress.setValue(0)
        self.statusBar().showMessage(message)
        self._set_busy(True)
        worker.start()

    def _finish(self, message: str) -> None:
        if self._worker is not None:
            self._worker.wait()
        self._worker = None
        self._set_busy(False)
        self.statusBar().showMessage(message)

    def _on_analysis_done(self, analysis: MediaAnalysis) -> None:
        self._profile = analysis.profile
        self._keyframes_ms = analysis.keyframes_ms
        self.timeline.set_profile(analysis.profile)
        self._finish("Analysis complete. Adjust the controls, then export.")
        self._refresh_detection()

    def _on_render_done(self, out_path: str) -> None:
        self._finish(f"Saved {out_path}")
        QMessageBox.information(self, "Export complete", f"Saved to:\n{out_path}")

    def _on_failed(self, message: str) -> None:
        self._finish("Failed.")
        self._show_error(message)

    # ---------- Live preview ----------

    def _lossless(self) -> bool:
        return bool(self.export_mode.currentData())

    def _current_mode(self) -> DetectionMode:
        return self.mode.currentData()

    def _on_mode_changed(self) -> None:
        voice = self._current_mode() is DetectionMode.VOICE
        for control, enabled in ((self.vad_threshold, voice), (self.threshold, not voice)):
            control.setEnabled(enabled)
            self._row_labels[control].setEnabled(enabled)
        self._refresh_detection()

    def _refresh_detection(self) -> None:
        if self._profile is None:
            return
        mode = self._current_mode()
        if mode is DetectionMode.VOICE and not self._profile.has_vad:
            self.lbl_stats.setText("Voice detection unavailable for this file.")
            self._result = None
            self.btn_export.setEnabled(False)
            return
        settings = DetectionSettings(
            threshold_db=self.threshold.value(),
            min_silence_ms=self.min_silence.value(),
            padding_ms=self.padding.value(),
            mode=mode,
            vad_threshold=self.vad_threshold.value() / 100,
        )
        precise = detect(self._profile, settings)
        # Lossless can only start segments on keyframes; preview what it will keep.
        self._result = snap_result(precise, self._keyframes_ms) if self._lossless() else precise
        self.timeline.set_detection(self._result, settings)
        r = self._result
        pct = 100 * r.removed_ms / max(1, r.duration_ms)
        text = (f"Original {_fmt(r.duration_ms)} → {_fmt(r.kept_ms)}   "
                f"removed {_fmt(r.removed_ms)} ({pct:.0f}%) in {len(r.cuts)} cuts")
        extra_ms = r.kept_ms - precise.kept_ms
        if self._lossless() and extra_ms > 0:
            text += f"   (+{extra_ms / 1000:.1f}s kept to start on keyframes)"
        self.lbl_stats.setText(text)
        self.btn_export.setEnabled(self._worker is None and bool(r.keep))

    def _set_busy(self, busy: bool) -> None:
        self.btn_open.setEnabled(not busy)
        self.btn_cancel.setEnabled(busy)
        self.btn_export.setEnabled(
            not busy and self._result is not None and bool(self._result.keep)
        )
        self.btn_swap_audio.setEnabled(not busy and self._has_video())
        # A finished analysis means the file has an audio track to extract.
        self.btn_export_audio.setEnabled(not busy and self._profile is not None)
        self.btn_remove_echo.setEnabled(not busy and self._profile is not None)
        self.btn_transcript.setEnabled(
            not busy and self._profile is not None and self._transcript_available)

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Silence Remover", message)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if self._worker is not None:
            self._worker.cancel()
            self._worker.wait()
        super().closeEvent(event)


def _fmt(ms: int) -> str:
    total_s = ms // 1000
    h, rem = divmod(total_s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
