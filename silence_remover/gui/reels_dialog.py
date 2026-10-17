"""Create reels: pick settings, find the best clips, review them, render.

1. Find clips   - transcribe (cached), score the moments, find the webcam.
   Or add your own time ranges ("Add your own clip"); Find is optional then.
2. Review       - tick the clips to keep, edit their titles, preview the crop.
3. Create reels - render the ticked clips into the output folder.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ..ffmpeg_tools import FFmpegError, probe
from ..reels.frames import grab_frames
from ..reels.highlights import Highlight
from ..reels.manual import manual_clip
from ..reels.preview import preview_boxes
from ..reels.timecode import parse_time
from .frame_preview import FramePreview
from .reels_settings_form import ReelSettingsForm
from .reels_workers import PrepareReelsWorker, ReelsPreparation, RenderReelsWorker

log = logging.getLogger(__name__)

COLUMNS = ["Use", "Time", "Length", "Score", "Title"]
TITLE_COLUMN = 4


class ReelsDialog(QDialog):
    def __init__(self, input_path: str, parent=None) -> None:
        super().__init__(parent)
        self.input_path = input_path
        self.setWindowTitle(f"Create reels - {Path(input_path).name}")
        self.resize(1200, 720)
        self._worker = None
        self._prep: ReelsPreparation | None = None
        self._picks: list[Highlight] = []
        self._saved_folder: Path | None = None
        self._frames: dict[int, object] = {}  # row -> preview frame
        self._duration_s, self._frame_size = _probe_video(input_path)
        self._build_ui()
        self._set_busy(False)

    # ---------- UI ----------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        body = QHBoxLayout()
        left = QVBoxLayout()
        self.form = ReelSettingsForm()
        self.form.changed.connect(self._refresh_preview)
        left.addWidget(self.form)
        folder = QHBoxLayout()
        src = Path(self.input_path)
        self.out_dir = QLineEdit(str(src.with_name(f"{src.stem}_reels")))
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._choose_folder)
        folder.addWidget(QLabel("Save to"))
        folder.addWidget(self.out_dir, 1)
        folder.addWidget(browse)
        left.addLayout(folder)
        self.btn_find = QPushButton("Find clips")
        self.btn_find.setToolTip("Transcribe the video (once; it is cached in the output "
                                 "folder) and pick its best moments.")
        self.btn_find.clicked.connect(self._find)
        left.addWidget(self.btn_find)
        left.addStretch(1)
        body.addLayout(left)

        right = QVBoxLayout()
        self.preview = FramePreview()
        right.addWidget(self.preview, 3)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(
            TITLE_COLUMN, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.itemSelectionChanged.connect(self._refresh_preview)
        right.addWidget(self.table, 2)
        right.addWidget(QLabel("Tick the clips to keep; double-click a title to edit it. "
                               "Changing length or number of clips needs Find clips again."))
        right.addLayout(self._build_manual_row())
        body.addLayout(right, 1)
        root.addLayout(body, 1)

        bottom = QHBoxLayout()
        self.status = QLabel("")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self._cancel)
        self.btn_create = QPushButton("Create reels")
        self.btn_create.clicked.connect(self._create)
        self.btn_open_folder = QPushButton("Open folder")
        self.btn_open_folder.setEnabled(False)
        self.btn_open_folder.clicked.connect(self._open_folder)
        bottom.addWidget(self.status, 1)
        bottom.addWidget(self.progress, 1)
        bottom.addWidget(self.btn_cancel)
        bottom.addWidget(self.btn_open_folder)
        bottom.addWidget(self.btn_create)
        root.addLayout(bottom)

    def _build_manual_row(self) -> QHBoxLayout:
        """Start / End / Title fields to add a moment the AI may have missed."""
        row = QHBoxLayout()
        self.manual_start = QLineEdit()
        self.manual_start.setPlaceholderText("Start, e.g. 1:25")
        self.manual_end = QLineEdit()
        self.manual_end.setPlaceholderText("End, e.g. 2:10")
        self.manual_title = QLineEdit()
        self.manual_title.setPlaceholderText("Title (optional)")
        self.btn_add_clip = QPushButton("Add clip")
        tip = ("Add your own time range as a clip. Times: seconds (85), m:ss (1:25) "
               "or h:mm:ss (1:02:05). Works without Find clips too.")
        for widget in (self.manual_start, self.manual_end, self.manual_title,
                       self.btn_add_clip):
            widget.setToolTip(tip)
        for field in (self.manual_start, self.manual_end, self.manual_title):
            field.returnPressed.connect(self._add_manual)
        self.btn_add_clip.clicked.connect(self._add_manual)
        row.addWidget(QLabel("Add your own clip:"))
        row.addWidget(self.manual_start)
        row.addWidget(self.manual_end)
        row.addWidget(self.manual_title, 1)
        row.addWidget(self.btn_add_clip)
        return row

    # ---------- Actions ----------

    def _add_manual(self) -> None:
        if self._worker is not None:
            return
        try:
            start = parse_time(self.manual_start.text())
            end = parse_time(self.manual_end.text())
            clip = manual_clip(start, end, self._duration_s, self.manual_title.text())
        except ValueError as exc:
            self._show_error(str(exc))
            return
        self._picks.append(clip)
        row = len(self._picks) - 1
        self._add_row(row, clip)
        for field in (self.manual_start, self.manual_end, self.manual_title):
            field.clear()
        self.table.selectRow(row)
        self._set_busy(False)
        self.status.setText(f"Added {_clock(clip.start_s)}-{_clock(clip.end_s)}.")

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Save reels to", self.out_dir.text())
        if folder:
            self.out_dir.setText(folder)

    def _find(self) -> None:
        try:
            settings = self.form.settings()
        except ValueError as exc:
            self._show_error(str(exc))
            return
        worker = PrepareReelsWorker(self.input_path, settings, self.form.ranker_name(),
                                    self._output_folder())
        self._start(worker, self._on_prepared,
                    "Finding clips… (the first run transcribes the video, which can "
                    "take a while)")

    def _create(self) -> None:
        picks = self._chosen_picks()
        if not picks:
            self._show_error("Tick at least one clip to create.")
            return
        try:
            settings = self.form.settings()
        except ValueError as exc:
            self._show_error(str(exc))
            return
        webcam = self._prep.webcam if self._prep is not None else None
        if settings.webcam is None and webcam is not None:
            settings = dataclasses.replace(settings, webcam=webcam)
        # Without Find clips (manual clips only), the worker fetches what it needs.
        prepared = self._prep.prepared if self._prep is not None else None
        worker = RenderReelsWorker(self.input_path, self._output_folder(), settings,
                                   prepared, picks)
        self._start(worker, self._on_rendered, f"Creating {len(picks)} reel(s)…")

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.status.setText("Cancelling…")

    # ---------- Results ----------

    def _on_prepared(self, prep: ReelsPreparation) -> None:
        self._prep = prep
        # Clips the user added by hand stay; the found ones are replaced.
        manual = [p for p in self._picks if p.manual]
        self._picks = list(prep.prepared.picks) + manual
        self._frames = {}
        self.preview.set_frame(None)
        self.table.clearSelection()
        self._fill_table()
        found = len(prep.prepared.picks)
        self._finish(f"Found {found} clip(s)." if found else
                     "No clip fits the length limits. Try a shorter minimum length.")
        if self._picks:
            self.table.selectRow(0)

    def _on_rendered(self, run) -> None:
        self._finish(f"Saved {len(run.reels)} reel(s).")
        lines = "\n".join(f"{r.output.name}  ({r.length_s:.0f} s)  {r.highlight.title}"
                          for r in run.reels)
        self._saved_folder = run.summary_path.parent
        self.btn_open_folder.setEnabled(True)
        QMessageBox.information(self, "Reels created",
                                f"{lines}\n\nSaved to:\n{self._saved_folder}")

    def _open_folder(self) -> None:
        if self._saved_folder is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._saved_folder)))

    def _fill_table(self) -> None:
        self.table.setRowCount(0)
        for row, pick in enumerate(self._picks):
            self._add_row(row, pick)

    def _add_row(self, row: int, pick: Highlight) -> None:
        """Append one ticked row; other rows keep their ticks and edited titles."""
        self.table.insertRow(row)
        use = QTableWidgetItem()
        use.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
                     | Qt.ItemFlag.ItemIsSelectable)
        use.setCheckState(Qt.CheckState.Checked)
        score = "manual" if pick.manual else f"{pick.score:.1f}"
        cells = [use, _fixed(f"{_clock(pick.start_s)}-{_clock(pick.end_s)}"),
                 _fixed(f"{pick.length_s:.0f} s"), _fixed(score), QTableWidgetItem(pick.title)]
        tip = "\n".join(t for t in (pick.hook, pick.reason) if t)
        for col, item in enumerate(cells):
            if tip:
                item.setToolTip(tip)
            self.table.setItem(row, col, item)

    def _chosen_picks(self) -> list[Highlight]:
        chosen = []
        for row, pick in enumerate(self._picks):
            if self.table.item(row, 0).checkState() == Qt.CheckState.Checked:
                title = self.table.item(row, TITLE_COLUMN).text().strip() or pick.title
                chosen.append(dataclasses.replace(pick, title=title))
        return chosen

    def _refresh_preview(self) -> None:
        """Redraw the selected clip's frame with the current layout's boxes.

        Runs in a Qt slot, where an escaping exception would abort the app,
        so every failure just keeps the previous preview.
        """
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows or rows[0].row() >= len(self._picks) or not all(self._frame_size):
            return
        try:
            frame = self._preview_frame(rows[0].row())
            if frame is None:
                return
            settings = self.form.settings()
            found_webcam = self._prep.webcam if self._prep is not None else None
            w, h = self._frame_size
            boxes = preview_boxes(settings, w, h, settings.webcam or found_webcam)
            self.preview.set_frame(frame, boxes)
        except Exception:  # noqa: BLE001 - see docstring
            log.exception("Could not draw the reel preview")

    def _preview_frame(self, row: int):
        """The clip's middle frame, read once per clip (settings only redraw boxes)."""
        if row not in self._frames:
            pick = self._picks[row]
            try:
                frames = grab_frames(self.input_path, [(pick.start_s + pick.end_s) / 2],
                                     self._frame_size)
            except (FFmpegError, OSError):
                frames = []
            self._frames[row] = frames[0] if frames else None
        return self._frames[row]

    # ---------- Worker plumbing ----------

    def _start(self, worker, on_done, message: str) -> None:
        self._worker = worker
        worker.progress.connect(lambda f: self.progress.setValue(int(f * 1000)))
        worker.finished_ok.connect(on_done)
        worker.failed.connect(self._on_failed)
        worker.cancelled.connect(lambda: self._finish("Cancelled."))
        self.progress.setValue(0)
        self.status.setText(message)
        self._set_busy(True)
        worker.start()

    def _finish(self, message: str) -> None:
        if self._worker is not None:
            self._worker.wait()
        self._worker = None
        self._set_busy(False)
        self.status.setText(message)

    def _on_failed(self, message: str) -> None:
        self._finish("Failed.")
        self._show_error(message)

    def _set_busy(self, busy: bool) -> None:
        self.btn_find.setEnabled(not busy)
        for widget in (self.manual_start, self.manual_end, self.manual_title,
                       self.btn_add_clip):
            widget.setEnabled(not busy)
        self.form.setEnabled(not busy)
        self.btn_cancel.setEnabled(busy)
        self.btn_create.setEnabled(not busy and bool(self._picks))

    def _output_folder(self) -> str:
        return self.out_dir.text().strip() or str(Path(self.input_path).parent)

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Create reels", message)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        self._stop_worker()
        super().closeEvent(event)

    def reject(self) -> None:
        # Esc / close button: QDialog only hides itself, so stop the job here.
        self._stop_worker()
        super().reject()

    def _stop_worker(self) -> None:
        worker, self._worker = self._worker, None
        if worker is not None:
            for signal in (worker.finished_ok, worker.failed, worker.cancelled):
                signal.disconnect()
            worker.cancel()
            worker.wait()


def _probe_video(input_path: str) -> tuple[float, tuple[int, int]]:
    """Duration and picture size, or zeros when the file can't be read."""
    try:
        info = probe(input_path)
    except (FFmpegError, OSError):
        return 0.0, (0, 0)
    return info.duration_s, (info.width, info.height)


def _fixed(text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
    return item


def _clock(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}:{secs:02d}"
