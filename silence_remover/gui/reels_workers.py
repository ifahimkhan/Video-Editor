"""Background threads for the reels dialog: pick the clips, then render them."""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import pyqtSignal

from ..analysis import AnalysisCancelled
from ..ffmpeg_tools import probe
from ..reels.analyze import resolve_webcam
from ..reels.highlights import Highlight
from ..reels.pipeline import Prepared, prepare_reels, render_reels
from ..reels.settings import Rect, ReelSettings
from ..render import RenderCancelled
from ..transcript import TranscriptCancelled
from .workers import _CancellableWorker

CANCELLED = (RenderCancelled, TranscriptCancelled, AnalysisCancelled)
WEBCAM_SHARE = 0.05  # finding the webcam is quick next to transcribing


@dataclass(frozen=True)
class ReelsPreparation:
    prepared: Prepared
    webcam: Rect | None  # None when the video is too small or has no picture
    frame_size: tuple[int, int]


class PrepareReelsWorker(_CancellableWorker):
    """Transcribe (cached), pick the clips and find the webcam."""

    finished_ok = pyqtSignal(object)  # ReelsPreparation

    def __init__(self, input_path: str, settings: ReelSettings, ranker: str,
                 cache_dir: str) -> None:
        super().__init__()
        self._input_path = input_path
        self._settings = settings
        self._ranker = ranker
        self._cache_dir = cache_dir

    def run(self) -> None:
        self._guarded(self._prepare, CANCELLED)

    def _prepare(self) -> ReelsPreparation:
        prepared = prepare_reels(
            self._input_path, self._settings, self._ranker, cache_dir=self._cache_dir,
            on_progress=lambda f: self.progress.emit(f * (1 - WEBCAM_SHARE)),
            is_cancelled=self._is_cancelled)
        info = probe(self._input_path)
        try:
            webcam = resolve_webcam(self._input_path, info, self._settings)
        except ValueError:  # a manual webcam box that doesn't fit: not needed to pick clips
            webcam = None
        self.progress.emit(1.0)
        return ReelsPreparation(prepared, webcam, (info.width, info.height))


class RenderReelsWorker(_CancellableWorker):
    finished_ok = pyqtSignal(object)  # ReelsRun

    def __init__(self, input_path: str, out_dir: str, settings: ReelSettings,
                 prepared: Prepared, picks: list[Highlight]) -> None:
        super().__init__()
        self._input_path = input_path
        self._out_dir = out_dir
        self._settings = settings
        self._prepared = prepared
        self._picks = picks

    def run(self) -> None:
        self._guarded(self._render, CANCELLED)

    def _render(self):
        return render_reels(self._input_path, self._out_dir, self._settings, self._prepared,
                            self._picks, on_progress=self.progress.emit,
                            is_cancelled=self._is_cancelled)
