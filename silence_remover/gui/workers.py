"""Background threads so analysis and export never freeze the UI."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from ..analysis import AnalysisCancelled, LoudnessProfile, analyze
from ..ffmpeg_tools import FFmpegError
from ..lossless import KeyframeScanCancelled, read_keyframes, render_lossless
from ..render import RenderCancelled, render
from ..segments import Segment
from ..swap_audio import swap_audio

AUDIO_PHASE = 0.85  # share of the analysis progress bar; keyframe scan gets the rest


@dataclass(frozen=True)
class MediaAnalysis:
    profile: LoudnessProfile
    keyframes_ms: np.ndarray  # empty for audio-only files


class _CancellableWorker(QThread):
    progress = pyqtSignal(float)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def _is_cancelled(self) -> bool:
        return self._cancel.is_set()

    def _guarded(self, job, cancelled_types: tuple[type[BaseException], ...]) -> None:
        """Run `job`, turning every outcome into exactly one signal."""
        try:
            result = job()
        except cancelled_types:
            self.cancelled.emit()
        except (FFmpegError, OSError, ValueError) as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # an escaped exception would abort the app
            self.failed.emit(f"Unexpected error: {exc!r}")
        else:
            self.finished_ok.emit(result)


class AnalysisWorker(_CancellableWorker):
    finished_ok = pyqtSignal(object)  # MediaAnalysis

    def __init__(self, media_path: str) -> None:
        super().__init__()
        self._media_path = media_path

    def run(self) -> None:
        self._guarded(self._analyze, (AnalysisCancelled, KeyframeScanCancelled))

    def _analyze(self) -> MediaAnalysis:
        profile = analyze(self._media_path,
                          lambda f: self.progress.emit(f * AUDIO_PHASE),
                          self._is_cancelled)
        keyframes = read_keyframes(self._media_path, self._is_cancelled)
        self.progress.emit(1.0)
        return MediaAnalysis(profile=profile, keyframes_ms=keyframes)


class RenderWorker(_CancellableWorker):
    finished_ok = pyqtSignal(object)  # output path as str

    def __init__(self, in_path: str, out_path: str, keep: tuple[Segment, ...],
                 lossless: bool = False) -> None:
        super().__init__()
        self._in_path = in_path
        self._out_path = out_path
        self._keep = keep
        self._lossless = lossless

    def run(self) -> None:
        self._guarded(self._render, (RenderCancelled,))

    def _render(self) -> str:
        export = render_lossless if self._lossless else render
        out = export(self._in_path, self._out_path, self._keep,
                     on_progress=self.progress.emit, is_cancelled=self._is_cancelled)
        return str(out)


class SwapAudioWorker(_CancellableWorker):
    finished_ok = pyqtSignal(object)  # SwapResult

    def __init__(self, video_path: str, audio_path: str, out_path: str) -> None:
        super().__init__()
        self._video_path = video_path
        self._audio_path = audio_path
        self._out_path = out_path

    def run(self) -> None:
        self._guarded(self._swap, (RenderCancelled,))

    def _swap(self):
        return swap_audio(self._video_path, self._audio_path, self._out_path,
                          on_progress=self.progress.emit, is_cancelled=self._is_cancelled)
