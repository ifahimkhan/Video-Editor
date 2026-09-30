"""Audio analysis: streams audio out of ffmpeg once and measures, per frame,
loudness (dBFS) and, when available, Silero speech probability.

Everything is computed once per file. Silence detection then runs on these
arrays alone, so slider changes in the UI are instant.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .ffmpeg_tools import CREATE_NO_WINDOW, FFmpegError, find_binary, probe
from .vad import VAD_FRAME_MS, VAD_SAMPLE_RATE, WINDOW_SAMPLES, SileroVad, load_vad

SAMPLE_RATE = VAD_SAMPLE_RATE  # 16 kHz: plenty for speech, required by Silero
FRAME_MS = 10                  # loudness envelope resolution
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS // 1000
BYTES_PER_SAMPLE = 2           # s16le
# Chunks must hold whole loudness frames (160) and whole VAD windows (512).
BLOCK_SAMPLES = int(np.lcm(SAMPLES_PER_FRAME, WINDOW_SAMPLES))
READ_CHUNK_SAMPLES = BLOCK_SAMPLES * 32   # ~5 s per read
FLOOR_DB = -100.0              # value used for digital silence

ProgressCallback = Callable[[float], None]
CancelCheck = Callable[[], bool]


class AnalysisCancelled(Exception):
    pass


@dataclass(frozen=True)
class LoudnessProfile:
    """Per-frame measurements of the source audio.

    db:          loudness (RMS) in dBFS, one value per `frame_ms`.
    peak_db:     peak sample level in dBFS per `frame_ms`, for the waveform.
    speech_prob: Silero speech probability (0..1), one value per
                 `vad_frame_ms`; None when voice detection was unavailable.
    """

    db: np.ndarray
    frame_ms: int
    duration_ms: int
    speech_prob: np.ndarray | None = None
    vad_frame_ms: int = VAD_FRAME_MS
    peak_db: np.ndarray | None = None

    def __post_init__(self) -> None:
        for arr in (self.db, self.speech_prob, self.peak_db):
            if arr is not None:
                arr.setflags(write=False)

    @property
    def has_vad(self) -> bool:
        return self.speech_prob is not None


def frames_to_db(samples: np.ndarray) -> np.ndarray:
    """RMS level in dBFS for each full frame of int16 samples."""
    frames = _frames(samples)
    return _to_db(np.sqrt(np.mean(np.square(frames), axis=1)))


def frames_to_peak_db(samples: np.ndarray) -> np.ndarray:
    """Peak level in dBFS for each full frame of int16 samples."""
    frames = _frames(samples)
    return _to_db(np.max(np.abs(frames), axis=1, initial=0.0))


def _frames(samples: np.ndarray) -> np.ndarray:
    usable = len(samples) - len(samples) % SAMPLES_PER_FRAME
    return (samples[:usable].astype(np.float32) / 32768.0).reshape(-1, SAMPLES_PER_FRAME)


def _to_db(linear: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(linear)
    return np.maximum(db, FLOOR_DB).astype(np.float32)


def analyze(
    media_path: str | Path,
    on_progress: ProgressCallback | None = None,
    is_cancelled: CancelCheck | None = None,
    use_vad: bool = True,
) -> LoudnessProfile:
    """Decode the first audio track to 16 kHz mono and measure it."""
    info = probe(media_path)
    if not info.has_audio:
        raise FFmpegError("The selected file has no audio track.")

    cmd = [
        find_binary("ffmpeg"), "-nostdin", "-v", "error", "-i", str(media_path),
        "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1",
    ]
    expected_samples = max(1, int(info.duration_s * SAMPLE_RATE))
    meter = _Meter(load_vad() if use_vad else None)

    with tempfile.TemporaryFile() as err_file:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=err_file,
            creationflags=CREATE_NO_WINDOW,
        )
        try:
            _pump(proc, meter, expected_samples, on_progress, is_cancelled)
        finally:
            _stop(proc)

        if proc.returncode not in (0, None):
            err_file.seek(0)
            detail = err_file.read().decode(errors="replace").strip()
            raise FFmpegError(f"Audio decoding failed: {detail[-500:]}")

    return meter.result()


class _Meter:
    """Accumulates loudness and speech probability chunk by chunk."""

    def __init__(self, vad: SileroVad | None) -> None:
        self._vad = vad
        self._db: list[np.ndarray] = []
        self._peak: list[np.ndarray] = []
        self._prob: list[np.ndarray] = []

    def feed(self, samples: np.ndarray) -> None:
        self._db.append(frames_to_db(samples))
        self._peak.append(frames_to_peak_db(samples))
        if self._vad is not None:
            self._prob.append(self._vad.process(samples.astype(np.float32) / 32768.0))

    def finish(self, tail: np.ndarray) -> None:
        """Last partial block: loudness drops the partial frame, VAD pads."""
        if tail.size == 0:
            return
        self._db.append(frames_to_db(tail))
        self._peak.append(frames_to_peak_db(tail))
        if self._vad is not None:
            padded = np.zeros(-(-tail.size // WINDOW_SAMPLES) * WINDOW_SAMPLES, np.float32)
            padded[:tail.size] = tail / 32768.0
            self._prob.append(self._vad.process(padded))

    def result(self) -> LoudnessProfile:
        db = _join(self._db)
        prob = None
        if self._vad is not None:
            prob = _join(self._prob)
        return LoudnessProfile(
            db=db, frame_ms=FRAME_MS, duration_ms=len(db) * FRAME_MS,
            speech_prob=prob, peak_db=_join(self._peak),
        )


def _join(chunks: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(chunks) if chunks else np.empty(0, np.float32)


def _pump(
    proc: subprocess.Popen,
    meter: _Meter,
    expected_samples: int,
    on_progress: ProgressCallback | None,
    is_cancelled: CancelCheck | None,
) -> None:
    assert proc.stdout is not None
    chunk_bytes = READ_CHUNK_SAMPLES * BYTES_PER_SAMPLE
    block_bytes = BLOCK_SAMPLES * BYTES_PER_SAMPLE
    leftover = b""
    samples_done = 0

    while True:
        if is_cancelled is not None and is_cancelled():
            raise AnalysisCancelled()
        data = proc.stdout.read(chunk_bytes)
        if not data:
            break
        buf = leftover + data
        usable = len(buf) - len(buf) % block_bytes
        leftover = buf[usable:]
        if usable:
            meter.feed(np.frombuffer(buf[:usable], dtype="<i2"))
        samples_done += usable // BYTES_PER_SAMPLE
        if on_progress is not None:
            on_progress(min(1.0, samples_done / expected_samples))

    usable_tail = len(leftover) - len(leftover) % BYTES_PER_SAMPLE
    meter.finish(np.frombuffer(leftover[:usable_tail], dtype="<i2"))
    proc.wait()


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.kill()
        proc.wait()
    if proc.stdout is not None:
        proc.stdout.close()
