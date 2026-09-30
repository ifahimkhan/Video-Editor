"""Silence detection on an analyzed profile (loudness or voice). Pure, no I/O."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .analysis import LoudnessProfile

THRESHOLD_RANGE_DB = (-70, -10)
VAD_THRESHOLD_RANGE = (0.05, 0.95)
MIN_SILENCE_RANGE_MS = (50, 5000)
PADDING_RANGE_MS = (0, 1000)
# Silero's recommended hysteresis: once speech starts, it continues until the
# probability drops this far below the threshold. Avoids flicker mid-word.
VAD_HYSTERESIS = 0.15
# Speech blips shorter than this (a cough, a click) don't count as speech.
MIN_SPEECH_MS = 100


class DetectionMode(str, Enum):
    LOUDNESS = "loudness"   # dBFS threshold, like Filmora
    VOICE = "voice"         # Silero VAD: ignores hum, music, keyboard noise


@dataclass(frozen=True)
class DetectionSettings:
    """Filmora-style controls.

    mode:           LOUDNESS uses threshold_db, VOICE uses vad_threshold.
    threshold_db:   frames quieter than this count as silence.
    vad_threshold:  speech probability (0..1) above which audio is voice.
    min_silence_ms: only silent stretches at least this long get cut.
    padding_ms:     "softness" kept around speech so words aren't clipped.
    """

    threshold_db: float = -35.0
    min_silence_ms: int = 500
    padding_ms: int = 150
    mode: DetectionMode = DetectionMode.LOUDNESS
    vad_threshold: float = 0.5

    def __post_init__(self) -> None:
        # Accept plain strings ("voice") from the CLI or saved settings.
        object.__setattr__(self, "mode", DetectionMode(self.mode))
        _check_range("threshold_db", self.threshold_db, THRESHOLD_RANGE_DB)
        _check_range("vad_threshold", self.vad_threshold, VAD_THRESHOLD_RANGE)
        _check_range("min_silence_ms", self.min_silence_ms, MIN_SILENCE_RANGE_MS)
        _check_range("padding_ms", self.padding_ms, PADDING_RANGE_MS)


@dataclass(frozen=True)
class Segment:
    start_ms: int
    end_ms: int

    @property
    def length_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass(frozen=True)
class DetectionResult:
    keep: tuple[Segment, ...]
    duration_ms: int

    @property
    def kept_ms(self) -> int:
        return sum(s.length_ms for s in self.keep)

    @property
    def removed_ms(self) -> int:
        return self.duration_ms - self.kept_ms

    @property
    def cuts(self) -> tuple[Segment, ...]:
        """Complement of `keep`: the stretches that will be removed."""
        return invert(self.keep, self.duration_ms)


def detect(profile: LoudnessProfile, settings: DetectionSettings) -> DetectionResult:
    if settings.mode is DetectionMode.VOICE:
        if profile.speech_prob is None:
            raise ValueError("Voice detection is not available for this file.")
        speech = voice_mask(profile.speech_prob, settings.vad_threshold,
                            MIN_SPEECH_MS // profile.vad_frame_ms)
        runs = _silent_runs(~speech, profile.vad_frame_ms)
        # VAD windows can run a few ms past the decoded audio.
        runs = tuple(Segment(r.start_ms, min(r.end_ms, profile.duration_ms))
                     for r in runs if r.start_ms < profile.duration_ms)
    else:
        runs = _silent_runs(profile.db < settings.threshold_db, profile.frame_ms)
    long_runs = tuple(r for r in runs if r.length_ms >= settings.min_silence_ms)
    speech = invert(long_runs, profile.duration_ms)
    keep = pad_and_merge(speech, settings.padding_ms, profile.duration_ms)
    return DetectionResult(keep=keep, duration_ms=profile.duration_ms)


def pad_and_merge(
    segments: tuple[Segment, ...], padding_ms: int, duration_ms: int
) -> tuple[Segment, ...]:
    """Grow every segment by `padding_ms` on both sides; merge overlaps."""
    merged: list[Segment] = []
    for seg in segments:
        start = max(0, seg.start_ms - padding_ms)
        end = min(duration_ms, seg.end_ms + padding_ms)
        if merged and start <= merged[-1].end_ms:
            merged[-1] = Segment(merged[-1].start_ms, max(end, merged[-1].end_ms))
        else:
            merged.append(Segment(start, end))
    return tuple(s for s in merged if s.length_ms > 0)


def invert(segments: tuple[Segment, ...], duration_ms: int) -> tuple[Segment, ...]:
    """Gaps between sorted, non-overlapping segments within [0, duration]."""
    gaps: list[Segment] = []
    cursor = 0
    for seg in segments:
        if seg.start_ms > cursor:
            gaps.append(Segment(cursor, seg.start_ms))
        cursor = max(cursor, seg.end_ms)
    if cursor < duration_ms:
        gaps.append(Segment(cursor, duration_ms))
    return tuple(gaps)


def voice_mask(prob: np.ndarray, threshold: float, min_speech_frames: int) -> np.ndarray:
    """Speech/non-speech per frame with hysteresis and blip removal."""
    low = max(0.01, threshold - VAD_HYSTERESIS)
    mask = np.zeros(prob.shape, dtype=bool)
    speaking = False
    for i, p in enumerate(prob.tolist()):
        if speaking:
            speaking = p >= low
        else:
            speaking = p >= threshold
        mask[i] = speaking
    if min_speech_frames > 1:
        for run in _runs(mask):
            if run[1] - run[0] < min_speech_frames:
                mask[run[0]:run[1]] = False
    return mask


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of True runs."""
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    edges = np.diff(padded)
    return list(zip(np.flatnonzero(edges == 1).tolist(),
                    np.flatnonzero(edges == -1).tolist()))


def _silent_runs(silent: np.ndarray, frame_ms: int) -> tuple[Segment, ...]:
    return tuple(Segment(s * frame_ms, e * frame_ms) for s, e in _runs(silent))


def _check_range(name: str, value: float, bounds: tuple[int, int]) -> None:
    low, high = bounds
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}, got {value}")
