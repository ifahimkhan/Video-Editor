"""Export a transcript (captions) of a file's speech with faster-whisper.

The first audio track is decoded to 16 kHz mono and passed to Whisper,
which skips non-speech with its built-in Silero VAD filter. Speech can be
written down in its own language or translated to English. The output
format follows the extension: .srt or .vtt captions, or plain .txt.

faster-whisper is optional. Without it `transcript_available()` is False
and `load_model()` raises TranscriptUnavailable. The first run of each
model size downloads it (medium is about 1.5 GB).
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .ffmpeg_tools import CREATE_NO_WINDOW, FFmpegError, find_binary, probe

SAMPLE_RATE = 16_000  # what Whisper expects
MODEL_SIZES = ("tiny", "base", "small", "medium", "large-v3")
DEFAULT_MODEL = "medium"
FORMATS = (".srt", ".vtt", ".txt")
MIN_SILENCE_MS = 500  # VAD: pauses at least this long split captions
BEAM_SIZE = 5
GPU_ENV_VAR = "USE_GPU"  # "1" runs Whisper on an NVIDIA GPU
# Progress shares: decoding, then loading the model, then transcribing.
DECODE_SHARE = 0.1
LOAD_SHARE = 0.05

Progress = Callable[[float], None]
Cancelled = Callable[[], bool]


class TranscriptUnavailable(RuntimeError):
    """faster-whisper is missing or its model could not be loaded."""


class TranscriptCancelled(Exception):
    pass


@dataclass(frozen=True)
class Caption:
    start_s: float
    end_s: float
    text: str


@dataclass(frozen=True)
class TranscriptResult:
    output: Path
    language: str
    language_probability: float
    captions: int
    translated: bool

    @property
    def summary(self) -> str:
        spoken = f"Detected language: {self.language} ({self.language_probability:.0%} sure)"
        action = "Translated to English" if self.translated else "Transcribed"
        return f"{spoken}. {action}: {self.captions} captions."


def transcript_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False
    return True


def load_model(model_size: str):
    """Load a faster-whisper model, downloading it on first use."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscriptUnavailable(
            "Transcripts need faster-whisper: pip install faster-whisper") from exc
    device = "cuda" if os.environ.get(GPU_ENV_VAR) == "1" else "cpu"
    try:
        return WhisperModel(model_size, device=device, compute_type="int8")
    except Exception as exc:  # download, CUDA and file errors all land here
        raise TranscriptUnavailable(
            f"Could not load the '{model_size}' speech model: {exc}") from exc


def format_timestamp(seconds: float, sep: str = ",") -> str:
    """HH:MM:SS,mmm (SRT) or, with sep='.', HH:MM:SS.mmm (WebVTT)."""
    total_ms = max(0, round(seconds * 1000))
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"


def render_captions(captions: Iterable[Caption], suffix: str) -> str:
    """The text of a caption file in the format named by `suffix`."""
    suffix = suffix.lower()
    if suffix not in FORMATS:
        raise ValueError("Transcript file must have one of the extensions "
                         + ", ".join(FORMATS))
    captions = tuple(captions)
    lines = [c.text.strip() for c in captions]
    if suffix == ".txt":
        return "".join(f"{line}\n" for line in lines)
    sep = "," if suffix == ".srt" else "."
    blocks = []
    for index, (caption, line) in enumerate(zip(captions, lines, strict=True), start=1):
        timing = (f"{format_timestamp(caption.start_s, sep)} --> "
                  f"{format_timestamp(caption.end_s, sep)}")
        blocks.append(f"{index}\n{timing}\n{line}\n" if suffix == ".srt"
                      else f"{timing}\n{line}\n")
    header = "WEBVTT\n\n" if suffix == ".vtt" else ""
    return header + "\n".join(blocks)


def transcribe(
    input_path: str | Path,
    output_path: str | Path,
    translate: bool = False,
    model_size: str = DEFAULT_MODEL,
    on_progress: Progress | None = None,
    is_cancelled: Cancelled | None = None,
) -> TranscriptResult:
    """Write the speech of `input_path` to `output_path` as captions or text.

    With `translate`, speech in any language is written in English.
    Raises ValueError for bad options, FFmpegError for unreadable files,
    TranscriptUnavailable without faster-whisper and TranscriptCancelled
    when `is_cancelled` returns True. A cancelled run writes nothing.
    """
    input_path, output_path = Path(input_path), Path(output_path)
    suffix = output_path.suffix.lower()
    if suffix not in FORMATS:
        raise ValueError("Transcript file must have one of the extensions "
                         + ", ".join(FORMATS))
    if model_size not in MODEL_SIZES:
        raise ValueError(f"Speech model must be one of {', '.join(MODEL_SIZES)}")
    report = on_progress or (lambda _f: None)
    cancelled = is_cancelled or (lambda: False)

    audio = decode_audio(input_path)
    _check(cancelled)
    report(DECODE_SHARE)
    model = load_model(model_size)
    _check(cancelled)
    report(DECODE_SHARE + LOAD_SHARE)

    segments, info = model.transcribe(
        audio, beam_size=BEAM_SIZE, vad_filter=True,
        task="translate" if translate else "transcribe",
        vad_parameters={"min_silence_duration_ms": MIN_SILENCE_MS},
    )
    duration_s = max(audio.size / SAMPLE_RATE, 1e-3)
    captions = []
    for segment in segments:  # lazy: each step runs the model on more audio
        _check(cancelled)
        captions.append(Caption(segment.start, segment.end, segment.text))
        done = min(1.0, segment.end / duration_s)
        report(DECODE_SHARE + LOAD_SHARE + done * (1 - DECODE_SHARE - LOAD_SHARE))

    _write_atomically(output_path, render_captions(captions, suffix))
    report(1.0)
    return TranscriptResult(output=output_path, language=info.language,
                            language_probability=info.language_probability,
                            captions=len(captions), translated=translate)


def decode_audio(input_path: Path) -> np.ndarray:
    """The first audio track as 16 kHz mono float32 in [-1, 1]."""
    info = probe(input_path)
    if not info.has_audio:
        raise FFmpegError("The selected file has no audio track to transcribe.")
    cmd = [
        find_binary("ffmpeg"), "-nostdin", "-v", "error", "-i", str(input_path),
        "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, creationflags=CREATE_NO_WINDOW)
    if proc.returncode != 0:
        detail = proc.stderr.decode(errors="replace").strip()
        raise FFmpegError(f"Audio decoding failed: {detail[-500:]}")
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def _check(cancelled: Cancelled) -> None:
    if cancelled():
        raise TranscriptCancelled()


def _write_atomically(path: Path, text: str) -> None:
    tmp = path.with_name(f"{path.name}.part")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
