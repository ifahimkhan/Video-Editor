"""Remove the echo of a doubled voice (two microphones recording at once).

Two passes over the audio, both streamed through ffmpeg so hour-long files
never sit in memory: the first measures the echo filter in every window
(see `echo_dsp`), the second runs the inverse filter and encodes the result.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Callable, Iterator

import numpy as np

from .echo_dsp import (
    WINDOW_S,
    EchoCanceller,
    TimedFilter,
    WindowEstimate,
    build_schedule,
    estimate_filter,
)
from .ffmpeg_tools import CREATE_NO_WINDOW, FFmpegError, find_binary, probe
from .render import RenderCancelled

PROCESS_SR = 48_000
CHUNK_FRAMES = PROCESS_SR  # one second per read/write
MIN_WINDOW_S = 2.0         # a shorter tail window is too little to measure
ANALYSIS_SHARE = 0.4       # of the progress bar; cleaning gets the rest
MAX_CHANNELS = 2
OUTPUT_CODECS = {
    ".mp3": ["-c:a", "libmp3lame", "-b:a", "192k", "-id3v2_version", "3"],
    ".wav": ["-c:a", "pcm_s16le"],
    ".flac": ["-c:a", "flac"],
    ".m4a": ["-c:a", "aac", "-b:a", "192k"],
}

Progress = Callable[[float], None]
Cancelled = Callable[[], bool]


class EchoNotFound(ValueError):
    """The recording has no measurable echo, so there is nothing to remove."""


@dataclass(frozen=True)
class EchoResult:
    output: Path
    delay_ms: float
    windows_total: int
    windows_cleaned: int

    @property
    def summary(self) -> str:
        return (f"Echo found at {self.delay_ms:.1f} ms; removed in "
                f"{self.windows_cleaned} of {self.windows_total} parts of the audio.")


def remove_echo(
    input_path: str | Path,
    output_path: str | Path,
    on_progress: Progress | None = None,
    is_cancelled: Cancelled | None = None,
) -> EchoResult:
    """Write `output_path`: the audio of `input_path` with its echo removed.

    The output format follows the extension (.mp3, .wav, .flac or .m4a).
    Raises EchoNotFound when there is no echo, FFmpegError for unreadable
    files, and RenderCancelled when `is_cancelled` returns True.
    """
    input_path, output_path = Path(input_path), Path(output_path)
    if output_path.suffix.lower() not in OUTPUT_CODECS:
        raise ValueError("Output file must have one of the extensions "
                         + ", ".join(OUTPUT_CODECS))
    if input_path.resolve() == output_path.resolve() or (
            output_path.exists() and input_path.exists()
            and os.path.samefile(input_path, output_path)):
        raise FFmpegError("Output file must be different from the input file.")
    info = probe(input_path)
    if not info.has_audio:
        raise FFmpegError("The selected file has no audio track.")

    report = on_progress or (lambda _f: None)
    cancelled = is_cancelled or (lambda: False)
    channels = min(MAX_CHANNELS, max(1, info.audio_channels))
    total = max(1, int(info.duration_s * PROCESS_SR))

    estimates = _measure(input_path, total,
                         lambda f: report(f * ANALYSIS_SHARE), cancelled)
    schedule = build_schedule(estimates)
    if not schedule:
        raise EchoNotFound("No echo was found in this recording, so nothing was changed.")
    _clean(input_path, output_path, schedule, channels, total,
           lambda f: report(ANALYSIS_SHARE + f * (1 - ANALYSIS_SHARE)), cancelled)
    report(1.0)
    used = [t.filter.delay_ms for t in schedule if t.filter is not None]
    return EchoResult(output=output_path, delay_ms=float(np.median(used)),
                      windows_total=len(schedule), windows_cleaned=len(used))


def _measure(path: Path, total: int, report: Progress,
             cancelled: Cancelled) -> list[WindowEstimate]:
    """Pass 1: the echo filter of every WINDOW_S window (mono mix)."""
    size = int(WINDOW_S * PROCESS_SR)
    estimates: list[WindowEstimate] = []
    pending: list[np.ndarray] = []
    start = done = 0
    with _decoder(path, channels=1) as stream:
        for chunk in _chunks(stream, channels=1, cancelled=cancelled):
            pending.append(chunk[0])
            done += chunk.shape[1]
            if done - start >= size:
                window = np.concatenate(pending)
                estimates.append(_estimate(start, window[:size]))
                pending, start = [window[size:]], start + size
            report(min(1.0, done / total))
    tail = np.concatenate(pending) if pending else np.zeros(0)
    if len(tail):
        long_enough = len(tail) >= MIN_WINDOW_S * PROCESS_SR
        estimates.append(_estimate(start, tail) if long_enough
                         else WindowEstimate(start, len(tail), None))
    return estimates


def _estimate(start: int, window: np.ndarray) -> WindowEstimate:
    return WindowEstimate(start, len(window), estimate_filter(window, PROCESS_SR))


def _clean(path: Path, output: Path, schedule: tuple[TimedFilter, ...], channels: int,
           total: int, report: Progress, cancelled: Cancelled) -> None:
    """Pass 2: run the inverse echo filter and encode to `output`.

    Encodes to a sibling temporary file that replaces `output` only on
    success, so a failed or cancelled run never touches an existing file.
    """
    canceller = EchoCanceller(schedule, channels)
    partial = output.with_name(f"{output.stem}.part{output.suffix}")
    try:
        with _decoder(path, channels) as stream, _encoder(path, partial, channels) as sink:
            done = 0
            for chunk in _chunks(stream, channels, cancelled):
                sink.stdin.write(canceller.process(chunk).T.astype(np.float32).tobytes())
                done += chunk.shape[1]
                report(min(1.0, done / total))
        os.replace(partial, output)
    finally:
        _remove_quietly(partial)


def _remove_quietly(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:  # e.g. still locked on Windows; never hide the real error
        pass


@contextmanager
def _decoder(path: Path, channels: int) -> Iterator[subprocess.Popen]:
    """ffmpeg decoding the first audio track to float32 at PROCESS_SR."""
    cmd = [find_binary("ffmpeg"), "-nostdin", "-v", "error", "-i", str(path),
           "-map", "0:a:0", "-f", "f32le", "-ac", str(channels),
           "-ar", str(PROCESS_SR), "-"]
    with _process(cmd, "Could not read the audio", stdout=subprocess.PIPE) as proc:
        yield proc


@contextmanager
def _encoder(source: Path, output: Path, channels: int) -> Iterator[subprocess.Popen]:
    """ffmpeg encoding float32 from stdin; tags are copied from `source`."""
    cmd = [find_binary("ffmpeg"), "-nostdin", "-y", "-v", "error",
           "-f", "f32le", "-ar", str(PROCESS_SR), "-ac", str(channels), "-i", "pipe:0",
           "-i", str(source), "-map", "0:a", "-map_metadata", "1",
           *OUTPUT_CODECS[output.suffix.lower()], str(output)]
    with _process(cmd, "Could not save the cleaned audio", stdin=subprocess.PIPE) as proc:
        yield proc
        _close_quietly(proc.stdin)  # end of input; a failure shows in the exit code


@contextmanager
def _process(cmd: list[str], failure: str, **pipes) -> Iterator[subprocess.Popen]:
    """Run `cmd`; raise FFmpegError with its stderr if it fails.

    A broken pipe (OSError) means this ffmpeg died, so it is reported with
    ffmpeg's own message. Other exceptions (cancel, the other process
    failing) kill this process and propagate unchanged.
    """
    with tempfile.TemporaryFile() as err_file:
        proc = subprocess.Popen(cmd, stderr=err_file, creationflags=CREATE_NO_WINDOW,
                                **pipes)
        try:
            yield proc
        except OSError as exc:
            _stop(proc)
            raise FFmpegError(f"{failure}: {_tail(err_file)}") from exc
        except BaseException:
            _stop(proc)
            raise
        finally:
            _close_quietly(proc.stdout)
            _close_quietly(proc.stdin)
        if proc.wait() != 0:
            raise FFmpegError(f"{failure}: {_tail(err_file)}")


def _stop(proc: subprocess.Popen) -> None:
    proc.kill()
    proc.wait()


def _close_quietly(pipe: IO[bytes] | None) -> None:
    """Close a pipe; flushing into a process that already exited may fail."""
    if pipe is None or pipe.closed:
        return
    try:
        pipe.close()
    except OSError:
        pass


def _tail(err_file: IO[bytes]) -> str:
    err_file.seek(0)
    return err_file.read().decode(errors="replace").strip()[-800:]


def _chunks(proc: subprocess.Popen, channels: int,
            cancelled: Cancelled) -> Iterator[np.ndarray]:
    """(channels, n) float64 chunks read from a decoder's stdout."""
    frame_bytes = 4 * channels
    while True:
        if cancelled():
            raise RenderCancelled()
        data = proc.stdout.read(CHUNK_FRAMES * frame_bytes)
        usable = len(data) - len(data) % frame_bytes
        if usable == 0:
            return
        samples = np.frombuffer(data[:usable], np.float32).astype(np.float64)
        yield samples.reshape(-1, channels).T
