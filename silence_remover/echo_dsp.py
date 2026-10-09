"""Find and cancel a doubled-voice echo (pure numpy).

When two microphones record the same voice and get mixed, the result is the
voice plus a delayed, filtered copy of it: x = s + h * s(t - D). The copy
shows up as a peak in the cepstrum at the delay D. Once the echo filter h is
measured (least squares around D), the inverse filter

    y[n] = x[n] - sum_k h[k] * y[n - lag0 - k]

recovers s. The filter only feeds back samples at least `lag0` old, so the
signal is processed in blocks of `lag0` samples with FFT convolution. Two
devices drift apart slowly, so h is measured per window and blended over time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

# The echo filter is measured once per window. Shorter windows follow a
# drifting delay better but hold less speech to measure from.
WINDOW_S = 10.0
MIN_DELAY_MS = 10.0
MAX_DELAY_MS = 200.0
# Cepstral peak height that counts as an echo. Echo-free speech peaks around
# 0.02-0.035; the doubled voice in real recordings reaches 0.07-0.15.
DETECT_THRESHOLD = 0.05
# Echo filter spans this much either side of the delay (the second mic's
# "room"). Wider filters start fitting the speech itself.
TAIL_MS = 1.5
# A window whose delay is further than this from the median of its
# neighbours locked onto something else (often the voice's own pitch) and is
# treated as no echo. Local, so a slowly drifting delay is still followed.
MAX_JUMP_MS = 0.5
NEIGHBOURS = 4  # detections on each side for that median
# Above gain 1 at any frequency the inverse filter can blow up; cap below it.
MAX_ECHO_GAIN = 0.95
GAIN_GRID = 1 << 14
GAIN_LIMIT_ROUNDS = 20
LPC_ORDER = 32
LPC_NOISE_FLOOR = 1e-3
DETECT_BAND_HZ = 8_000.0
ESTIMATE_ITERATIONS = 3
RIDGE = 1e-3
MIN_RMS = 1e-4


@dataclass(frozen=True)
class EchoFilter:
    """Echo impulse response: taps[k] is the echo gain at lag `lag0 + k` samples."""
    lag0: int
    taps: np.ndarray
    delay_ms: float
    strength: float

    def peak_gain(self) -> float:
        return float(np.abs(np.fft.rfft(self.taps, GAIN_GRID)).max())


@dataclass(frozen=True)
class WindowEstimate:
    start: int
    length: int
    filter: EchoFilter | None


@dataclass(frozen=True)
class TimedFilter:
    """The filter to use at sample `center`; filters are blended in between."""
    center: int
    filter: EchoFilter | None


def _fft_size(n: int) -> int:
    return 1 << int(np.ceil(np.log2(max(2, n))))


def find_echo(x: np.ndarray, sr: int) -> tuple[float, float]:
    """(delay_ms, strength) of the strongest echo between MIN and MAX delay.

    Uses the cepstrum of the band below DETECT_BAND_HZ, where speech lives;
    strength is the cepstral peak height (see DETECT_THRESHOLD).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or np.sqrt(np.mean(x**2)) < MIN_RMS:
        return 0.0, 0.0
    n = _fft_size(2 * len(x))
    spectrum = np.fft.rfft(x - x.mean(), n)
    bins = int(n * min(DETECT_BAND_HZ, sr / 2) / sr) + 1
    cepstrum = np.fft.irfft(np.log(np.abs(spectrum[:bins]) + 1e-9))
    band_sr = sr * (len(cepstrum) / n)
    lo = int(MIN_DELAY_MS / 1000 * band_sr)
    hi = min(int(MAX_DELAY_MS / 1000 * band_sr), len(cepstrum) // 2)
    if hi <= lo:
        return 0.0, 0.0
    peak = lo + int(np.argmax(cepstrum[lo:hi]))
    return peak / band_sr * 1000, float(cepstrum[peak])


def estimate_filter(x: np.ndarray, sr: int) -> EchoFilter | None:
    """Measure the echo filter of one window, or None when there is no echo."""
    x = np.asarray(x, dtype=np.float64)
    delay_ms, strength = find_echo(x, sr)
    if strength < DETECT_THRESHOLD:
        return None
    half = int(round(TAIL_MS / 1000 * sr))
    lag0 = int(round(delay_ms / 1000 * sr)) - half
    span = 2 * half + 1
    if lag0 < 1 or lag0 + span >= len(x):
        return None
    taps = np.zeros(span)
    estimate = x
    whiten = _whitener(x)
    target = whiten(x)
    for _ in range(ESTIMATE_ITERATIONS):
        taps = _limit_gain(_least_squares_taps(target, whiten(estimate), lag0, span))
        trial = EchoFilter(lag0, taps, delay_ms, strength)
        estimate = cancel_echo(x, (TimedFilter(0, trial),))
    return EchoFilter(lag0, taps, delay_ms, strength)


def _limit_gain(taps: np.ndarray) -> np.ndarray:
    """Cap the filter's gain at MAX_ECHO_GAIN only at the frequencies that exceed it.

    Where the audio has no content (above a lossy codec's cut-off, or above
    8 kHz in upsampled speech) least squares leaves the gain unconstrained,
    often near 1. Scaling the whole filter down would weaken the cancellation
    where the voice is, so the excess is clipped per frequency and the taps
    are re-truncated, a few rounds until it settles; a last uniform scale
    covers what remains.
    """
    target = 0.98 * MAX_ECHO_GAIN
    for _ in range(GAIN_LIMIT_ROUNDS):
        spectrum = np.fft.rfft(taps, GAIN_GRID)
        mag = np.abs(spectrum)
        if mag.max() <= MAX_ECHO_GAIN:
            return taps
        spectrum *= np.minimum(1.0, target / np.maximum(mag, 1e-12))
        taps = np.fft.irfft(spectrum, GAIN_GRID)[:len(taps)]
    gain = float(np.abs(np.fft.rfft(taps, GAIN_GRID)).max())
    return taps * (MAX_ECHO_GAIN / gain) if gain > MAX_ECHO_GAIN else taps


def _whitener(x: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """LPC inverse filter that flattens the spectrum of `x`.

    The echo model x = s + h * s(t - D) still holds after any fixed filter,
    but on flat-spectrum signals least squares is well conditioned; on raw
    speech (strong lows, weak highs) it turns the voice's own pitch
    correlation into large errors in h.
    """
    n = _fft_size(2 * len(x))
    auto = np.fft.irfft(np.abs(np.fft.rfft(x, n)) ** 2, n)[:LPC_ORDER + 1]
    auto[0] *= 1 + LPC_NOISE_FLOOR  # keeps empty bands from being boosted wildly
    idx = np.arange(LPC_ORDER)
    coeffs = np.linalg.solve(auto[np.abs(idx[:, None] - idx[None, :])], auto[1:])
    kernel = np.concatenate([[1.0], -coeffs])
    return lambda signal: np.convolve(signal, kernel)[:len(signal)]


def _least_squares_taps(x: np.ndarray, y: np.ndarray, lag0: int, span: int) -> np.ndarray:
    """Taps h minimising |x[n] - sum_k h[k] y[n - lag0 - k]|^2 (normal equations)."""
    n = _fft_size(2 * len(x))
    Y = np.fft.rfft(y, n)
    auto = np.fft.irfft(np.abs(Y) ** 2, n)[:span]
    cross = np.fft.irfft(np.fft.rfft(x, n) * np.conj(Y), n)[lag0:lag0 + span]
    idx = np.arange(span)
    gram = auto[np.abs(idx[:, None] - idx[None, :])]
    gram[idx, idx] += RIDGE * auto[0]
    return np.linalg.solve(gram, cross)


def build_schedule(estimates: Sequence[WindowEstimate]) -> tuple[TimedFilter, ...]:
    """One TimedFilter per window; () when no window has an echo.

    Windows whose delay strays more than MAX_JUMP_MS from the median of the
    neighbouring detections are dropped as false detections.
    """
    found = [i for i, e in enumerate(estimates) if e.filter is not None]
    delays = np.array([estimates[i].filter.delay_ms for i in found])
    consistent = set()
    for k, i in enumerate(found):
        local = np.median(delays[max(0, k - NEIGHBOURS):k + NEIGHBOURS + 1])
        if abs(delays[k] - local) <= MAX_JUMP_MS:
            consistent.add(i)
    if not consistent:
        return ()
    return tuple(
        TimedFilter(center=e.start + e.length // 2,
                    filter=e.filter if i in consistent else None)
        for i, e in enumerate(estimates)
    )


class EchoCanceller:
    """Streaming inverse echo filter; feed consecutive chunks to `process`.

    Between window centers the two neighbouring filters are blended linearly.
    """

    def __init__(self, schedule: Sequence[TimedFilter], channels: int) -> None:
        active = [t.filter for t in schedule if t.filter is not None]
        self._channels = channels
        self._position = 0
        if not active:
            self._centers = None
            return
        self._lag_lo = min(f.lag0 for f in active)
        lag_hi = max(f.lag0 + len(f.taps) for f in active) - 1
        span = lag_hi - self._lag_lo + 1
        self._span = span
        self._lag_hi = lag_hi
        self._block = self._lag_lo
        self._nfft = _fft_size(self._block + 2 * span)
        self._centers = np.array([t.center for t in schedule], dtype=np.int64)
        self._spectra = np.stack([self._spectrum(t.filter) for t in schedule])
        self._history = np.zeros((channels, lag_hi))

    def _spectrum(self, echo: EchoFilter | None) -> np.ndarray:
        dense = np.zeros(self._span)
        if echo is not None:
            offset = echo.lag0 - self._lag_lo
            dense[offset:offset + len(echo.taps)] = echo.taps
        return np.fft.rfft(dense, self._nfft)

    def _spectrum_at(self, sample: int) -> np.ndarray:
        """Echo filter spectrum at `sample`, blended between window centers."""
        i = int(np.searchsorted(self._centers, sample))
        if i == 0:
            return self._spectra[0]
        if i == len(self._centers):
            return self._spectra[-1]
        c0, c1 = self._centers[i - 1], self._centers[i]
        a = (sample - c0) / (c1 - c0)
        return (1 - a) * self._spectra[i - 1] + a * self._spectra[i]

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """Clean a (channels, n) chunk; returns a new (channels, n) array."""
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[0] != self._channels:
            raise ValueError(f"Expected a ({self._channels}, n) chunk")
        n = chunk.shape[1]
        if self._centers is None:
            self._position += n
            return chunk.copy()
        hist = self._lag_hi
        buffer = np.concatenate([self._history, np.zeros_like(chunk)], axis=1)
        for start in range(0, n, self._block):
            stop = min(start + self._block, n)
            p = hist + start
            past = buffer[:, p - self._lag_hi:p + (stop - start) - self._lag_lo]
            full = np.fft.irfft(np.fft.rfft(past, self._nfft, axis=1)
                                * self._spectrum_at(self._position + start),
                                self._nfft, axis=1)
            feedback = full[:, self._span - 1:self._span - 1 + stop - start]
            buffer[:, p:p + stop - start] = chunk[:, start:stop] - feedback
        self._history = buffer[:, -hist:].copy()
        self._position += n
        return buffer[:, hist:].copy()


def cancel_echo(x: np.ndarray, schedule: Sequence[TimedFilter]) -> np.ndarray:
    """Clean a whole signal, shape (n,) or (channels, n)."""
    x = np.asarray(x, dtype=np.float64)
    mono = x.ndim == 1
    frames = x[None, :] if mono else x
    out = EchoCanceller(schedule, channels=frames.shape[0]).process(frames)
    return out[0] if mono else out
