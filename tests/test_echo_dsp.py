"""Echo detection and cancellation on synthetic signals (pure numpy, no ffmpeg)."""

import numpy as np
import pytest

from silence_remover.echo_dsp import (
    DETECT_THRESHOLD,
    MAX_ECHO_GAIN,
    WINDOW_S,
    EchoCanceller,
    EchoFilter,
    TimedFilter,
    WindowEstimate,
    build_schedule,
    cancel_echo,
    estimate_filter,
    find_echo,
)

SR = 48_000


def speechlike(seconds: float, seed: int = 0) -> np.ndarray:
    """Coloured noise gated into 'syllables' and pauses, like speech at a glance."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1 / SR)
    noise = np.fft.irfft(spectrum / (1 + freqs / 600), n)
    gate_len = int(0.15 * SR)
    gates = rng.random(n // gate_len + 1) < 0.65
    envelope = np.repeat(gates, gate_len)[:n].astype(float)
    envelope = np.convolve(envelope, np.hanning(960) / 480, mode="same")
    signal = noise * envelope
    return 0.1 * signal / np.abs(signal).max()


def delayed(signal: np.ndarray, delay_ms: float) -> np.ndarray:
    d = int(round(delay_ms / 1000 * SR))
    out = np.zeros_like(signal)
    out[d:] = signal[:-d]
    return out


def with_echo(clean: np.ndarray, delay_ms: float, gain: float) -> np.ndarray:
    """A second microphone: a delayed, slightly smeared copy of the voice."""
    return (clean + gain * delayed(clean, delay_ms)
            + 0.3 * gain * delayed(clean, delay_ms + 0.8))


def residual_db(clean: np.ndarray, echoed: np.ndarray, cleaned: np.ndarray) -> float:
    """Echo left after cleaning, relative to the echo that was there (dB)."""
    skip = SR  # let the first estimate settle
    left = cleaned[skip:] - clean[skip:]
    added = echoed[skip:] - clean[skip:]
    return 10 * np.log10(np.sum(left**2) / np.sum(added**2))


def windows(signal: np.ndarray, window_s: float = WINDOW_S) -> list[WindowEstimate]:
    size = int(window_s * SR)
    return [WindowEstimate(start=s, length=len(signal[s:s + size]),
                           filter=estimate_filter(signal[s:s + size], SR))
            for s in range(0, len(signal), size)]


# ---------- detection ----------

@pytest.mark.parametrize("delay_ms", [15.0, 26.4, 120.0])
def test_find_echo_measures_delay(delay_ms):
    delay, strength = find_echo(with_echo(speechlike(20), delay_ms, 0.5), SR)
    assert delay == pytest.approx(delay_ms, abs=0.2)
    assert strength >= DETECT_THRESHOLD


def test_clean_signal_has_no_echo():
    clean = speechlike(20, seed=3)
    _, strength = find_echo(clean, SR)
    assert strength < DETECT_THRESHOLD
    assert estimate_filter(clean, SR) is None


def test_digital_silence_has_no_echo():
    assert find_echo(np.zeros(SR * 5), SR) == (0.0, 0.0)
    assert estimate_filter(np.zeros(SR * 5), SR) is None


def test_filter_peaks_at_echo_delay():
    echo = estimate_filter(with_echo(speechlike(20), 26.4, 0.5), SR)
    assert echo is not None
    peak_lag = echo.lag0 + int(np.argmax(echo.taps))
    assert peak_lag / SR * 1000 == pytest.approx(26.4, abs=0.1)
    assert echo.taps.max() == pytest.approx(0.5, abs=0.05)


# ---------- cancellation ----------

def test_constant_echo_is_removed():
    clean = speechlike(30)
    echoed = with_echo(clean, 26.4, 0.6)
    echo = estimate_filter(echoed, SR)
    cleaned = cancel_echo(echoed, (TimedFilter(center=0, filter=echo),))
    assert residual_db(clean, echoed, cleaned) < -20


def test_drifting_echo_is_tracked():
    """Two devices on different clocks: the delay creeps 26.0 -> 26.25 ms in
    80 s (3 ppm; the recording this was built for drifts under 1 ppm)."""
    clean = speechlike(80, seed=1)
    n = np.arange(len(clean))
    delay_samples = (26.0 + 0.25 * n / len(n)) / 1000 * SR
    echoed = clean + 0.5 * np.interp(n - delay_samples, n, clean, left=0.0)
    cleaned = cancel_echo(echoed, build_schedule(windows(echoed)))
    assert residual_db(clean, echoed, cleaned) < -15


def test_echo_louder_than_safe_is_capped_and_stable():
    clean = speechlike(20, seed=2)
    echoed = with_echo(clean, 30.0, 0.95)
    echo = estimate_filter(echoed, SR)
    assert echo is not None
    assert echo.peak_gain() <= MAX_ECHO_GAIN + 1e-9
    cleaned = cancel_echo(echoed, (TimedFilter(center=0, filter=echo),))
    assert np.all(np.isfinite(cleaned))
    assert np.abs(cleaned).max() < 2 * np.abs(echoed).max()


def test_stereo_channels_are_cleaned_alike():
    clean = speechlike(20)
    echoed = with_echo(clean, 26.4, 0.5)
    stereo = np.stack([echoed, 0.5 * echoed])
    schedule = (TimedFilter(center=0, filter=estimate_filter(echoed, SR)),)
    cleaned = cancel_echo(stereo, schedule)
    assert cleaned.shape == stereo.shape
    np.testing.assert_allclose(cleaned[1], 0.5 * cleaned[0], atol=1e-9)


def test_chunk_size_does_not_change_output():
    echoed = with_echo(speechlike(10), 26.4, 0.5)
    schedule = (TimedFilter(center=0, filter=estimate_filter(echoed, SR)),)
    whole = cancel_echo(echoed, schedule)
    canceller = EchoCanceller(schedule, channels=1)
    parts = [canceller.process(echoed[None, s:s + 7_777])
             for s in range(0, len(echoed), 7_777)]
    np.testing.assert_allclose(np.concatenate(parts, axis=1)[0], whole, atol=1e-9)


def test_no_filters_passes_audio_through():
    signal = speechlike(3)
    out = cancel_echo(signal, (TimedFilter(center=0, filter=None),))
    np.testing.assert_array_equal(out, signal)
    np.testing.assert_array_equal(cancel_echo(signal, ()), signal)


# ---------- schedule ----------

def fake_filter(delay_ms: float) -> EchoFilter:
    return EchoFilter(lag0=int(delay_ms * SR / 1000), taps=np.array([0.5]),
                      delay_ms=delay_ms, strength=0.1)


def schedule_for(delays: list[float | None]) -> tuple[TimedFilter, ...]:
    return build_schedule([
        WindowEstimate(start=i * 100, length=100,
                       filter=None if d is None else fake_filter(d))
        for i, d in enumerate(delays)])


def test_schedule_drops_windows_far_from_their_neighbours():
    """28.7 ms is the kind of stray a voice's pitch produces next to a 26.4 ms echo."""
    schedule = schedule_for([26.4, 26.4, 90.0, 26.5, 28.7, 26.4, None])
    assert [t.center for t in schedule] == [50, 150, 250, 350, 450, 550, 650]
    assert [t.filter is not None for t in schedule] == [
        True, True, False, True, False, True, False]


def test_schedule_follows_a_slowly_drifting_delay():
    delays = list(np.linspace(26.0, 29.0, 40))
    assert all(t.filter is not None for t in schedule_for(delays))


def test_schedule_without_any_echo_is_empty():
    assert build_schedule([WindowEstimate(start=0, length=10, filter=None)]) == ()
