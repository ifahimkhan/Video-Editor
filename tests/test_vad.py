"""Silero VAD: streaming model wrapper and voice-mode detection."""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from silence_remover import cli
from silence_remover.analysis import analyze
from silence_remover.segments import (
    DetectionMode,
    DetectionSettings,
    Segment,
    detect,
    voice_mask,
)
from silence_remover.vad import (
    MODEL_ENV_VAR,
    WINDOW_SAMPLES,
    SileroVad,
    VadUnavailable,
    load_vad,
    vad_available,
)

from .conftest import make_profile, requires_ffmpeg

SPEECH = Path(__file__).parent / "fixtures" / "speech.flac"
requires_vad = pytest.mark.skipif(not vad_available(), reason="onnxruntime/model missing")


@pytest.fixture(scope="session")
def speech_noise_speech(tmp_path_factory):
    """speech (~6 s) | 2 s silence | 3 s loud brown noise | speech (~6 s)."""
    path = tmp_path_factory.mktemp("vad") / "mix.wav"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-i", str(SPEECH),
        "-f", "lavfi", "-i",
        "anoisesrc=color=brown:amplitude=0.3:duration=3:sample_rate=16000",
        "-filter_complex",
        "[0:a]asplit[x][y];[x]apad=pad_dur=2[s1];[s1][1:a][y]concat=n=3:v=0:a=1[a]",
        "-map", "[a]", str(path),
    ], check=True)
    return path


# ---------- pure logic ----------

def test_voice_mask_hysteresis_holds_speech_through_dips():
    prob = np.array([0.1, 0.6, 0.4, 0.4, 0.2, 0.6, 0.1], dtype=np.float32)
    # Enters at >=0.5, stays while >=0.35, leaves at 0.2.
    assert voice_mask(prob, 0.5, 1).tolist() == [0, 1, 1, 1, 0, 1, 0]


def test_voice_mask_drops_short_blips():
    prob = np.array([0.0, 0.9, 0.0, 0.9, 0.9, 0.9, 0.0], dtype=np.float32)
    assert voice_mask(prob, 0.5, 3).tolist() == [0, 0, 0, 1, 1, 1, 0]


def test_voice_mode_uses_probabilities_not_loudness():
    # Loud for the whole second, but only the middle is speech.
    base = make_profile([(960, -10.0)])
    prob = np.array([0.0] * 10 + [0.9] * 10 + [0.0] * 10, dtype=np.float32)
    profile = type(base)(db=base.db.copy(), frame_ms=base.frame_ms,
                         duration_ms=base.duration_ms, speech_prob=prob)
    loud = detect(profile, DetectionSettings(min_silence_ms=100, padding_ms=0))
    voice = detect(profile, DetectionSettings(min_silence_ms=100, padding_ms=0,
                                              mode="voice"))
    assert loud.keep == (Segment(0, 960),)
    assert voice.keep == (Segment(320, 640),)


def test_voice_mode_without_probabilities_raises():
    with pytest.raises(ValueError, match="Voice detection"):
        detect(make_profile([(1000, -10.0)]), DetectionSettings(mode=DetectionMode.VOICE))


def test_invalid_mode_and_vad_threshold_rejected():
    with pytest.raises(ValueError):
        DetectionSettings(mode="music")
    with pytest.raises(ValueError):
        DetectionSettings(vad_threshold=1.5)


def test_missing_model_disables_vad(monkeypatch, tmp_path):
    monkeypatch.setenv(MODEL_ENV_VAR, str(tmp_path / "nope.onnx"))
    assert not vad_available()
    assert load_vad() is None
    with pytest.raises(VadUnavailable):
        SileroVad()


# ---------- model ----------

@requires_vad
def test_streaming_matches_single_pass():
    """Carrying state across chunks gives the same result as one big call."""
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(WINDOW_SAMPLES * 40) * 0.1).astype(np.float32)
    whole = SileroVad().process(audio)
    vad = SileroVad()
    pieces = np.concatenate([vad.process(c) for c in np.split(audio, 8)])
    assert whole.shape == (40,)
    np.testing.assert_allclose(pieces, whole, atol=1e-4)


@requires_vad
def test_rejects_partial_window():
    with pytest.raises(ValueError):
        SileroVad().process(np.zeros(100, dtype=np.float32))


@requires_vad
@requires_ffmpeg
def test_voice_mode_cuts_loud_noise_that_loudness_mode_keeps(speech_noise_speech):
    profile = analyze(speech_noise_speech)
    assert profile.has_vad
    noise = Segment(8000, 11000)

    def overlap(result):
        return sum(max(0, min(s.end_ms, noise.end_ms) - max(s.start_ms, noise.start_ms))
                   for s in result.keep)

    loud = detect(profile, DetectionSettings(padding_ms=100))
    voice = detect(profile, DetectionSettings(padding_ms=100, mode="voice"))
    assert overlap(loud) > 2500            # noise is loud, so it's kept
    assert overlap(voice) < 300            # but it isn't speech
    assert voice.kept_ms > 8000            # both speech parts survive


@requires_ffmpeg
def test_analyze_without_vad(speech_clip):
    assert analyze(speech_clip, use_vad=False).speech_prob is None


@requires_vad
@requires_ffmpeg
def test_cli_voice_mode(speech_noise_speech, tmp_path, capsys):
    out = tmp_path / "voice.wav"
    assert cli.main([str(speech_noise_speech), str(out), "--mode", "voice"]) == 0
    assert out.exists()
    assert "Keeping" in capsys.readouterr().out
