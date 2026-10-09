"""Remove a doubled-voice echo from real audio files through ffmpeg."""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from silence_remover import cli
from silence_remover.echo_removal import EchoNotFound, EchoResult, remove_echo
from silence_remover.ffmpeg_tools import FFmpegError, probe
from silence_remover.render import RenderCancelled

from .conftest import requires_ffmpeg

SR = 48_000
SPEECH = Path(__file__).parent / "fixtures" / "speech.flac"
DELAY_MS = 26.4
GAIN = 0.5
CANCEL_IN_PASS_2 = 30  # pass 1 checks once per second of the 24 s clip

pytestmark = requires_ffmpeg


def decode(path: Path, channels: int = 1) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le",
         "-ac", str(channels), "-ar", str(SR), "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, channels).T.astype(np.float64)


def encode(path: Path, signal: np.ndarray) -> Path:
    """Write (channels, n) float samples to `path` (format from the extension)."""
    frames = np.atleast_2d(signal)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR),
         "-ac", str(frames.shape[0]), "-i", "-", str(path)],
        input=frames.T.astype(np.float32).tobytes(), check=True)
    return path


def add_echo(clean: np.ndarray) -> np.ndarray:
    d = int(round(DELAY_MS / 1000 * SR))
    echo = np.zeros_like(clean)
    echo[d:] = clean[:-d]
    return clean + GAIN * echo


def residual_db(clean: np.ndarray, echoed: np.ndarray, cleaned: np.ndarray) -> float:
    n = min(len(clean), len(cleaned))
    left = cleaned[SR:n] - clean[SR:n]
    added = echoed[SR:n] - clean[SR:n]
    return 10 * np.log10(np.sum(left**2) / np.sum(added**2))


@pytest.fixture(scope="module")
def clean_speech() -> np.ndarray:
    """24 s of real speech (the 6 s fixture four times), mono at 48 kHz."""
    return 0.5 * np.tile(decode(SPEECH)[0], 4)


@pytest.fixture(scope="module")
def echoed_wav(tmp_path_factory, clean_speech) -> Path:
    return encode(tmp_path_factory.mktemp("echo") / "echoed.wav", add_echo(clean_speech))


# ---------- result ----------

def test_summary_text():
    result = EchoResult(output=Path("o.mp3"), delay_ms=26.43, windows_total=246,
                        windows_cleaned=112)
    assert result.summary == "Echo found at 26.4 ms; removed in 112 of 246 parts of the audio."


# ---------- real ffmpeg ----------

def test_echo_is_removed_from_real_speech(echoed_wav, clean_speech, tmp_path):
    progress = []
    result = remove_echo(echoed_wav, tmp_path / "clean.wav", on_progress=progress.append)
    assert result.delay_ms == pytest.approx(DELAY_MS, abs=0.3)
    assert result.windows_total == 3
    assert result.windows_cleaned >= 2
    cleaned = decode(result.output)[0]
    assert len(cleaned) == pytest.approx(len(clean_speech), abs=SR // 100)
    # One speaker's pitch repeats near the echo delay, which biases the blind
    # estimate; about -13 dB is the ceiling here (synthetic noise reaches -29).
    assert residual_db(clean_speech, add_echo(clean_speech), cleaned) < -10
    assert progress == sorted(progress)
    assert progress[-1] == 1.0


def test_stereo_mp3_in_and_out(clean_speech, tmp_path):
    echoed = add_echo(clean_speech)
    source = encode(tmp_path / "echoed.mp3", np.stack([echoed, 0.8 * echoed]))
    result = remove_echo(source, tmp_path / "clean.mp3")
    info = probe(result.output)
    assert info.has_audio and info.audio_channels == 2
    assert info.duration_s == pytest.approx(probe(source).duration_s, abs=0.1)


@pytest.mark.parametrize("suffix", [".m4a", ".flac"])
def test_other_output_formats(echoed_wav, tmp_path, suffix):
    result = remove_echo(echoed_wav, tmp_path / f"clean{suffix}")
    assert probe(result.output).has_audio


def test_audio_without_echo_is_rejected(clean_speech, tmp_path):
    source = encode(tmp_path / "clean.wav", clean_speech)
    out = tmp_path / "out.wav"
    with pytest.raises(EchoNotFound, match="No echo"):
        remove_echo(source, out)
    assert not out.exists()


def test_invalid_inputs(echoed_wav, tmp_path):
    with pytest.raises(FFmpegError, match="different"):
        remove_echo(echoed_wav, echoed_wav)
    with pytest.raises(ValueError, match="extension"):
        remove_echo(echoed_wav, tmp_path / "out.ogg")
    with pytest.raises(FFmpegError, match="not found"):
        remove_echo(tmp_path / "missing.mp3", tmp_path / "out.mp3")
    video_only = tmp_path / "noaudio.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=duration=1", str(video_only)], check=True)
    with pytest.raises(FFmpegError, match="no audio"):
        remove_echo(video_only, tmp_path / "out.mp3")


@pytest.mark.parametrize("cancel_after", [0, CANCEL_IN_PASS_2])
def test_cancel_removes_partial_output(echoed_wav, tmp_path, cancel_after):
    """Cancel during analysis (first check) or during cleaning (later)."""
    calls = []

    def is_cancelled() -> bool:
        calls.append(1)
        return len(calls) > cancel_after

    out = tmp_path / "c.mp3"
    with pytest.raises(RenderCancelled):
        remove_echo(echoed_wav, out, is_cancelled=is_cancelled)
    assert not out.exists()


# ---------- CLI ----------

def test_cli_remove_echo(echoed_wav, tmp_path, capsys):
    out = tmp_path / "cli.mp3"
    assert cli.main([str(echoed_wav), str(out), "--remove-echo"]) == 0
    assert out.exists()
    assert "Echo found at 26." in capsys.readouterr().out


def test_cli_reports_missing_echo(clean_speech, tmp_path, capsys):
    source = encode(tmp_path / "clean.wav", clean_speech)
    assert cli.main([str(source), str(tmp_path / "o.mp3"), "--remove-echo"]) == 1
    assert "No echo" in capsys.readouterr().err


def test_cli_rejects_remove_echo_with_other_jobs():
    with pytest.raises(SystemExit) as exc:
        cli.main(["in.mp3", "out.mp3", "--remove-echo", "--extract-audio"])
    assert exc.value.code == 2


@pytest.mark.parametrize("cancel_after", [0, CANCEL_IN_PASS_2])
def test_cancel_keeps_a_file_being_overwritten(echoed_wav, tmp_path, cancel_after):
    out = tmp_path / "keep.mp3"
    out.write_bytes(b"original")
    calls = []

    def is_cancelled() -> bool:
        calls.append(1)
        return len(calls) > cancel_after

    with pytest.raises(RenderCancelled):
        remove_echo(echoed_wav, out, is_cancelled=is_cancelled)
    assert out.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [out]          # no temporary file left


def test_encoder_failure_reports_ffmpeg_error(echoed_wav, tmp_path, monkeypatch):
    from silence_remover import echo_removal

    monkeypatch.setitem(echo_removal.OUTPUT_CODECS, ".mp3", ["-c:a", "no_such_codec"])
    out = tmp_path / "x.mp3"
    out.write_bytes(b"original")
    with pytest.raises(FFmpegError, match="Could not save the cleaned audio: .*no_such_codec"):
        remove_echo(echoed_wav, out)
    assert out.read_bytes() == b"original"
