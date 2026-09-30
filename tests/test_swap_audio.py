"""Swap a video's soundtrack for another audio file."""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from silence_remover import cli
from silence_remover.analysis import analyze
from silence_remover.ffmpeg_tools import FFmpegError, probe
from silence_remover.render import RenderCancelled
from silence_remover.swap_audio import (
    SwapResult,
    audio_codec_for,
    audio_filter,
    swap_audio,
)

from .conftest import make_clip, requires_ffmpeg

LOUD_DB = -30.0     # lavfi sine plays at 1/8 amplitude: about -21 dBFS RMS
QUIET_DB = -60.0


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *map(str, args)], check=True)


def tone(path: Path, seconds: float) -> Path:
    ffmpeg("-f", "lavfi", "-i", f"sine=frequency=880:duration={seconds}", path)
    return path


def silence(path: Path, seconds: float) -> Path:
    ffmpeg("-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-t", seconds, path)
    return path


def video_md5(path: Path) -> str:
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v",
                          "-c", "copy", "-f", "md5", "-"],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def level_between(profile, start_s: float, end_s: float) -> float:
    frames = profile.db[int(start_s * 100):int(end_s * 100)]
    return float(np.median(frames))


# ---------- pure logic ----------

@pytest.mark.parametrize("offset, expected", [
    (0, "apad"),
    (250, "adelay=250:all=1,apad"),
    (-500, "atrim=start=0.500,asetpts=PTS-STARTPTS,apad"),
])
def test_audio_filter(offset, expected):
    assert audio_filter(offset) == expected


def test_audio_filter_rejects_huge_offset():
    with pytest.raises(ValueError):
        audio_filter(120_000)


def test_audio_codec_matches_container():
    assert audio_codec_for(Path("x.webm"))[1] == "libopus"
    assert audio_codec_for(Path("x.mp4"))[1] == "aac"
    assert audio_codec_for(Path("x.MKV"))[1] == "aac"


@pytest.mark.parametrize("audio_s, warning", [
    (10.2, None),
    (7.0, "3.0s shorter"),
    (13.5, "3.5s longer"),
])
def test_length_warning(audio_s, warning):
    result = SwapResult(output=Path("o.mp4"), video_s=10.0, audio_s=audio_s)
    if warning is None:
        assert result.length_warning is None
    else:
        assert warning in result.length_warning


# ---------- real ffmpeg ----------

@pytest.fixture(scope="module")
def tone_video(tmp_path_factory):
    """10 s video whose original audio is a loud tone throughout."""
    path = tmp_path_factory.mktemp("swap") / "video.mp4"
    make_clip(path, 10, [])
    return path


@requires_ffmpeg
def test_audio_is_replaced_and_video_untouched(tone_video, tmp_path):
    progress = []
    result = swap_audio(tone_video, silence(tmp_path / "quiet.wav", 10),
                        tmp_path / "out.mp4", on_progress=progress.append)
    out_profile = analyze(result.output, use_vad=False)
    assert level_between(out_profile, 0.5, 9.5) < QUIET_DB      # the new, silent audio
    assert level_between(analyze(tone_video, use_vad=False), 0.5, 9.5) > LOUD_DB
    assert video_md5(result.output) == video_md5(tone_video)    # picture bit-identical
    assert probe(result.output).duration_s == pytest.approx(10.0, abs=0.1)
    assert result.length_warning is None
    assert progress[-1] == 1.0


@requires_ffmpeg
def test_shorter_audio_is_padded_to_video_length(tone_video, tmp_path):
    result = swap_audio(tone_video, tone(tmp_path / "short.mp3", 5), tmp_path / "out.mp4")
    profile = analyze(result.output, use_vad=False)
    assert probe(result.output).duration_s == pytest.approx(10.0, abs=0.1)
    assert level_between(profile, 0.5, 4.5) > LOUD_DB
    assert level_between(profile, 5.5, 9.5) < QUIET_DB           # padded with silence
    assert "shorter" in result.length_warning


@requires_ffmpeg
def test_longer_audio_is_trimmed(tone_video, tmp_path):
    result = swap_audio(tone_video, tone(tmp_path / "long.wav", 14), tmp_path / "out.mp4")
    assert probe(result.output).duration_s == pytest.approx(10.0, abs=0.1)
    assert "longer" in result.length_warning


@requires_ffmpeg
def test_offset_delays_and_advances_audio(tone_video, tmp_path):
    clean = tone(tmp_path / "clean.wav", 10)
    later = analyze(swap_audio(tone_video, clean, tmp_path / "later.mp4",
                               offset_ms=1000).output, use_vad=False)
    assert level_between(later, 0.1, 0.9) < QUIET_DB             # 1 s of silence first
    assert level_between(later, 1.2, 9.5) > LOUD_DB

    earlier = swap_audio(tone_video, clean, tmp_path / "earlier.mp4", offset_ms=-2000)
    profile = analyze(earlier.output, use_vad=False)
    assert level_between(profile, 0.5, 7.5) > LOUD_DB
    assert level_between(profile, 8.2, 9.8) < QUIET_DB           # ran out 2 s early


@requires_ffmpeg
def test_mkv_output(tone_video, tmp_path):
    result = swap_audio(tone_video, tone(tmp_path / "a.wav", 10), tmp_path / "out.mkv")
    info = probe(result.output)
    assert info.has_video and info.has_audio


@requires_ffmpeg
def test_invalid_inputs(tone_video, tmp_path):
    video_only = tmp_path / "noaudio.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=duration=1", video_only)
    audio = tone(tmp_path / "a.wav", 2)
    with pytest.raises(FFmpegError, match="audio file has no audio"):
        swap_audio(tone_video, video_only, tmp_path / "x.mp4")
    with pytest.raises(FFmpegError, match="no video"):
        swap_audio(audio, audio, tmp_path / "x.mp4")
    with pytest.raises(FFmpegError, match="different"):
        swap_audio(tone_video, audio, tone_video)
    with pytest.raises(FFmpegError, match="not found"):
        swap_audio(tone_video, tmp_path / "missing.wav", tmp_path / "x.mp4")


@requires_ffmpeg
def test_cancel(tone_video, tmp_path):
    out = tmp_path / "c.mp4"
    with pytest.raises(RenderCancelled):
        swap_audio(tone_video, tone(tmp_path / "a.wav", 10), out, is_cancelled=lambda: True)
    assert not out.exists()


@requires_ffmpeg
def test_cli_swap_audio(tone_video, tmp_path, capsys):
    out = tmp_path / "cli.mp4"
    short = tone(tmp_path / "s.wav", 6)
    assert cli.main([str(tone_video), str(out), "--swap-audio", str(short),
                     "--audio-offset", "100"]) == 0
    assert out.exists()
    assert "Note: The new audio is" in capsys.readouterr().out
    assert cli.main([str(tone_video), str(tmp_path / "y.mp4"),
                     "--swap-audio", str(tmp_path / "missing.wav")]) == 1
