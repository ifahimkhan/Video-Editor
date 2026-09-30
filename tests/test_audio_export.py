"""MP3 audio extraction."""

import subprocess

import pytest

from silence_remover import cli
from silence_remover.audio_export import extract_mp3
from silence_remover.ffmpeg_tools import FFmpegError, probe
from silence_remover.render import RenderCancelled

from .conftest import make_clip, requires_ffmpeg

pytestmark = requires_ffmpeg


def audio_stream(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_name,bit_rate", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    codec, bitrate = out.stdout.strip().split(",")
    return codec, int(bitrate)


def test_extracts_full_length_mp3(speech_clip, tmp_path):
    progress = []
    out = extract_mp3(speech_clip, tmp_path / "a.mp3", on_progress=progress.append)
    info = probe(out)
    assert not info.has_video and info.has_audio
    # Full length: silence is NOT removed.
    assert info.duration_s == pytest.approx(10.0, abs=0.1)
    codec, bitrate = audio_stream(out)
    assert codec == "mp3"
    assert bitrate == pytest.approx(192_000, rel=0.05)
    assert progress[-1] == 1.0


def test_custom_bitrate(speech_clip, tmp_path):
    out = extract_mp3(speech_clip, tmp_path / "b.mp3", bitrate_kbps=128)
    assert audio_stream(out)[1] == pytest.approx(128_000, rel=0.05)


def test_rejects_bad_arguments(speech_clip, tmp_path):
    with pytest.raises(ValueError, match="bitrate"):
        extract_mp3(speech_clip, tmp_path / "c.mp3", bitrate_kbps=100)
    with pytest.raises(ValueError, match=".mp3"):
        extract_mp3(speech_clip, tmp_path / "c.wav")


def test_video_without_audio_is_reported(tmp_path):
    silent = tmp_path / "noaudio.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc2=duration=1", str(silent)], check=True)
    with pytest.raises(FFmpegError, match="no audio"):
        extract_mp3(silent, tmp_path / "x.mp3")


def test_cancel_removes_partial_file(speech_clip, tmp_path):
    out = tmp_path / "cancel.mp3"
    with pytest.raises(RenderCancelled):
        extract_mp3(speech_clip, out, is_cancelled=lambda: True)
    assert not out.exists()


def test_audio_only_input(tmp_path):
    clip = tmp_path / "voice.m4a"
    make_clip(clip, 3, [], video=False)
    assert probe(extract_mp3(clip, tmp_path / "v.mp3")).duration_s == pytest.approx(3, abs=0.1)


def test_cli_extract_audio(speech_clip, tmp_path):
    out = tmp_path / "cli.mp3"
    assert cli.main([str(speech_clip), str(out), "--extract-audio",
                     "--mp3-bitrate", "320"]) == 0
    assert audio_stream(out)[1] == pytest.approx(320_000, rel=0.05)
    assert cli.main([str(speech_clip), str(tmp_path / "bad.wav"), "--extract-audio"]) == 1
