"""Transcript export: caption formatting, the pipeline with a fake model, CLI."""

from types import SimpleNamespace

import pytest

from silence_remover import cli, transcript
from silence_remover.ffmpeg_tools import FFmpegError
from silence_remover.transcript import (
    Caption,
    TranscriptCancelled,
    format_timestamp,
    render_captions,
    transcribe,
)

from .conftest import make_clip, requires_ffmpeg

CAPTIONS = (Caption(0.0, 1.5, " Hello there. "), Caption(61.25, 3725.007, "Bye."))


def test_format_timestamp_srt_and_vtt():
    assert format_timestamp(0) == "00:00:00,000"
    assert format_timestamp(3725.007) == "01:02:05,007"
    assert format_timestamp(1.9996) == "00:00:02,000"           # rounds, never 1000 ms
    assert format_timestamp(61.25, sep=".") == "00:01:01.250"
    assert format_timestamp(-1) == "00:00:00,000"


def test_render_srt():
    assert render_captions(CAPTIONS, ".srt") == (
        "1\n00:00:00,000 --> 00:00:01,500\nHello there.\n\n"
        "2\n00:01:01,250 --> 01:02:05,007\nBye.\n"
    )


def test_render_vtt_and_txt():
    vtt = render_captions(CAPTIONS, ".VTT")
    assert vtt.startswith("WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nHello there.\n")
    assert render_captions(CAPTIONS, ".txt") == "Hello there.\nBye.\n"


def test_render_rejects_unknown_format():
    with pytest.raises(ValueError, match="extensions"):
        render_captions(CAPTIONS, ".docx")


class FakeModel:
    """Stands in for faster_whisper.WhisperModel."""

    def __init__(self, segments):
        self.segments = segments
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        info = SimpleNamespace(language="en", language_probability=0.97, duration=4.0)
        return iter(self.segments), info


def _seg(start, end, text):
    return SimpleNamespace(start=start, end=end, text=text)


@pytest.fixture
def fake_model(monkeypatch):
    model = FakeModel([_seg(0.2, 1.8, " One."), _seg(2.0, 3.9, " Two.")])
    loaded = []
    monkeypatch.setattr(transcript, "load_model",
                        lambda size: loaded.append(size) or model)
    model.loaded = loaded
    return model


@pytest.fixture(scope="module")
def tone_clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("transcript") / "talk.mp4"
    make_clip(path, 4, [(1.0, 2.0)])
    return path


@requires_ffmpeg
def test_transcribe_writes_srt(tone_clip, tmp_path, fake_model):
    out = tmp_path / "talk.srt"
    progress = []
    result = transcribe(tone_clip, out, translate=True, model_size="small",
                        on_progress=progress.append)
    assert out.read_text(encoding="utf-8").startswith(
        "1\n00:00:00,200 --> 00:00:01,800\nOne.\n")
    assert result.captions == 2 and result.language == "en"
    assert "en" in result.summary and "2 captions" in result.summary
    audio, kwargs = fake_model.calls[0]
    assert audio.dtype.name == "float32" and audio.size == pytest.approx(4 * 16_000, rel=0.02)
    assert kwargs["task"] == "translate" and kwargs["vad_filter"] is True
    assert fake_model.loaded == ["small"]
    assert progress[-1] == 1.0 and progress == sorted(progress)


@requires_ffmpeg
def test_transcribe_original_language(tone_clip, tmp_path, fake_model):
    transcribe(tone_clip, tmp_path / "talk.txt")
    assert fake_model.calls[0][1]["task"] == "transcribe"
    assert (tmp_path / "talk.txt").read_text(encoding="utf-8") == "One.\nTwo.\n"


@requires_ffmpeg
def test_transcribe_cancel_leaves_no_file(tone_clip, tmp_path, fake_model):
    out = tmp_path / "talk.srt"
    with pytest.raises(TranscriptCancelled):
        transcribe(tone_clip, out, is_cancelled=lambda: True)
    assert not out.exists()


def test_transcribe_validates_before_work(tmp_path, fake_model):
    with pytest.raises(ValueError, match="extensions"):
        transcribe(tmp_path / "in.mp4", tmp_path / "out.docx")
    with pytest.raises(ValueError, match="model"):
        transcribe(tmp_path / "in.mp4", tmp_path / "out.srt", model_size="huge")
    assert not fake_model.calls


@requires_ffmpeg
def test_transcribe_needs_audio(tmp_path, fake_model):
    import subprocess

    silent = tmp_path / "picture.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "color=size=64x64:duration=1", str(silent)], check=True)
    with pytest.raises(FFmpegError, match="no audio"):
        transcribe(silent, tmp_path / "out.srt")


def test_missing_faster_whisper_is_reported(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_whisper(name, *args, **kwargs):
        if name == "faster_whisper":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_whisper)
    assert not transcript.transcript_available()
    with pytest.raises(transcript.TranscriptUnavailable, match="faster-whisper"):
        transcript.load_model("medium")


@requires_ffmpeg
def test_cli_transcript(tone_clip, tmp_path, fake_model, capsys):
    out = tmp_path / "talk.vtt"
    assert cli.main([str(tone_clip), str(out), "--transcript", "--translate",
                     "--whisper-model", "base"]) == 0
    assert out.read_text(encoding="utf-8").startswith("WEBVTT")
    assert fake_model.calls[0][1]["task"] == "translate"
    assert fake_model.loaded == ["base"]
    assert "Saved" in capsys.readouterr().out


def test_cli_transcript_bad_extension(tmp_path, fake_model):
    assert cli.main([str(tmp_path / "in.mp4"), str(tmp_path / "out.doc"),
                     "--transcript"]) == 1
