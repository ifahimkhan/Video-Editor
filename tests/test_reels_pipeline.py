"""End-to-end reels pipeline: transcript (fake Whisper) -> picks -> rendered clips."""

import json
from types import SimpleNamespace

import pytest

from silence_remover import cli, transcript
from silence_remover.ffmpeg_tools import probe
from silence_remover.reels.pipeline import create_reels, find_highlights
from silence_remover.reels.settings import Layout, ReelSettings

from .conftest import make_clip, requires_ffmpeg
from .test_reels_gemini import FakeClient, pick

TEXTS = ["Here's the trick nobody tells you.", "You cache the result.", "Then reuse it.",
         "That's it.", "and also something else", "Why is this slow?",
         "Because it runs twice.", "Fix it with one line.", "Done."]
SETTINGS = ReelSettings(layout=Layout.FIT_BARS, min_length_s=5, max_length_s=12,
                        clip_count=2)


class FakeWhisper:
    def __init__(self):
        self.calls = 0

    def transcribe(self, audio, **kwargs):
        self.calls += 1
        segments = [SimpleNamespace(start=i * 4.0 + 0.2, end=i * 4.0 + 3.6, text=t)
                    for i, t in enumerate(TEXTS)]
        return iter(segments), SimpleNamespace(language="en", language_probability=0.99)


@pytest.fixture
def whisper(monkeypatch):
    model = FakeWhisper()
    monkeypatch.setattr(transcript, "load_model", lambda size: model)
    return model


@pytest.fixture(scope="module")
def talk(tmp_path_factory):
    path = tmp_path_factory.mktemp("talk") / "talk.mp4"
    make_clip(path, 37, [])
    return path


@requires_ffmpeg
def test_create_reels_renders_clips_and_summary(talk, tmp_path, whisper):
    progress = []
    run = create_reels(talk, tmp_path / "out", SETTINGS, on_progress=progress.append)
    assert len(run.reels) == 2
    for reel in run.reels:
        assert reel.output.is_file()
        assert reel.output.name.startswith("talk_reel")
        length = reel.highlight.end_s - reel.highlight.start_s
        assert 5 <= length <= 12
        assert probe(reel.output).duration_s == pytest.approx(length, abs=0.2)
    summary = json.loads(run.summary_path.read_text(encoding="utf-8"))
    assert [c["file"] for c in summary["clips"]] == [r.output.name for r in run.reels]
    assert summary["ranker"] == "local"
    assert progress[-1] == 1.0 and progress == sorted(progress)


@requires_ffmpeg
def test_transcript_is_cached_between_runs(talk, tmp_path, whisper):
    out = tmp_path / "out"
    find_highlights(talk, SETTINGS, cache_dir=out)
    find_highlights(talk, SETTINGS, cache_dir=out)
    assert whisper.calls == 1
    assert (out / "talk.transcript.json").is_file()


@requires_ffmpeg
def test_corrupt_cache_is_ignored(talk, tmp_path, whisper):
    out = tmp_path / "out"
    out.mkdir()
    (out / "talk.transcript.json").write_text("{not json", encoding="utf-8")
    assert find_highlights(talk, SETTINGS, cache_dir=out)
    assert whisper.calls == 1


@requires_ffmpeg
def test_gemini_ranker_titles_are_used(talk, tmp_path, whisper):
    client = FakeClient({"clips": [pick(5, 7, title="Why it's slow", hook="One-line fix")]})
    picks = find_highlights(talk, SETTINGS, ranker="gemini", client=client)
    assert any(p.title == "Why it's slow" and p.hook == "One-line fix" for p in picks)
    assert "Why is this slow?" in client.prompts[0][0]


def test_unknown_ranker_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        find_highlights(tmp_path / "talk.mp4", SETTINGS, ranker="magic")


@requires_ffmpeg
def test_too_short_video_is_reported(tmp_path, monkeypatch):
    model = FakeWhisper()
    model.transcribe = lambda audio, **kw: (
        iter([SimpleNamespace(start=0.1, end=1.0, text="Hi.")]),
        SimpleNamespace(language="en", language_probability=0.9))
    monkeypatch.setattr(transcript, "load_model", lambda size: model)
    clip = tmp_path / "short.mp4"
    make_clip(clip, 2, [])
    with pytest.raises(ValueError, match="No part"):
        create_reels(clip, tmp_path / "out", SETTINGS)


@requires_ffmpeg
def test_cli_reels_dry_run_lists_picks(talk, tmp_path, whisper, capsys):
    out = tmp_path / "out"
    code = cli.main([str(talk), str(out), "--reels", "--clips", "2", "--min-length", "5",
                     "--max-length", "12", "--layout", "fit-bars", "--dry-run"])
    assert code == 0
    printed = capsys.readouterr().out
    assert "Here's the trick" in printed or "Why is this slow" in printed
    assert not list(out.glob("*.mp4"))


@requires_ffmpeg
def test_cli_reels_renders(talk, tmp_path, whisper):
    out = tmp_path / "out"
    code = cli.main([str(talk), str(out), "--reels", "--clips", "1", "--min-length", "5",
                     "--max-length", "12", "--layout", "fit-bars"])
    assert code == 0
    assert len(list(out.glob("talk_reel*.mp4"))) == 1


@requires_ffmpeg
@pytest.mark.parametrize("content", ["[]", "null", '{"key": 1}'])
def test_unusable_cache_contents_are_ignored(talk, tmp_path, whisper, content):
    out = tmp_path / "out"
    out.mkdir()
    (out / "talk.transcript.json").write_text(content, encoding="utf-8")
    assert find_highlights(talk, SETTINGS, cache_dir=out)


@requires_ffmpeg
def test_cancel_after_cached_transcript_stops_before_ranking(talk, tmp_path, whisper):
    from silence_remover.render import RenderCancelled
    out = tmp_path / "out"
    find_highlights(talk, SETTINGS, cache_dir=out)  # fills the cache
    with pytest.raises(RenderCancelled):
        find_highlights(talk, SETTINGS, cache_dir=out, is_cancelled=lambda: True)
