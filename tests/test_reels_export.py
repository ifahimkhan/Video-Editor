"""Reel export integration tests against real ffmpeg."""

import subprocess

import numpy as np
import pytest

from silence_remover import cli
from silence_remover.ffmpeg_tools import FFmpegError, probe
from silence_remover.reels.export import render_reel
from silence_remover.reels.settings import Aspect, Layout, ReelSettings
from silence_remover.render import RenderCancelled

from .conftest import requires_ffmpeg


def make_landscape_clip(path, duration_s: int = 6) -> None:
    subprocess.run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", f"testsrc2=size=320x180:rate=25:duration={duration_s}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration_s}:sample_rate=48000",
        "-map", "0:v", "-c:v", "libx264", "-preset", "ultrafast",
        "-map", "1:a", "-c:a", "aac", "-shortest", str(path),
    ], check=True)


def video_size(path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    w, h = out.stdout.strip().split(",")
    return int(w), int(h)


@pytest.fixture(scope="module")
def landscape_clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("reels") / "landscape.mp4"
    make_landscape_clip(path)
    return path


@requires_ffmpeg
def test_probe_reports_video_size(landscape_clip):
    info = probe(landscape_clip)
    assert (info.width, info.height) == (320, 180)


@requires_ffmpeg
@pytest.mark.parametrize("layout", list(Layout))
def test_render_reel_each_layout(landscape_clip, tmp_path, layout):
    progress = []
    out = render_reel(landscape_clip, tmp_path / f"{layout.value}.mp4", 1.0, 3.0,
                      ReelSettings(layout=layout, aspect=Aspect.SQUARE),
                      on_progress=progress.append)
    assert video_size(out) == (1080, 1080)
    info = probe(out)
    assert info.has_audio
    assert info.duration_s == pytest.approx(2.0, abs=0.15)
    assert progress[-1] == 1.0


@requires_ffmpeg
def test_render_reel_rejects_range_outside_video(landscape_clip, tmp_path):
    with pytest.raises(ValueError):
        render_reel(landscape_clip, tmp_path / "x.mp4", 5.0, 9.0, ReelSettings())
    with pytest.raises(ValueError):
        render_reel(landscape_clip, tmp_path / "x.mp4", 3.0, 2.0, ReelSettings())


@requires_ffmpeg
def test_render_reel_rejects_same_input_and_output(landscape_clip):
    with pytest.raises(FFmpegError):
        render_reel(landscape_clip, landscape_clip, 0.0, 1.0, ReelSettings())


@requires_ffmpeg
def test_render_reel_can_be_cancelled(landscape_clip, tmp_path):
    out = tmp_path / "cancelled.mp4"
    with pytest.raises(RenderCancelled):
        render_reel(landscape_clip, out, 0.0, 5.0, ReelSettings(),
                    is_cancelled=lambda: True)
    assert not out.exists()


@requires_ffmpeg
def test_cli_reel(landscape_clip, tmp_path):
    out = tmp_path / "cli.mp4"
    code = cli.main([str(landscape_clip), str(out), "--reel", "--start", "1",
                     "--end", "3", "--layout", "split", "--webcam", "240,135,80,45"])
    assert code == 0
    assert video_size(out) == (1080, 1920)


@requires_ffmpeg
def test_cli_reel_bad_webcam_is_an_error(landscape_clip, tmp_path):
    code = cli.main([str(landscape_clip), str(tmp_path / "o.mp4"), "--reel",
                     "--layout", "split", "--webcam", "nonsense"])
    assert code == 1


def frame_at(path, t: float):
    import numpy as np
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(path), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
    w, h = video_size(path)
    return np.frombuffer(raw, np.uint8).reshape(h, w)


@requires_ffmpeg
def test_render_reel_burns_captions_and_title(tmp_path):
    from silence_remover.transcript import Word
    clip = tmp_path / "flat.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=0x404040:size=320x180:"
        "rate=25:duration=4", "-f", "lavfi", "-i", "sine=duration=4",
        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(clip),
    ], check=True)
    words = [Word(1.0, 1.4, "hello"), Word(1.5, 2.5, "world")]
    plain = render_reel(clip, tmp_path / "plain.mp4", 0.5, 3.5,
                        ReelSettings(layout=Layout.FIT_BARS))
    captioned = render_reel(clip, tmp_path / "cap.mp4", 0.5, 3.5,
                            ReelSettings(layout=Layout.FIT_BARS, captions=True,
                                         title_overlay=True),
                            words=words, title="My title")
    # Captions sit at 72% height; the title near the top. Both add bright pixels.
    cap_band = slice(int(1920 * 0.68), int(1920 * 0.76))
    title_band = slice(100, 320)
    a, b = frame_at(plain, 1.2), frame_at(captioned, 1.2)
    assert (b[cap_band] > 200).sum() > 2000 and (a[cap_band] > 200).sum() == 0
    assert (b[title_band] > 200).sum() > 1000


@requires_ffmpeg
def test_render_reel_removes_silence_inside_clip(tmp_path):
    from silence_remover.segments import Segment

    from .conftest import make_clip
    clip = tmp_path / "pauses.mp4"
    make_clip(clip, 10, [(3, 5)])
    keep = (Segment(0, 2000), Segment(4000, 8000))  # clip 1-9 s, pause at 3-5 s
    out = render_reel(clip, tmp_path / "cut.mp4", 1.0, 9.0,
                      ReelSettings(layout=Layout.FIT_BARS, aspect=Aspect.SQUARE), keep=keep)
    assert probe(out).duration_s == pytest.approx(6.0, abs=0.15)


@requires_ffmpeg
def test_render_reel_all_silent_keep_is_an_error(landscape_clip, tmp_path):
    with pytest.raises(ValueError, match="silent"):
        render_reel(landscape_clip, tmp_path / "x.mp4", 1.0, 3.0, ReelSettings(), keep=())


@requires_ffmpeg
def test_cli_reel_with_silence_removal_and_title(tmp_path):
    from .conftest import make_clip
    clip = tmp_path / "pauses.mp4"
    make_clip(clip, 10, [(3, 5)])
    out = tmp_path / "o.mp4"
    code = cli.main([str(clip), str(out), "--reel", "--start", "1", "--end", "9",
                     "--layout", "fit-bars", "--remove-silence", "--title-overlay",
                     "--title", "Hello"])
    assert code == 0
    assert probe(out).duration_s < 7.5


@requires_ffmpeg
def test_long_captions_and_titles_stay_inside_the_frame(tmp_path):
    from silence_remover.transcript import Word
    clip = tmp_path / "flat.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=0x202020:size=320x180:"
        "rate=25:duration=3", "-c:v", "libx264", "-preset", "ultrafast", str(clip),
    ], check=True)
    words = [Word(0.2, 0.8, "WWWWWWWWWW"), Word(0.8, 2.5, "MMMMMMMMMMM")]
    out = render_reel(clip, tmp_path / "wide.mp4", 0, 3,
                      ReelSettings(layout=Layout.FIT_BARS, captions=True, title_overlay=True),
                      words=words, title="A very long title that keeps going and going on")
    frame = frame_at(out, 1.5)
    edges = np.concatenate([frame[:, :12], frame[:, -12:]], axis=1)
    assert (edges > 150).sum() == 0


@requires_ffmpeg
def test_cli_reel_keeps_pauses_without_flag(tmp_path):
    from .conftest import make_clip
    clip = tmp_path / "pauses.mp4"
    make_clip(clip, 10, [(3, 5)])
    out = tmp_path / "o.mp4"
    assert cli.main([str(clip), str(out), "--reel", "--start", "1", "--end", "9",
                     "--layout", "fit-bars"]) == 0
    assert probe(out).duration_s == pytest.approx(8.0, abs=0.15)


@requires_ffmpeg
def test_cli_reel_accepts_minutes_and_seconds(landscape_clip, tmp_path):
    out = tmp_path / "mmss.mp4"
    assert cli.main([str(landscape_clip), str(out), "--reel", "--start", "0:01",
                     "--end", "0:03.5", "--layout", "fit-bars"]) == 0
    assert probe(out).duration_s == pytest.approx(2.5, abs=0.15)


def test_cli_rejects_bad_time():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["a.mp4", "b.mp4", "--reel", "--start", "1:75"])


def format_tags(path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True)
    import json
    return {k.lower(): v for k, v in json.loads(out.stdout).get("format", {})
            .get("tags", {}).items()}


@pytest.fixture(scope="module")
def titled_source(tmp_path_factory):
    path = tmp_path_factory.mktemp("meta") / "titled.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
        "color=c=0x202020:size=320x180:rate=25:duration=4", "-f", "lavfi", "-i",
        "sine=duration=4", "-metadata", "title=Original long recording",
        "-metadata", "comment=private note", "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac", "-shortest", str(path)], check=True)
    return path


@requires_ffmpeg
def test_reel_does_not_copy_source_metadata(titled_source, tmp_path):
    out = render_reel(titled_source, tmp_path / "r.mp4", 0, 3,
                      ReelSettings(layout=Layout.FIT_BARS))
    tags = format_tags(out)
    assert "title" not in tags and "comment" not in tags


@requires_ffmpeg
def test_reel_metadata_title_is_the_clip_title(titled_source, tmp_path):
    out = render_reel(titled_source, tmp_path / "r.mp4", 0, 3,
                      ReelSettings(layout=Layout.FIT_BARS), title="Fix it in one line")
    assert format_tags(out)["title"] == "Fix it in one line"
