"""Reel export integration tests against real ffmpeg."""

import subprocess

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
