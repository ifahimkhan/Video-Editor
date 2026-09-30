"""Fast lossless export: keyframe snapping and stream-copy cutting."""

import subprocess

import numpy as np
import pytest

from silence_remover import cli
from silence_remover.analysis import analyze
from silence_remover.lossless import (
    _parse_packet_line,
    build_concat_list,
    read_keyframes,
    render_lossless,
    snap_result,
    snap_to_keyframes,
)
from silence_remover.render import RenderCancelled
from silence_remover.segments import DetectionResult, DetectionSettings, Segment, detect

from .conftest import make_clip, requires_ffmpeg

KF = np.array([0.0, 1000.0, 2000.0, 3000.0, 4000.0])


def run(args):
    return subprocess.run(args, capture_output=True, text=True, check=True)


def stream_info(path, stream):
    out = run(["ffprobe", "-v", "error", "-select_streams", f"{stream}:0",
               "-show_entries", "stream=codec_name,duration", "-of", "csv=p=0", str(path)])
    codec, duration = out.stdout.strip().split(",")
    return codec, float(duration)


def frame_hashes(path, frames):
    out = run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0",
               "-frames:v", str(frames), "-f", "framemd5", "-"])
    return [line.rsplit(",", 1)[-1].strip() for line in out.stdout.splitlines()
            if line and not line.startswith("#")]


@pytest.fixture(scope="session")
def gop_clip(tmp_path_factory):
    """12 s, keyframe every 1 s (30 fps, -g 30, B-frames), silent 2.3-4.6 and 7.2-9.5 s."""
    path = tmp_path_factory.mktemp("gop") / "gop.mp4"
    make_clip(path, 12, [(2.3, 4.6), (7.2, 9.5)],
              video_args=["-g", "30", "-bf", "2"], fps=30)
    return path


# ---------- pure logic ----------

@pytest.mark.parametrize("line, expected", [
    ("12.345000,K__\n", 12.345),
    ("12.345000,__\n", None),
    ("N/A,K__\n", None),
    ("\n", None),
])
def test_parse_packet_line(line, expected):
    assert _parse_packet_line(line) == expected


def test_snap_moves_start_back_to_keyframe():
    keep = (Segment(0, 1500), Segment(2600, 3400))
    assert snap_to_keyframes(keep, KF, 5000) == (Segment(0, 1500), Segment(2000, 3400))


def test_snap_merges_segments_that_now_overlap():
    keep = (Segment(1000, 2100), Segment(2500, 3000))
    assert snap_to_keyframes(keep, KF, 5000) == (Segment(1000, 3000),)


def test_snap_keeps_exact_keyframe_starts_and_handles_no_keyframes():
    keep = (Segment(3000, 3500),)
    assert snap_to_keyframes(keep, KF, 5000) == keep
    assert snap_to_keyframes(keep, np.empty(0), 5000) == keep


def test_snap_result_only_grows_kept_time():
    result = DetectionResult(keep=(Segment(1200, 1800), Segment(3300, 4500)), duration_ms=5000)
    snapped = snap_result(result, KF)
    assert snapped.kept_ms >= result.kept_ms
    assert snapped.keep == (Segment(1000, 1800), Segment(3000, 4500))


def test_concat_list_quotes_paths(tmp_path):
    odd = tmp_path / "it's a clip.mp4"
    text = build_concat_list([odd])
    assert text.startswith("ffconcat version 1.0\n")
    assert "it'\\''s a clip.mp4'" in text
    with pytest.raises(ValueError):
        build_concat_list([])


# ---------- real ffmpeg ----------

@requires_ffmpeg
def test_read_keyframes(gop_clip, tmp_path):
    kf = read_keyframes(gop_clip)
    np.testing.assert_allclose(kf, np.arange(0, 12000, 1000), atol=1)
    audio = tmp_path / "a.m4a"
    make_clip(audio, 2, [], video=False)
    assert read_keyframes(audio).size == 0


@requires_ffmpeg
def test_lossless_export_is_clean_synced_and_bit_exact(gop_clip, tmp_path):
    profile = analyze(gop_clip, use_vad=False)
    result = detect(profile, DetectionSettings(padding_ms=100))
    snapped = snap_result(result, read_keyframes(gop_clip))
    assert snapped.keep[1].start_ms == 4000          # 4.5 s snapped back to 4 s

    progress = []
    out = render_lossless(gop_clip, tmp_path / "out.mp4", snapped.keep,
                          on_progress=progress.append)

    decode = subprocess.run(["ffmpeg", "-v", "error", "-i", str(out), "-f", "null", "-"],
                            capture_output=True, text=True)
    assert decode.stderr.strip() == ""               # no out-of-order frames at joins

    v_codec, v_dur = stream_info(out, "v")
    a_codec, a_dur = stream_info(out, "a")
    assert (v_codec, a_codec) == ("h264", "aac")
    assert abs(v_dur - a_dur) < 0.06
    assert v_dur == pytest.approx(snapped.kept_ms / 1000, abs=0.4)

    # First segment starts at 0: its frames must be the untouched source frames.
    assert frame_hashes(out, 60) == frame_hashes(gop_clip, 60)
    assert progress[-1] == 1.0


@requires_ffmpeg
def test_lossless_audio_only(tmp_path):
    clip = tmp_path / "voice.m4a"
    make_clip(clip, 6, [(2, 4)], video=False)
    keep = detect(analyze(clip, use_vad=False), DetectionSettings()).keep
    out = render_lossless(clip, tmp_path / "out.m4a", keep)
    assert stream_info(out, "a")[1] < 5


@requires_ffmpeg
def test_lossless_cancel_and_errors(gop_clip, tmp_path):
    out = tmp_path / "x.mp4"
    with pytest.raises(RenderCancelled):
        render_lossless(gop_clip, out, (Segment(0, 3000),), is_cancelled=lambda: True)
    assert not out.exists()
    with pytest.raises(ValueError):
        render_lossless(gop_clip, out, ())


@requires_ffmpeg
def test_cli_lossless(gop_clip, tmp_path, capsys):
    out = tmp_path / "cli.mp4"
    assert cli.main([str(gop_clip), str(out), "--lossless"]) == 0
    assert out.exists()
    assert "Lossless: +" in capsys.readouterr().out
