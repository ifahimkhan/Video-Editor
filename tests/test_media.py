"""Integration tests against real ffmpeg."""

import subprocess

import numpy as np
import pytest

from silence_remover import cli
from silence_remover.analysis import AnalysisCancelled, analyze, frames_to_db
from silence_remover.ffmpeg_tools import FFmpegError, probe
from silence_remover.render import RenderCancelled, build_filter_graph, render
from silence_remover.segments import DetectionSettings, Segment, detect

from .conftest import make_clip, requires_ffmpeg


def stream_duration(path, stream: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", f"{stream}:0",
         "-show_entries", "stream=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def test_frames_to_db_levels():
    full_scale = np.full(160, 32767, dtype=np.int16)
    silence = np.zeros(160, dtype=np.int16)
    db = frames_to_db(np.concatenate([full_scale, silence, silence[:50]]))
    assert db.shape == (2,)
    assert db[0] == pytest.approx(0.0, abs=0.01)
    assert db[1] == -100.0


def test_filter_graph_uses_offset_timestamps():
    graph = build_filter_graph((Segment(0, 1000), Segment(3000, 4000)), 5000, True, True)
    assert "between(t,0.000,1.000)+between(t,3.000,4.000)" in graph
    assert "gt(T,1.000)*2.000+gt(T,4.000)*1.000" in graph
    assert "[v]" in graph and "[a]" in graph


def test_filter_graph_rejects_empty_keep():
    with pytest.raises(ValueError):
        build_filter_graph((), 1000, True, True)


@requires_ffmpeg
def test_analyze_and_detect_real_clip(speech_clip):
    profile = analyze(speech_clip)
    assert abs(profile.duration_ms - 10_000) <= 50
    result = detect(profile, DetectionSettings(-35, 500, 0))
    # 2-4 s pause is cut, the 0.3 s pause at 6 s is not.
    assert len(result.cuts) == 1
    cut = result.cuts[0]
    assert abs(cut.start_ms - 2000) <= 40 and abs(cut.end_ms - 4000) <= 40


@requires_ffmpeg
def test_render_keeps_audio_and_video_in_sync(speech_clip, tmp_path):
    keep = detect(analyze(speech_clip), DetectionSettings(-35, 500, 100)).keep
    expected_s = sum(s.length_ms for s in keep) / 1000
    progress = []
    out = render(speech_clip, tmp_path / "out.mp4", keep, on_progress=progress.append)

    assert stream_duration(out, "v") == pytest.approx(expected_s, abs=0.08)
    assert stream_duration(out, "a") == pytest.approx(expected_s, abs=0.08)
    assert progress[-1] == 1.0


@requires_ffmpeg
def test_many_cuts_do_not_drift(tmp_path):
    """60 alternating cuts: audio and video must still end together."""
    clip = tmp_path / "choppy.mp4"
    silences = [(i + 0.4, i + 1.0) for i in range(0, 60, 1)]
    make_clip(clip, 61, silences)
    keep = detect(analyze(clip), DetectionSettings(-35, 300, 50)).keep
    assert len(keep) >= 55
    out = render(clip, tmp_path / "out.mp4", keep)
    assert abs(stream_duration(out, "v") - stream_duration(out, "a")) < 0.06


@requires_ffmpeg
def test_audio_only_input(tmp_path):
    clip = tmp_path / "voice.m4a"
    make_clip(clip, 6, [(2, 4)], video=False)
    keep = detect(analyze(clip), DetectionSettings()).keep
    out = render(clip, tmp_path / "out.m4a", keep)
    info = probe(out)
    assert not info.has_video and info.has_audio
    assert info.duration_s < 5


@requires_ffmpeg
def test_cancel_analysis_and_render(speech_clip, tmp_path):
    with pytest.raises(AnalysisCancelled):
        analyze(speech_clip, is_cancelled=lambda: True)
    out = tmp_path / "cancelled.mp4"
    with pytest.raises(RenderCancelled):
        render(speech_clip, out, (Segment(0, 9000),), is_cancelled=lambda: True)
    assert not out.exists()


@requires_ffmpeg
def test_errors_are_reported(speech_clip, tmp_path):
    with pytest.raises(FFmpegError):
        analyze(tmp_path / "missing.mp4")
    with pytest.raises(FFmpegError):
        render(speech_clip, speech_clip, (Segment(0, 1000),))
    silent_video = tmp_path / "noaudio.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc2=duration=1", str(silent_video)], check=True)
    with pytest.raises(FFmpegError, match="no audio"):
        analyze(silent_video)


@requires_ffmpeg
def test_cli_dry_run_and_export(speech_clip, tmp_path, capsys):
    assert cli.main([str(speech_clip), str(tmp_path / "x.mp4"), "--dry-run"]) == 0
    assert "1 cuts" in capsys.readouterr().out
    assert cli.main([str(speech_clip), str(tmp_path / "y.mp4")]) == 0
    assert (tmp_path / "y.mp4").exists()
    assert cli.main([str(speech_clip), str(tmp_path / "z.mp4"), "-t", "5"]) == 1


def test_sum_expr_is_balanced_and_complete():
    from silence_remover.render import sum_expr
    assert sum_expr(["a"]) == "a"
    assert sum_expr(["a", "b", "c", "d"]) == "((a+b)+(c+d))"
    expr = sum_expr([f"x{i}" for i in range(1000)])
    assert all(f"x{i}" in expr for i in (0, 499, 999))
    # Never a flat a+b+c chain: each parenthesis group holds at most one '+'.
    plus_per_group = [0]
    for ch in expr:
        if ch == "(":
            plus_per_group.append(0)
        elif ch == ")":
            assert plus_per_group.pop() <= 1
        elif ch == "+":
            plus_per_group[-1] += 1
    assert plus_per_group == [0]
    with pytest.raises(ValueError):
        sum_expr([])


@requires_ffmpeg
def test_precise_export_with_more_than_100_cuts(tmp_path):
    """Regression: FFmpeg rejects flat expressions over 100 terms (ENOMEM)."""
    clip = tmp_path / "many.mp4"
    make_clip(clip, 30, [])
    keep = tuple(Segment(i * 200, i * 200 + 100) for i in range(150))
    out = render(clip, tmp_path / "out.mp4", keep)
    assert stream_duration(out, "v") == pytest.approx(15.0, abs=0.1)
    assert stream_duration(out, "a") == pytest.approx(15.0, abs=0.1)
