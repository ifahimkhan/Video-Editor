"""Tests for tracked layouts: planning, crop expressions and real renders."""

import re
import subprocess

import numpy as np
import pytest

from silence_remover import cli
from silence_remover.ffmpeg_tools import probe
from silence_remover.reels.analyze import ClipAnalysis, analyze_clip, resolve_webcam
from silence_remover.reels.export import render_reel
from silence_remover.reels.framing import Shot
from silence_remover.reels.motion import build_tracked_graph, piecewise_expr
from silence_remover.reels.planner import make_plan
from silence_remover.reels.settings import Aspect, Layout, Rect, ReelSettings

from .conftest import requires_ffmpeg
from .test_reels_export import video_size
from .test_reels_face_detect import FACE_CENTER, FACE_JPG, requires_face_model

FPS = 5
CAM = Rect(1600, 760, 320, 320)
TYPING_Y = 300


def eval_expr(expr: str, t: float) -> float:
    """Evaluate an ffmpeg expression made of if/lt/gte and arithmetic."""
    py = re.sub(r"\bif\(", "IF(", expr)
    return float(eval(py, {"IF": lambda c, a, b: a if c else b,
                           "lt": lambda a, b: a < b,
                           "gte": lambda a, b: a >= b, "t": t}))


def test_piecewise_constant_is_a_number():
    assert piecewise_expr(np.array([0, 1, 2.]), np.array([5, 5, 5.])) == "5"


def test_piecewise_expr_interpolates_and_clamps():
    times = np.array([0, 1, 2, 3, 4.])
    values = np.array([0, 100, 100, 300, 0.])
    expr = piecewise_expr(times, values)
    assert eval_expr(expr, 0) == 0
    assert eval_expr(expr, 0.5) == pytest.approx(50)
    assert eval_expr(expr, 1.5) == pytest.approx(100)
    assert eval_expr(expr, 2.5) == pytest.approx(200)
    assert eval_expr(expr, 3.75) == pytest.approx(75)
    assert eval_expr(expr, 99) == 0  # held after the last sample


def test_piecewise_expr_handles_many_samples():
    times = np.arange(900) / FPS
    values = np.round(500 + 400 * np.sin(times))
    expr = piecewise_expr(times, values)
    for t in (0.0, 13.3, 100.1, 179.0):
        assert eval_expr(expr, t) == pytest.approx(np.interp(t, times, values), abs=0.51)


def fake_analysis(n: int = 50, typing_from: int = 20) -> ClipAnalysis:
    typing = np.arange(n) >= typing_from
    focus = np.where(typing[:, None], [[400.0, TYPING_Y]], np.nan)
    return ClipAnalysis(
        fps=FPS, duration_s=n / FPS,
        faces=np.tile([1740.0, 834.0], (n, 1)),
        focus=focus, typing=typing, jumps=np.zeros(n, bool),
        webcam=CAM, screen_color="0x1e1e1e",
    )


@pytest.mark.parametrize("layout", [m for m in Layout if m.tracked])
def test_tracked_graphs_end_in_v(layout):
    plan = make_plan(fake_analysis(), ReelSettings(layout=layout), (1920, 1080))
    graph = build_tracked_graph(plan)
    assert graph.startswith("[0:v:0]") and graph.rstrip().endswith("[v]")


def test_code_graph_covers_webcam_and_moves_crop():
    plan = make_plan(fake_analysis(), ReelSettings(layout=Layout.CODE), (1920, 1080))
    graph = build_tracked_graph(plan)
    assert f"drawbox=x={CAM.x}:y={CAM.y}:w={CAM.w}:h={CAM.h}:color=0x1e1e1e" in graph
    assert "x='if(" in graph


def test_auto_plan_switches_to_code_and_overlays_it():
    plan = make_plan(fake_analysis(), ReelSettings(layout=Layout.AUTO), (1920, 1080))
    assert [s.kind for s in plan.shots] == ["speaker", "code"]
    graph = build_tracked_graph(plan)
    assert "overlay" in graph and "enable='between(t," in graph


def test_auto_without_typing_is_speaker_only():
    plan = make_plan(fake_analysis(typing_from=999), ReelSettings(layout=Layout.AUTO),
                     (1920, 1080))
    assert plan.shots == (Shot(0.0, 10.0, "speaker"),)
    assert "overlay" not in build_tracked_graph(plan)


def test_split_plan_uses_panel_sizes():
    plan = make_plan(fake_analysis(), ReelSettings(layout=Layout.SPLIT), (1920, 1080))
    assert plan.code.w / plan.code.h == pytest.approx(1080 / 1152, abs=0.01)
    assert plan.speaker.w / plan.speaker.h == pytest.approx(1080 / 768, abs=0.02)


@pytest.fixture(scope="module")
def tutorial_video(tmp_path_factory):
    """6 s screen recording: face bottom-right, 'code' typed from 1 s to 4 s."""
    path = tmp_path_factory.mktemp("tut") / "tutorial.mp4"
    chars = ",".join(
        f"drawbox=x={200 + i * 30}:y={TYPING_Y}:w=20:h=24:color=white:t=fill"
        f":enable='gte(t,{1 + i * 0.2:.1f})'"
        for i in range(15)
    )
    subprocess.run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", "color=c=0x1e1e1e:size=1920x1080:rate=25:duration=6",
        "-loop", "1", "-i", str(FACE_JPG),
        "-f", "lavfi", "-i", "sine=duration=6",
        "-filter_complex",
        f"[0:v]{chars}[scr];[scr][1:v]overlay={CAM.x}:{CAM.y}:shortest=1,format=yuv420p[v]",
        "-map", "[v]", "-map", "2:a", "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac", "-t", "6", str(path),
    ], check=True)
    return path


@requires_ffmpeg
@requires_face_model
def test_analyze_clip_finds_face_and_typing(tutorial_video):
    info = probe(tutorial_video)
    webcam = resolve_webcam(tutorial_video, info, ReelSettings())
    assert abs(webcam.x - CAM.x) <= 6 and abs(webcam.y - CAM.y) <= 6
    analysis = analyze_clip(tutorial_video, info, 0.0, 6.0, webcam, need_faces=True)
    found = ~np.isnan(analysis.faces[:, 0])
    assert found.mean() > 0.8
    assert np.nanmedian(analysis.faces[:, 0]) == pytest.approx(CAM.x + FACE_CENTER[0], abs=15)
    t = np.arange(len(analysis.typing)) / FPS
    assert analysis.typing[(t > 1.8) & (t < 4)].all()
    assert not analysis.typing[t > 5.6].any()
    fx, fy = np.nanmedian(analysis.focus[(t > 2) & (t < 4)], axis=0)
    assert 200 <= fx <= 650 and fy == pytest.approx(TYPING_Y + 12, abs=40)
    assert analysis.screen_color == "0x1e1e1e"


@requires_ffmpeg
@pytest.mark.parametrize("layout", [m for m in Layout if m.tracked])
def test_render_tracked_layouts(tutorial_video, tmp_path, layout):
    progress = []
    out = render_reel(tutorial_video, tmp_path / f"{layout.value}.mp4", 0.5, 5.5,
                      ReelSettings(layout=layout, aspect=Aspect.VERTICAL),
                      on_progress=progress.append)
    assert video_size(out) == (1080, 1920)
    assert probe(out).duration_s == pytest.approx(5.0, abs=0.15)
    assert progress[-1] == 1.0
    assert progress == sorted(progress)


@requires_ffmpeg
def test_cli_auto_reel_with_zoom(tutorial_video, tmp_path):
    out = tmp_path / "auto.mp4"
    code = cli.main([str(tutorial_video), str(out), "--reel", "--layout", "auto",
                     "--code-zoom", "1.5", "--end", "5"])
    assert code == 0 and video_size(out) == (1080, 1920)


def test_piecewise_expr_rejects_empty_path():
    with pytest.raises(ValueError):
        piecewise_expr(np.array([]), np.array([]))
