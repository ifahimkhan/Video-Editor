"""Offscreen GUI tests: window wiring, timeline, workers."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PyQt6.QtWidgets")
pytest.importorskip("pyqtgraph")

from silence_remover.analysis import analyze  # noqa: E402
from silence_remover.gui.main_window import MainWindow, Timeline  # noqa: E402
from silence_remover.gui.timeline import TimelineWidget  # noqa: E402
from silence_remover.gui.workers import MediaAnalysis, RenderWorker  # noqa: E402
from silence_remover.lossless import read_keyframes  # noqa: E402
from silence_remover.segments import DetectionSettings  # noqa: E402

from .conftest import make_clip, requires_ffmpeg  # noqa: E402

pytestmark = requires_ffmpeg


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(scope="module")
def gop_media(tmp_path_factory):
    path = tmp_path_factory.mktemp("gui") / "gop.mp4"
    make_clip(path, 12, [(2.3, 4.6), (7.2, 9.5)], video_args=["-g", "30"], fps=30)
    return path, MediaAnalysis(analyze(path), read_keyframes(path))


@pytest.fixture
def window(app, gop_media):
    path, analysis = gop_media
    w = MainWindow()
    w.show()
    w._input_path = str(path)
    w._on_analysis_done(analysis)
    app.processEvents()
    yield w
    w.close()


def test_uses_pyqtgraph_timeline():
    assert Timeline.__name__ == "WaveformTimeline"


def test_detection_preview_and_stats(window):
    assert "2 cuts" in window.lbl_stats.text()
    assert window.btn_export.isEnabled()


def test_dragging_threshold_line_moves_slider(window, app):
    window.timeline._thresh.setValue(0.6)
    app.processEvents()
    assert window.threshold.value() == pytest.approx(-29, abs=1)


def test_lossless_mode_previews_keyframe_snapping(window, app):
    precise_kept = window._result.kept_ms
    window.export_mode.setCurrentIndex(1)
    app.processEvents()
    assert window._result.kept_ms > precise_kept
    assert "kept to start on keyframes" in window.lbl_stats.text()
    assert all(s.start_ms % 1000 == 0 for s in window._result.keep)


def test_voice_mode_shows_probability_lane(window, app):
    if not window._profile.has_vad:
        pytest.skip("VAD unavailable")
    window.mode.setCurrentIndex(1)
    app.processEvents()
    assert window.timeline._lane.isVisible()
    window.timeline._vad_line.setValue(0.3)
    app.processEvents()
    assert window.vad_threshold.value() == 30


def test_zoom_redraws_only_visible_range(window, app):
    window.timeline._wave.setXRange(5, 6, padding=0)
    app.processEvents()
    assert len(window.timeline._peak_top.getData()[0]) < 110


@pytest.mark.parametrize("lossless", [False, True])
def test_render_worker(window, tmp_path, lossless):
    out = tmp_path / f"out_{lossless}.mp4"
    worker = RenderWorker(window._input_path, str(out), window._result.keep, lossless)
    results, errors = [], []
    worker.finished_ok.connect(results.append)
    worker.failed.connect(errors.append)
    worker.run()  # synchronously, in this thread
    assert errors == [] and results == [str(out)]
    assert out.stat().st_size > 0


def test_fallback_timeline_accepts_same_calls(app, gop_media, window):
    fallback = TimelineWidget()
    fallback.set_profile(gop_media[1].profile)
    fallback.set_detection(window._result, DetectionSettings())
    fallback.grab()


def _wait_idle(window, app, timeout_s=60):
    import time
    deadline = time.monotonic() + timeout_s
    while window._worker is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert window._worker is None, "worker did not finish"


def test_threaded_analysis_then_lossless_export(app, gop_media, tmp_path, monkeypatch):
    """Real QThreads: open -> analyze -> export, as the buttons do it."""
    from silence_remover.gui.workers import AnalysisWorker

    path = str(gop_media[0])
    w = MainWindow()
    w._input_path = path
    w._start(AnalysisWorker(path), w._on_analysis_done, "Analyzing")
    assert not w.btn_open.isEnabled()          # busy while analyzing
    _wait_idle(w, app)
    assert w._profile is not None and w._keyframes_ms.size == 12
    assert w.btn_export.isEnabled()

    w.export_mode.setCurrentIndex(1)
    out = tmp_path / "threaded.mp4"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        lambda *a, **k: (str(out), ""))
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *a, **k: None)
    w._choose_output()
    _wait_idle(w, app)
    assert out.exists() and "Saved" in w.statusBar().currentMessage()
    w.close()


def test_export_audio_button_saves_mp3(app, window, tmp_path, monkeypatch):
    from silence_remover.ffmpeg_tools import probe

    assert window.btn_export_audio.isEnabled()
    out = tmp_path / "chosen_name"                 # no extension typed
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        lambda *a, **k: (str(out), ""))
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *a, **k: None)
    window._choose_audio_output()
    assert not window.btn_export_audio.isEnabled()  # busy while exporting
    _wait_idle(window, app)
    mp3 = out.with_suffix(".mp3")
    assert mp3.exists()
    info = probe(mp3)
    assert info.has_audio and not info.has_video
    assert window.btn_export_audio.isEnabled()


def test_export_audio_disabled_until_file_analyzed(app):
    w = MainWindow()
    assert not w.btn_export_audio.isEnabled()
    w.close()
