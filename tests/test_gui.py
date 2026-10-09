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


def test_swap_audio_button_replaces_soundtrack(app, window, tmp_path, monkeypatch):
    import subprocess

    from silence_remover.ffmpeg_tools import probe

    assert window.btn_swap_audio.isEnabled()
    clean = tmp_path / "clean.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "sine=frequency=880:duration=12", str(clean)], check=True)
    out = tmp_path / "swapped"                                  # no extension typed
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName",
                        lambda *a, **k: (str(clean), ""))
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        lambda *a, **k: (str(out), ""))
    shown = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "information",
                        lambda *a, **k: shown.append(a[2]))
    monkeypatch.setattr(window, "_ask_swap_trim", lambda: False)  # just swap
    window._choose_swap_audio()
    assert not window.btn_swap_audio.isEnabled()                # busy while swapping
    _wait_idle(window, app)
    result = out.with_suffix(".mp4")
    info = probe(result)
    assert info.has_video and info.has_audio
    assert info.duration_s == pytest.approx(12, abs=0.2)        # silence kept
    assert shown and str(result) in shown[0]
    assert window.btn_swap_audio.isEnabled()


def _swap_with_choice(app, window, tmp_path, monkeypatch, choice):
    import subprocess

    clean = tmp_path / "clean.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "sine=frequency=880:duration=12", str(clean)], check=True)
    out = tmp_path / "swapped.mp4"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName",
                        lambda *a, **k: (str(clean), ""))
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        lambda *a, **k: (str(out), ""))
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(window, "_ask_swap_trim", lambda: choice)
    window._choose_swap_audio()
    _wait_idle(window, app)
    return out


def test_swap_audio_can_also_remove_silence(app, window, tmp_path, monkeypatch):
    from silence_remover.ffmpeg_tools import probe

    out = _swap_with_choice(app, window, tmp_path, monkeypatch, True)
    info = probe(out)
    assert info.has_video and info.has_audio
    assert info.duration_s == pytest.approx(window._result.kept_ms / 1000, abs=0.3)
    assert not list(tmp_path.glob("*.swap-tmp*"))               # temp file cleaned up


def test_swap_audio_choice_cancelled_does_nothing(app, window, tmp_path, monkeypatch):
    out = _swap_with_choice(app, window, tmp_path, monkeypatch, None)
    assert not out.exists()
    assert window.btn_swap_audio.isEnabled()


def test_swap_audio_disabled_for_audio_only_files(app, tmp_path):
    clip = tmp_path / "voice.m4a"
    make_clip(clip, 2, [], video=False)
    w = MainWindow()
    w._input_path = str(clip)
    w._on_analysis_done(MediaAnalysis(analyze(clip), read_keyframes(clip)))
    assert not w.btn_swap_audio.isEnabled()
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


def test_remove_echo_button_saves_clean_audio(app, tmp_path, monkeypatch):
    import numpy as np

    from silence_remover.ffmpeg_tools import probe

    from .test_echo_removal import SPEECH, add_echo, decode, encode

    echoed = encode(tmp_path / "echoed.wav", add_echo(0.5 * np.tile(decode(SPEECH)[0], 2)))
    w = MainWindow()
    w._input_path = str(echoed)
    w._on_analysis_done(MediaAnalysis(analyze(echoed), read_keyframes(echoed)))
    assert w.btn_remove_echo.isEnabled()
    out = tmp_path / "no_echo"                         # no extension typed
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        lambda *a, **k: (str(out), ""))
    shown = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "information",
                        lambda *a, **k: shown.append(a[2]))
    w._choose_remove_echo()
    assert not w.btn_remove_echo.isEnabled()           # busy while cleaning
    _wait_idle(w, app)
    mp3 = out.with_suffix(".mp3")
    assert probe(mp3).has_audio
    assert shown and "Echo found at 26." in shown[0] and str(mp3) in shown[0]
    assert w.btn_remove_echo.isEnabled()
    w.close()


def test_remove_echo_reports_audio_without_echo(app, window, tmp_path, monkeypatch):
    out = tmp_path / "x.mp3"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        lambda *a, **k: (str(out), ""))
    errors = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical",
                        lambda *a, **k: errors.append(a[2]))
    window._choose_remove_echo()
    _wait_idle(window, app)
    assert errors and "No echo" in errors[0]
    assert not out.exists()


def test_remove_echo_disabled_until_file_analyzed(app):
    w = MainWindow()
    assert not w.btn_remove_echo.isEnabled()
    w.close()


def test_transcript_button_saves_captions(app, window, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from silence_remover import transcript

    class FakeModel:
        def transcribe(self, audio, **kwargs):
            self.task = kwargs["task"]
            info = SimpleNamespace(language="en", language_probability=0.9)
            return iter([SimpleNamespace(start=0.5, end=2.0, text=" Hi.")]), info

    model = FakeModel()
    monkeypatch.setattr(transcript, "load_model", lambda size: model)
    window._transcript_available = True
    window._set_busy(False)
    assert window.btn_transcript.isEnabled()
    out = tmp_path / "talk"                                     # no extension typed
    monkeypatch.setattr(window, "_ask_transcript_language", lambda: True)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        lambda *a, **k: (str(out), "Web subtitles (*.vtt)"))
    shown = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "information",
                        lambda *a, **k: shown.append(a[2]))
    window._choose_transcript()
    assert not window.btn_transcript.isEnabled()                # busy while working
    _wait_idle(window, app)
    result = out.with_suffix(".vtt")
    assert result.read_text(encoding="utf-8").startswith("WEBVTT")
    assert model.task == "translate"
    assert shown and str(result) in shown[0] and "1 captions" in shown[0]


def test_transcript_cancelled_choice_does_nothing(app, window, monkeypatch):
    monkeypatch.setattr(window, "_ask_transcript_language", lambda: None)
    called = []
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        lambda *a, **k: called.append(1) or ("", ""))
    window._choose_transcript()
    assert not called and window._worker is None
