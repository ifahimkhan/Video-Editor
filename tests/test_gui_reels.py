"""Offscreen tests for the Create reels dialog."""

import json
import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PyQt6.QtWidgets")
QtCore = pytest.importorskip("PyQt6.QtCore")

from silence_remover import transcript  # noqa: E402
from silence_remover.gui.reels_dialog import ReelsDialog  # noqa: E402
from silence_remover.reels.settings import Layout  # noqa: E402

from .conftest import make_clip, requires_ffmpeg  # noqa: E402
from .test_reels_pipeline import FakeWhisper  # noqa: E402

pytestmark = requires_ffmpeg


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(scope="module")
def talk(tmp_path_factory):
    path = tmp_path_factory.mktemp("reels_gui") / "talk.mp4"
    make_clip(path, 37, [])
    return path


@pytest.fixture
def dialog(app, talk, tmp_path, monkeypatch):
    monkeypatch.setattr(transcript, "load_model", lambda size: FakeWhisper())
    d = ReelsDialog(str(talk))
    form = d.form
    form.layout_box.setCurrentIndex(form.layout_box.findData(Layout.FIT_BARS))
    form.min_length.setValue(5)
    form.max_length.setValue(12)
    form.clips.setValue(2)
    d.out_dir.setText(str(tmp_path / "out"))
    d.show()
    yield d
    d.close()


def wait_idle(d, app, timeout_s=60):
    deadline = time.monotonic() + timeout_s
    while d._worker is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert d._worker is None, "worker did not finish"


def test_defaults_keep_pauses_and_suggest_output_folder(app, talk, monkeypatch):
    d = ReelsDialog(str(talk))
    settings = d.form.settings()
    assert settings.remove_silence is False
    assert not d.form.remove_silence.isChecked()
    assert d.out_dir.text().endswith("talk_reels")
    assert not d.btn_create.isEnabled()  # nothing found yet
    d.close()


def test_controls_enable_with_layout(dialog):
    form = dialog.form
    form.layout_box.setCurrentIndex(form.layout_box.findData(Layout.SPLIT))
    assert form.split_ratio.isEnabled() and form.code_zoom.isEnabled()
    form.layout_box.setCurrentIndex(form.layout_box.findData(Layout.FIT_BLUR))
    assert not form.split_ratio.isEnabled() and not form.code_zoom.isEnabled()
    form.captions.setChecked(False)
    assert not form.caption_style.isEnabled()


def test_find_then_create_only_ticked_clips(dialog, app, tmp_path, monkeypatch):
    dialog._find()
    assert not dialog.btn_find.isEnabled()  # busy
    wait_idle(dialog, app)
    table = dialog.table
    assert table.rowCount() == 2
    assert dialog.btn_create.isEnabled()

    table.selectRow(0)
    app.processEvents()
    assert not dialog.preview.pixmap().isNull()

    table.item(1, 0).setCheckState(QtCore.Qt.CheckState.Unchecked)
    table.item(0, 4).setText("Edited title")
    shown = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "information",
                        lambda *a, **k: shown.append(a[2]))
    dialog._create()
    wait_idle(dialog, app)
    out = tmp_path / "out"
    assert [p.name for p in out.glob("*.mp4")] == ["talk_reel01.mp4"]
    summary = json.loads((out / "reels.json").read_text(encoding="utf-8"))
    assert summary["clips"][0]["title"] == "Edited title"
    assert shown and "talk_reel01.mp4" in shown[0]
    assert dialog.btn_open_folder.isEnabled()  # opened only when asked


def test_create_with_nothing_ticked_warns(dialog, app, monkeypatch):
    dialog._find()
    wait_idle(dialog, app)
    for row in range(dialog.table.rowCount()):
        dialog.table.item(row, 0).setCheckState(QtCore.Qt.CheckState.Unchecked)
    errors = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *a, **k: errors.append(a[2]))
    dialog._create()
    assert errors and dialog._worker is None


def test_invalid_lengths_are_reported(dialog, monkeypatch):
    dialog.form.min_length.setValue(60)
    dialog.form.max_length.setValue(30)
    errors = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *a, **k: errors.append(a[2]))
    dialog._find()
    assert errors and dialog._worker is None


def test_cancel_stops_finding(dialog, app):
    dialog._find()
    dialog._cancel()
    wait_idle(dialog, app)
    assert dialog.btn_find.isEnabled()


def test_main_window_opens_reels_dialog(app, talk, monkeypatch):
    from silence_remover.analysis import analyze
    from silence_remover.gui import main_window
    from silence_remover.gui.workers import MediaAnalysis
    from silence_remover.lossless import read_keyframes

    opened = []
    monkeypatch.setattr(main_window.ReelsDialog, "exec", lambda self: opened.append(self))
    monkeypatch.setattr(main_window, "transcript_available", lambda: True)
    w = main_window.MainWindow()
    assert not w.btn_reels.isEnabled()
    w._input_path = str(talk)
    w._on_analysis_done(MediaAnalysis(analyze(talk), read_keyframes(talk)))
    assert w.btn_reels.isEnabled()
    w._open_reels()
    assert opened and opened[0].input_path == str(talk)
    w.close()


def test_preview_survives_bad_layout_math(dialog, app, monkeypatch):
    from silence_remover.gui import reels_dialog
    dialog._find()
    wait_idle(dialog, app)
    dialog.table.selectRow(0)
    app.processEvents()

    def boom(*a, **k):
        raise ValueError("odd numbers")
    monkeypatch.setattr(reels_dialog, "preview_boxes", boom)
    dialog.form.code_zoom.setValue(1.5)  # triggers a redraw; must not raise
    app.processEvents()


def test_preview_frame_is_read_once_per_clip(dialog, app, monkeypatch):
    from silence_remover.gui import reels_dialog
    calls = []
    real = reels_dialog.grab_frames
    monkeypatch.setattr(reels_dialog, "grab_frames",
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    dialog._find()
    wait_idle(dialog, app)
    dialog.table.selectRow(0)
    app.processEvents()
    for zoom in (1.2, 1.4, 1.6):
        dialog.form.code_zoom.setValue(zoom)
        app.processEvents()
    assert len(calls) == 1


def test_escape_stops_running_job(dialog, app):
    dialog._find()
    dialog.reject()
    assert dialog._worker is None
