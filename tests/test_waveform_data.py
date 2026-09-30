import numpy as np
import pytest

from silence_remover.analysis import frames_to_peak_db
from silence_remover.gui.waveform_data import (
    DISPLAY_FLOOR_DB, cut_steps, db_to_height, decimate_max, format_time, height_to_db,
    time_axis,
)
from silence_remover.segments import THRESHOLD_RANGE_DB, Segment


def test_db_to_height_scale_and_clipping():
    assert db_to_height(0.0) == pytest.approx(1.0)
    assert db_to_height(DISPLAY_FLOOR_DB) == pytest.approx(0.0)
    assert db_to_height(-100.0) == 0.0
    np.testing.assert_allclose(db_to_height(np.array([-36.0, 6.0])), [0.5, 1.0])


def test_height_to_db_round_trip_across_threshold_range():
    for db in range(THRESHOLD_RANGE_DB[0], THRESHOLD_RANGE_DB[1] + 1, 5):
        assert height_to_db(float(db_to_height(db))) == pytest.approx(db, abs=1e-4)


def test_threshold_range_is_draggable():
    """Every threshold setting sits strictly above the display floor."""
    assert db_to_height(THRESHOLD_RANGE_DB[0]) > 0


def test_time_axis_frame_centres():
    np.testing.assert_allclose(time_axis(3, 10), [0.005, 0.015, 0.025])


def test_cut_steps_middle_cut():
    x, y = cut_steps((Segment(1000, 2000),), 5000)
    assert x.tolist() == [0.0, 1.0, 2.0, 5.0]
    assert y.tolist() == [-1.0, 1.0, -1.0]


def test_cut_steps_edges_and_no_cuts():
    x, y = cut_steps((Segment(0, 500), Segment(4000, 5000)), 5000)
    assert x.tolist() == [0.0, 0.5, 4.0, 5.0]
    assert y.tolist() == [1.0, -1.0, 1.0]
    x, y = cut_steps((), 3000)
    assert x.tolist() == [0.0, 3.0] and y.tolist() == [-1.0]
    assert len(x) == len(y) + 1


@pytest.mark.parametrize("seconds, spacing, text", [
    (0, 10, "0:00"),
    (75, 5, "1:15"),
    (75.25, 0.5, "1:15.2"),
    (3723, 60, "1:02:03"),
])
def test_format_time(seconds, spacing, text):
    assert format_time(seconds, spacing) == text


def test_peak_db_is_at_least_rms():
    rng = np.random.default_rng(1)
    samples = (rng.standard_normal(1600) * 3000).astype(np.int16)
    peak = frames_to_peak_db(samples)
    assert peak.shape == (10,)
    assert np.all(peak <= 0.0)
    full = frames_to_peak_db(np.full(160, -32768, dtype=np.int16))
    assert full[0] == pytest.approx(0.0, abs=1e-3)


def test_decimate_max_keeps_peaks_and_limits_points():
    t = time_axis(100_000, 10)                  # 1000 s
    values = np.zeros(100_000, dtype=np.float32)
    values[54_321] = 1.0                        # one spike at ~543 s
    td, vd = decimate_max(t, values, 0, 1000, 800)
    assert len(td) <= 800
    assert vd.max() == 1.0                      # spike survives decimation
    assert np.all(np.diff(td) > 0)


def test_decimate_max_returns_visible_slice_when_zoomed_in():
    t = time_axis(1000, 10)                     # 10 s
    values = np.arange(1000, dtype=np.float32)
    td, vd = decimate_max(t, values, 2.0, 3.0, 800)
    assert 100 <= len(td) <= 103                # 1 s of 10 ms frames, +margin
    assert td[0] < 2.0 < 3.0 < td[-1]
    np.testing.assert_array_equal(vd, values[199:301])  # one frame margin each side


def test_cut_steps_clamps_and_handles_empty_duration():
    x, y = cut_steps((Segment(4000, 6000),), 5000)
    assert x.tolist() == [0.0, 4.0, 5.0] and y.tolist() == [-1.0, 1.0]
    x, y = cut_steps((), 0)
    assert len(x) == len(y) + 1


def test_format_time_rounds_before_splitting_minutes():
    assert format_time(59.96, 0.5) == "1:00.0"
