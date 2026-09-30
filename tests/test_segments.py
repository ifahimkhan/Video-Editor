import pytest

from silence_remover.segments import (
    DetectionSettings,
    Segment,
    detect,
    invert,
    pad_and_merge,
)

from .conftest import make_profile

LOUD, QUIET = -10.0, -80.0


def settings(threshold=-35, min_silence=500, padding=0):
    return DetectionSettings(threshold, min_silence, padding)


def test_cuts_long_silence_and_keeps_speech():
    profile = make_profile([(1000, LOUD), (2000, QUIET), (1000, LOUD)])
    result = detect(profile, settings())
    assert result.keep == (Segment(0, 1000), Segment(3000, 4000))
    assert result.removed_ms == 2000


def test_short_pause_below_min_silence_is_kept():
    profile = make_profile([(1000, LOUD), (300, QUIET), (1000, LOUD)])
    result = detect(profile, settings(min_silence=500))
    assert result.keep == (Segment(0, 2300),)
    assert result.cuts == ()


def test_padding_shrinks_the_cut_on_both_sides():
    profile = make_profile([(1000, LOUD), (2000, QUIET), (1000, LOUD)])
    result = detect(profile, settings(padding=200))
    assert result.keep == (Segment(0, 1200), Segment(2800, 4000))


def test_padding_larger_than_gap_merges_segments():
    profile = make_profile([(1000, LOUD), (600, QUIET), (1000, LOUD)])
    result = detect(profile, settings(padding=400))
    assert result.keep == (Segment(0, 2600),)


def test_leading_and_trailing_silence_removed():
    profile = make_profile([(1000, QUIET), (500, LOUD), (1000, QUIET)])
    result = detect(profile, settings(padding=100))
    assert result.keep == (Segment(900, 1600),)
    assert result.cuts == (Segment(0, 900), Segment(1600, 2500))


def test_threshold_decides_what_is_silence():
    profile = make_profile([(1000, LOUD), (1000, -30.0), (1000, LOUD)])
    assert detect(profile, settings(threshold=-35)).cuts == ()
    assert detect(profile, settings(threshold=-25)).cuts == (Segment(1000, 2000),)


def test_all_silent_keeps_nothing():
    profile = make_profile([(3000, QUIET)])
    result = detect(profile, settings())
    assert result.keep == ()
    assert result.kept_ms == 0


def test_all_loud_keeps_everything():
    profile = make_profile([(3000, LOUD)])
    assert detect(profile, settings()).keep == (Segment(0, 3000),)


def test_invert_and_pad_helpers():
    segs = (Segment(100, 200), Segment(500, 600))
    assert invert(segs, 1000) == (Segment(0, 100), Segment(200, 500), Segment(600, 1000))
    assert invert((), 50) == (Segment(0, 50),)
    assert pad_and_merge(segs, 50, 620) == (Segment(50, 250), Segment(450, 620))


@pytest.mark.parametrize("kwargs", [
    {"threshold_db": 0}, {"threshold_db": -100},
    {"min_silence_ms": 10}, {"padding_ms": -1}, {"padding_ms": 5000},
])
def test_settings_validation(kwargs):
    with pytest.raises(ValueError):
        DetectionSettings(**kwargs)
