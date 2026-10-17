"""Tests for typed timestamps (manual reel clips)."""

import pytest

from silence_remover.reels.timecode import format_time, parse_time


@pytest.mark.parametrize("text, seconds", [
    ("85", 85.0), ("85.5", 85.5), ("1:25", 85.0), ("01:25.5", 85.5),
    ("1:02:05", 3725.0), (" 2:00 ", 120.0), ("0:00", 0.0), ("1,5", 1.5),
])
def test_parse_time_accepts_common_forms(text, seconds):
    assert parse_time(text) == pytest.approx(seconds)


@pytest.mark.parametrize("text", ["", "abc", "1:2:3:4", "1:75", "-5", "1:-2", "1::2", "nan"])
def test_parse_time_rejects_bad_input(text):
    with pytest.raises(ValueError):
        parse_time(text)


def test_format_time_round_trips():
    assert format_time(85.0) == "1:25"
    assert format_time(3725.5) == "1:02:05.5"
    assert parse_time(format_time(3725.5)) == pytest.approx(3725.5)
