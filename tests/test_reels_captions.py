"""Tests for burned-in caption (ASS) generation and time remapping."""

import re

import pytest

from silence_remover.reels.captions import (
    ass_time,
    build_ass,
    caption_y,
    clip_keep,
    group_words,
    remap_words,
)
from silence_remover.reels.settings import CaptionStyle, Layout, ReelSettings
from silence_remover.segments import Segment
from silence_remover.transcript import Word


def words(spec: str, start: float = 0.0, step: float = 0.4) -> list[Word]:
    out, t = [], start
    for text in spec.split():
        out.append(Word(t, t + step - 0.05, text))
        t += step
    return out


def dialogues(ass: str) -> list[str]:
    return [line for line in ass.splitlines() if line.startswith("Dialogue:")]


def test_ass_time_format():
    assert ass_time(0) == "0:00:00.00"
    assert ass_time(3725.456) == "1:02:05.46"
    assert ass_time(-1) == "0:00:00.00"


def test_group_words_limits_count_and_breaks_on_sentence_end():
    groups = group_words(words("one two three four five six. seven eight"))
    assert [len(g) for g in groups] == [4, 2, 2]
    assert groups[1][-1].text == "six."


def test_group_words_breaks_on_long_pause():
    ws = words("hello there") + words("after pause", start=3.0)
    assert [len(g) for g in group_words(ws)] == [2, 2]


def test_group_words_limits_characters():
    groups = group_words(words("internationalization configuration done"))
    assert all(sum(len(w.text) + 1 for w in g) - 1 <= 22 or len(g) == 1 for g in groups)


def test_build_ass_highlights_each_word_in_turn():
    ass = build_ass(words("make it fast"), ReelSettings(captions=True), duration_s=5)
    lines = dialogues(ass)
    assert len(lines) == 3
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass
    for i, line in enumerate(lines):
        highlighted = re.findall(r"\{\\c&H[0-9A-F]+&\}(\w+)\{\\c", line)
        assert highlighted == [["MAKE", "IT", "FAST"][i]]


def test_word_events_are_continuous_within_a_line():
    ass = build_ass(words("make it fast"), ReelSettings(captions=True), duration_s=5)
    times = [line.split(",")[1:3] for line in dialogues(ass)]
    for (_, end), (start, _) in zip(times, times[1:], strict=False):
        assert end == start


def test_box_style_keeps_case_and_uses_opaque_box():
    ass = build_ass(words("Make it fast"),
                    ReelSettings(captions=True, caption_style=CaptionStyle.BOX), duration_s=5)
    assert "Make" in ass and "MAKE" not in ass
    style = next(line for line in ass.splitlines() if line.startswith("Style: Caption"))
    assert style.split(",")[15] == "3"  # BorderStyle 3 = opaque box


def test_text_is_escaped():
    ass = build_ass([Word(0, 1, "{\\b1}hack")], ReelSettings(captions=True), duration_s=2)
    line = dialogues(ass)[0]
    assert "{\\b1}" not in line and "HACK" in line


def test_title_overlay_event():
    ass = build_ass([], ReelSettings(title_overlay=True), duration_s=10, title="Async in 30s")
    lines = dialogues(ass)
    assert len(lines) == 1 and "Async in 30s" in lines[0] and ",Title," in lines[0]
    assert lines[0].split(",")[2] == "0:00:04.00"


def test_caption_position_follows_layout():
    split = ReelSettings(layout=Layout.SPLIT, split_ratio=0.6)
    assert caption_y(split) == 1152
    assert caption_y(ReelSettings(layout=Layout.SPEAKER)) == pytest.approx(1920 * 0.72, abs=1)


def test_remap_words_shifts_to_clip_time_and_drops_outside():
    ws = [Word(9.0, 9.5, "before"), Word(10.2, 10.6, "in"), Word(19.8, 20.5, "edge"),
          Word(25, 26, "after")]
    out = remap_words(ws, 10.0, 20.0)
    assert [w.text for w in out] == ["in", "edge"]
    assert out[0].start_s == pytest.approx(0.2)
    assert out[1].end_s == pytest.approx(10.0)  # clamped to the clip end


def test_remap_words_through_silence_cuts():
    # Clip-relative keep: 0-2 s and 4-6 s (2 s of silence removed).
    keep = (Segment(0, 2000), Segment(4000, 6000))
    ws = [Word(10.5, 11.0, "a"), Word(12.5, 13.0, "gone"), Word(14.5, 15.0, "b")]
    out = remap_words(ws, 10.0, 16.0, keep)
    assert [w.text for w in out] == ["a", "b"]
    assert out[1].start_s == pytest.approx(2.5)


def test_clip_keep_intersects_and_shifts():
    keep_full = (Segment(0, 12_000), Segment(14_000, 30_000))
    assert clip_keep(keep_full, 10.0, 20.0) == (Segment(0, 2000), Segment(4000, 10_000))


def test_clip_keep_without_cuts_is_none():
    assert clip_keep((Segment(0, 30_000),), 10.0, 20.0) is None


def test_clip_keep_all_silent_is_empty():
    assert clip_keep((Segment(0, 5000),), 10.0, 20.0) == ()


def test_sentence_end_inside_quotes_breaks_the_line():
    groups = group_words(words('he said "stop." then left'))
    assert groups[0][-1].text == '"stop."'


def test_bold_lines_are_shorter_than_other_styles():
    ws = words("alpha beta gamma delta")  # 22 chars with spaces
    bold = group_words(ws, CaptionStyle.BOLD)
    plain = group_words(ws, CaptionStyle.MINIMAL)
    assert len(bold) > len(plain)
