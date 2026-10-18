"""Tests for picking highlight clips from a transcript (local scorer)."""

import numpy as np
import pytest

from silence_remover.analysis import FRAME_MS, LoudnessProfile
from silence_remover.reels.highlights import (
    Highlight,
    candidate_windows,
    clip_from_sentences,
    pick_local,
    score_window,
    select_non_overlapping,
)
from silence_remover.reels.settings import ReelSettings
from silence_remover.transcript import Caption

SETTINGS = ReelSettings(min_length_s=30, max_length_s=60, clip_count=3)


def sentences(texts: list[str], length_s: float = 8.0, gap_s: float = 0.5) -> list[Caption]:
    out, t = [], 0.0
    for text in texts:
        out.append(Caption(t, t + length_s, text))
        t += length_s + gap_s
    return out


FILLER = [f"This is sentence number {i} about the code." for i in range(40)]


def test_candidates_respect_length_limits_and_sentence_bounds():
    caps = sentences(FILLER)
    windows = candidate_windows(caps, SETTINGS, duration_s=400)
    assert windows
    for first, last in windows:
        clip = clip_from_sentences(caps, first, last, duration_s=400)
        assert 30 <= clip.end_s - clip.start_s <= 60
        assert clip.start_s <= caps[first].start_s and clip.end_s >= caps[last].end_s


def test_clip_padding_stays_inside_video():
    caps = sentences(["Hello there."] * 5)
    clip = clip_from_sentences(caps, 0, 4, duration_s=caps[-1].end_s)
    assert clip.start_s == 0 and clip.end_s == caps[-1].end_s


def test_long_single_sentence_cannot_fit():
    caps = [Caption(0, 90, "One very long rambling sentence.")]
    assert candidate_windows(caps, SETTINGS, duration_s=90) == []


def test_hook_and_complete_ending_score_higher():
    plain = sentences(["and then we do the thing", "it is like that", "more of it",
                       "and so on", "continuing"])
    hooked = sentences(["Here's the mistake most people make with async code?",
                        "You await inside a loop.", "That runs one call at a time.",
                        "Use gather instead.", "Now it is five times faster."])
    assert score_window(hooked, 0, 4, None) > score_window(plain, 0, 4, None)


def test_gaps_lower_the_score():
    tight = sentences(["Here is a tip."] * 5, gap_s=0.2)
    loose = sentences(["Here is a tip."] * 5, length_s=4.0, gap_s=4.4)
    assert score_window(tight, 0, 4, None) > score_window(loose, 0, 4, None)


def test_louder_section_scores_higher():
    caps = sentences(["Here is a tip."] * 10)
    frames = int(caps[-1].end_s * 1000 / FRAME_MS) + 1
    db = np.full(frames, -30.0, np.float32)
    db[: frames // 2] = -18.0  # first half is louder
    profile = LoudnessProfile(db=db, frame_ms=FRAME_MS, duration_ms=frames * FRAME_MS)
    assert score_window(caps, 0, 3, profile) > score_window(caps, 6, 9, profile)


def test_select_non_overlapping_takes_best_first():
    picks = [Highlight(0, 40, 9.0), Highlight(30, 70, 8.0), Highlight(80, 120, 7.0)]
    chosen = select_non_overlapping(picks, 3)
    assert [(h.start_s, h.end_s) for h in chosen] == [(0, 40), (80, 120)]


def test_pick_local_returns_requested_count_in_time_order():
    texts = FILLER[:]
    texts[20] = "Why does this crash? Here's the secret."
    caps = sentences(texts)
    picks = pick_local(caps, SETTINGS, duration_s=caps[-1].end_s)
    assert len(picks) == 3
    starts = [p.start_s for p in picks]
    assert starts == sorted(starts)
    for a, b in zip(picks, picks[1:], strict=False):
        assert a.end_s <= b.start_s
    # The hook sentence starts one of the picks.
    assert any(abs(p.start_s - caps[20].start_s) < 0.5 for p in picks)
    assert all(p.title for p in picks)


def test_pick_local_with_short_transcript_returns_fewer():
    caps = sentences(FILLER[:5])  # ~42 s: room for one clip only
    picks = pick_local(caps, SETTINGS, duration_s=caps[-1].end_s)
    assert len(picks) == 1


def test_pick_local_with_no_speech_is_empty():
    assert pick_local([], SETTINGS, duration_s=100) == []


def test_highlight_rejects_bad_range():
    with pytest.raises(ValueError):
        Highlight(10, 5, 1.0)


def test_overlapping_captions_never_cut_words():
    caps = [Caption(0, 12, "First part."), Caption(11, 24, "Second part."),
            Caption(23, 40, "Third part.")]
    clip = clip_from_sentences(caps, 1, 2, duration_s=40)
    assert clip.start_s <= 11 and clip.end_s >= 40 - 1e-9
