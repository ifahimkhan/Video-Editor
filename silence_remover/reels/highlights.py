"""Pick the best self-contained moments of a talk as reel clips. Pure logic.

A clip is always a run of whole transcript sentences, so it never starts or
ends mid-word. Every run between the length limits is scored on:

* speech density   - little dead air;
* hook             - opens with a question or a hook phrase ("here's", "mistake");
* complete ending  - finishes on . ! or ?;
* clean start      - doesn't open with "and", "so", "but"...;
* energy           - louder than the talk's typical level (when loudness is known);
* length fit       - near the middle of the allowed range.

The best non-overlapping runs win. `gemini_ranker` reuses the same clip and
selection helpers so both rankers obey the same rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..analysis import LoudnessProfile
from ..transcript import Caption
from .settings import ReelSettings

START_PAD_S = 0.2   # keep a breath before the first word...
END_PAD_S = 0.4     # ...and let the last one ring out
TITLE_MAX_CHARS = 60
HOOK_PHRASES = (
    "how", "why", "what if", "here's", "here is", "secret", "mistake", "never",
    "always", "stop", "trick", "tip", "best", "easiest", "fastest", "you need",
    "don't", "most people", "the problem", "did you know", "watch",
)
WEAK_STARTS = {"and", "so", "but", "because", "or", "then", "also", "which", "that", "um", "uh"}
WEIGHTS = {"density": 0.30, "hook": 0.20, "ending": 0.20, "start": 0.15,
           "energy": 0.10, "length": 0.05}
ENERGY_RANGE_DB = 12.0  # window loudness -6..+6 dB around the median maps to 0..1
_HOOK_RE = re.compile(r"\b(" + "|".join(re.escape(p) for p in HOOK_PHRASES) + r")\b")


@dataclass(frozen=True)
class Highlight:
    start_s: float
    end_s: float
    score: float          # 0..10
    title: str = ""
    hook: str = ""
    reason: str = ""
    first: int = -1       # sentence indices (inclusive), -1 when unknown
    last: int = -1

    def __post_init__(self) -> None:
        if self.end_s <= self.start_s:
            raise ValueError(f"Highlight must end after it starts ({self.start_s}..{self.end_s}).")

    @property
    def length_s(self) -> float:
        return self.end_s - self.start_s

    def overlaps(self, other: Highlight) -> bool:
        return self.start_s < other.end_s and other.start_s < self.end_s


def clip_from_sentences(captions: Sequence[Caption], first: int, last: int,
                        duration_s: float, score: float = 0.0) -> Highlight:
    """Clip covering sentences first..last, padded but not into the neighbours."""
    # Pad into the gaps around the run, never into a neighbour - but if
    # Whisper's segments overlap, never cut into this run's own words either.
    floor = min(captions[first - 1].end_s, captions[first].start_s) if first > 0 else 0.0
    ceiling = (max(captions[last + 1].start_s, captions[last].end_s)
               if last + 1 < len(captions) else duration_s)
    start = max(floor, 0.0, captions[first].start_s - START_PAD_S)
    end = min(ceiling, duration_s, captions[last].end_s + END_PAD_S)
    return Highlight(start, max(end, start + 1e-3), score, title=make_title(captions[first].text),
                     first=first, last=last)


def candidate_windows(captions: Sequence[Caption], settings: ReelSettings,
                      duration_s: float) -> list[tuple[int, int]]:
    """Every (first, last) sentence run whose clip fits the length limits."""
    windows = []
    for first in range(len(captions)):
        for last in range(first, len(captions)):
            length = clip_from_sentences(captions, first, last, duration_s).length_s
            if length > settings.max_length_s:
                break
            if length >= settings.min_length_s:
                windows.append((first, last))
    return windows


def score_window(captions: Sequence[Caption], first: int, last: int,
                 profile: LoudnessProfile | None,
                 settings: ReelSettings | None = None) -> float:
    """0..10 score for the clip made of sentences first..last."""
    settings = settings or ReelSettings()
    run = captions[first:last + 1]
    span = run[-1].end_s - run[0].start_s
    spoken = sum(c.end_s - c.start_s for c in run)
    opening, closing = _clean(run[0].text), _clean(run[-1].text)
    features = {
        "density": min(1.0, spoken / span) if span > 0 else 0.0,
        "hook": 1.0 if opening.endswith("?") or _HOOK_RE.search(opening.lower()) else 0.0,
        "ending": 1.0 if closing.endswith((".", "!", "?")) else 0.0,
        "start": 0.0 if _first_word(opening) in WEAK_STARTS else 1.0,
        "energy": _energy(profile, run[0].start_s, run[-1].end_s),
        "length": _length_fit(span, settings),
    }
    return 10.0 * sum(WEIGHTS[k] * v for k, v in features.items())


def select_non_overlapping(candidates: Sequence[Highlight], count: int) -> list[Highlight]:
    """Best-scoring clips that don't overlap, returned in time order."""
    chosen: list[Highlight] = []
    for clip in sorted(candidates, key=lambda h: -h.score):
        if len(chosen) == count:
            break
        if not any(clip.overlaps(c) for c in chosen):
            chosen.append(clip)
    return sorted(chosen, key=lambda h: h.start_s)


def score_candidates(captions: Sequence[Caption], settings: ReelSettings, duration_s: float,
                     profile: LoudnessProfile | None = None) -> list[Highlight]:
    """Every clip that fits the length limits, with its local score."""
    return [
        clip_from_sentences(captions, first, last, duration_s,
                            score=score_window(captions, first, last, profile, settings))
        for first, last in candidate_windows(captions, settings, duration_s)
    ]


def pick_local(captions: Sequence[Caption], settings: ReelSettings, duration_s: float,
               profile: LoudnessProfile | None = None) -> list[Highlight]:
    """Top `settings.clip_count` clips by the local score."""
    return select_non_overlapping(score_candidates(captions, settings, duration_s, profile),
                                  settings.clip_count)


def make_title(text: str) -> str:
    """Short title from a sentence, cut at a word boundary."""
    text = _clean(text)
    if len(text) <= TITLE_MAX_CHARS:
        return text
    cut = text[:TITLE_MAX_CHARS].rsplit(" ", 1)[0]
    return cut.rstrip(",;:") + "…"


def _clean(text: str) -> str:
    return " ".join(text.split())


def _first_word(text: str) -> str:
    words = re.findall(r"[a-z']+", text.lower())
    return words[0] if words else ""


def _energy(profile: LoudnessProfile | None, start_s: float, end_s: float) -> float:
    if profile is None or profile.db.size == 0:
        return 0.5
    a = int(start_s * 1000 / profile.frame_ms)
    b = max(a + 1, int(end_s * 1000 / profile.frame_ms))
    window = profile.db[a:b]
    if window.size == 0:
        return 0.5
    relative = float(np.mean(window) - np.median(profile.db))
    return float(np.clip(relative / ENERGY_RANGE_DB + 0.5, 0.0, 1.0))


def _length_fit(length_s: float, settings: ReelSettings) -> float:
    spread = max(1.0, settings.max_length_s - settings.min_length_s)
    target = (settings.min_length_s + settings.max_length_s) / 2
    return float(np.clip(1.0 - abs(length_s - target) / spread, 0.0, 1.0))
