"""Reels for time ranges the user picks by hand (moments the AI may miss).

A manual clip doesn't need Find clips: `prepare_manual` fetches only what the
chosen options need - the transcript (cached) for captions, the loudness
profile for pause cuts - and returns a `Prepared` with no picks of its own.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..analysis import analyze
from ..transcript import DEFAULT_MODEL, SpeechTranscript
from .highlights import Highlight
from .pipeline import Prepared
from .settings import LENGTH_RANGE_S, ReelSettings
from .timecode import format_time
from .transcript_cache import cached_transcript

MIN_MANUAL_S = 1.0
MAX_MANUAL_S = float(LENGTH_RANGE_S[1])
# ffprobe's duration and the last frame's timestamp can differ slightly.
END_TOLERANCE_S = 0.05


def manual_clip(start_s: float, end_s: float, duration_s: float,
                title: str = "") -> Highlight:
    """A user-chosen range as a clip; raises ValueError with a readable reason."""
    if start_s < 0:
        raise ValueError("The start can't be before the beginning of the video.")
    if end_s <= start_s:
        raise ValueError("The end must be after the start.")
    if end_s > duration_s + END_TOLERANCE_S:
        raise ValueError(f"The end ({format_time(end_s)}) is past the end of the video "
                         f"({format_time(duration_s)}).")
    length = end_s - start_s
    if length < MIN_MANUAL_S:
        raise ValueError(f"A clip must be at least {MIN_MANUAL_S:.0f} s long.")
    if length > MAX_MANUAL_S:
        raise ValueError(f"A clip can be at most {MAX_MANUAL_S:.0f} s long "
                         f"(this one is {length:.0f} s).")
    title = " ".join(title.split()) or f"Clip {format_time(start_s)}-{format_time(end_s)}"
    return Highlight(start_s, min(end_s, duration_s), score=0.0, title=title,
                     reason="Added by you", manual=True)


def prepare_manual(
    input_path: str | Path,
    settings: ReelSettings,
    model_size: str = DEFAULT_MODEL,
    cache_dir: str | Path | None = None,
    on_progress: Callable[[float], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> Prepared:
    """What rendering needs for manual clips when Find clips wasn't run."""
    input_path = Path(input_path)
    report = on_progress or (lambda _f: None)
    if settings.captions:
        speech = cached_transcript(input_path, model_size, cache_dir,
                                   lambda f: report(0.9 * f), is_cancelled)
    else:
        speech = SpeechTranscript((), "", 0.0)
    profile = (analyze(input_path, use_vad=False, is_cancelled=is_cancelled)
               if settings.remove_silence else None)
    report(1.0)
    return Prepared(speech, profile, [], "manual")
