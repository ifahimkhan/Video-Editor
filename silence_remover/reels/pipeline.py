"""Whole reels job: transcribe, pick the best moments, render each as a reel.

The transcript is cached as ``<name>.transcript.json`` in the output folder,
so trying other settings or rankers doesn't transcribe the video again.
A ``reels.json`` summary lists every clip with its title, hook and score.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..analysis import LoudnessProfile, analyze
from ..ffmpeg_tools import probe
from ..render import RenderCancelled
from ..segments import DetectionSettings, Segment, detect
from ..transcript import DEFAULT_MODEL, SpeechTranscript
from .analyze import resolve_webcam
from .captions import clip_keep
from .export import render_reel
from .gemini_ranker import GeminiClient, GeminiUnavailable, JsonGenerator, rank_with_gemini
from .highlights import Highlight, pick_local
from .settings import ReelSettings
from .transcript_cache import cached_transcript, write_json

log = logging.getLogger(__name__)

RANKERS = ("local", "gemini")
SUMMARY_NAME = "reels.json"
# Pauses cut inside a reel: tighter than the main tool's defaults, short
# videos feel slow with long pauses.
REEL_SILENCE = DetectionSettings(threshold_db=-35.0, min_silence_ms=350, padding_ms=120)
# Progress shares: transcript + picking, then rendering every clip.
PICK_SHARE = 0.4

Progress = Callable[[float], None]


@dataclass(frozen=True)
class ReelResult:
    output: Path
    highlight: Highlight
    length_s: float  # final length (shorter than the highlight when pauses are cut)


@dataclass(frozen=True)
class ReelsRun:
    reels: tuple[ReelResult, ...]
    summary_path: Path
    ranker: str


@dataclass(frozen=True)
class Prepared:
    """Everything needed to render reels without transcribing or picking again."""

    speech: SpeechTranscript
    profile: LoudnessProfile
    picks: list[Highlight]
    ranker: str


def find_highlights(
    input_path: str | Path,
    settings: ReelSettings,
    ranker: str = "local",
    model_size: str = DEFAULT_MODEL,
    cache_dir: str | Path | None = None,
    gemini_model: str | None = None,
    client: JsonGenerator | None = None,
    on_progress: Progress | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> list[Highlight]:
    """The best `settings.clip_count` moments of the video, in time order."""
    return prepare_reels(input_path, settings, ranker, model_size, cache_dir, gemini_model,
                         client, on_progress, is_cancelled).picks


def create_reels(
    input_path: str | Path,
    out_dir: str | Path,
    settings: ReelSettings,
    ranker: str = "local",
    model_size: str = DEFAULT_MODEL,
    gemini_model: str | None = None,
    client: JsonGenerator | None = None,
    on_progress: Progress | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> ReelsRun:
    """Pick the best moments of `input_path` and save each as a reel in `out_dir`."""
    report = on_progress or (lambda _f: None)
    prep = prepare_reels(input_path, settings, ranker, model_size, out_dir, gemini_model,
                         client, lambda f: report(PICK_SHARE * f), is_cancelled)
    return render_reels(input_path, out_dir, settings, prep,
                        on_progress=lambda f: report(PICK_SHARE + f * (1 - PICK_SHARE)),
                        is_cancelled=is_cancelled)


def prepare_reels(
    input_path: str | Path,
    settings: ReelSettings,
    ranker: str = "local",
    model_size: str = DEFAULT_MODEL,
    cache_dir: str | Path | None = None,
    gemini_model: str | None = None,
    client: JsonGenerator | None = None,
    on_progress: Progress | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> Prepared:
    """Transcribe (cached), measure loudness and pick the clips."""
    return _prepare(Path(input_path), settings, ranker, model_size, cache_dir, gemini_model,
                    client, on_progress, is_cancelled)


def render_reels(
    input_path: str | Path,
    out_dir: str | Path,
    settings: ReelSettings,
    prepared: Prepared,
    picks: list[Highlight] | None = None,
    on_progress: Progress | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> ReelsRun:
    """Render `picks` (default: all prepared picks) as reels in `out_dir`."""
    input_path, out_dir = Path(input_path), Path(out_dir)
    picks = prepared.picks if picks is None else picks
    if not picks:
        raise ValueError(
            f"No part of the video fits a {settings.min_length_s}-{settings.max_length_s} s "
            "clip of whole sentences. Try a shorter minimum length.")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = on_progress or (lambda _f: None)
    if settings.layout.tracked and settings.webcam is None:
        # Find the webcam once for all clips instead of once per clip.
        webcam = resolve_webcam(input_path, probe(input_path), settings)
        settings = dataclasses.replace(settings, webcam=webcam)
    keep_all = detect(prepared.profile, REEL_SILENCE).keep if settings.remove_silence else None

    results = []
    share = 1 / len(picks)
    for i, pick in enumerate(picks):
        keep = clip_keep(keep_all, pick.start_s, pick.end_s) if keep_all is not None else None
        if keep == ():
            log.warning("Skipping %.1f-%.1f s: it is all silence.", pick.start_s, pick.end_s)
            continue
        out = out_dir / f"{input_path.stem}_reel{len(results) + 1:02d}.mp4"
        base = i * share
        render_reel(input_path, out, pick.start_s, pick.end_s, settings,
                    on_progress=lambda f, base=base: report(base + f * share),
                    is_cancelled=is_cancelled, words=prepared.speech.words, title=pick.title,
                    keep=keep)
        results.append(ReelResult(out, pick, _kept_length(pick, keep)))

    summary = _write_summary(out_dir, input_path, results, prepared.ranker)
    report(1.0)
    return ReelsRun(tuple(results), summary, prepared.ranker)


def _prepare(input_path: Path, settings: ReelSettings, ranker: str, model_size: str,
             cache_dir: str | Path | None, gemini_model: str | None,
             client: JsonGenerator | None, on_progress: Progress | None,
             is_cancelled: Callable[[], bool] | None) -> Prepared:
    if ranker not in RANKERS:
        raise ValueError(f"Ranker must be one of {', '.join(RANKERS)}.")
    report = on_progress or (lambda _f: None)
    speech = cached_transcript(input_path, model_size, cache_dir,
                               lambda f: report(0.9 * f), is_cancelled)
    _check(is_cancelled)
    duration_s = probe(input_path).duration_s
    profile = analyze(input_path, use_vad=False, is_cancelled=is_cancelled)
    _check(is_cancelled)
    captions = [c for c in speech.captions if c.end_s > c.start_s]
    if ranker == "gemini" and client is None:
        try:
            client = GeminiClient(model=gemini_model)
        except GeminiUnavailable as exc:
            log.warning("%s Using the local scorer instead.", exc)
    if ranker == "gemini" and client is not None:
        picks = rank_with_gemini(captions, settings, duration_s, client, profile)
    else:
        picks = pick_local(captions, settings, duration_s, profile)
    _check(is_cancelled)
    report(1.0)
    return Prepared(speech, profile, picks, ranker)


def _kept_length(pick: Highlight, keep: tuple[Segment, ...] | None) -> float:
    return sum(s.length_ms for s in keep) / 1000 if keep is not None else pick.length_s


def _check(is_cancelled: Callable[[], bool] | None) -> None:
    if is_cancelled is not None and is_cancelled():
        raise RenderCancelled()


def _write_summary(out_dir: Path, input_path: Path, results: list[ReelResult],
                   ranker: str) -> Path:
    clips = [{
        "file": r.output.name,
        "start_s": round(r.highlight.start_s, 3),
        "end_s": round(r.highlight.end_s, 3),
        "length_s": round(r.length_s, 3),
        "title": r.highlight.title,
        "hook": r.highlight.hook,
        "score": round(r.highlight.score, 2),
        "reason": r.highlight.reason,
    } for r in results]
    path = out_dir / SUMMARY_NAME
    write_json(path, {"source": input_path.name, "ranker": ranker, "clips": clips})
    return path
