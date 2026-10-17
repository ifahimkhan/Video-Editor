"""Whole reels job: transcribe, pick the best moments, render each as a reel.

The transcript is cached as ``<name>.transcript.json`` in the output folder,
so trying other settings or rankers doesn't transcribe the video again.
A ``reels.json`` summary lists every clip with its title, hook and score.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..analysis import analyze
from ..ffmpeg_tools import probe
from ..render import RenderCancelled
from ..transcript import DEFAULT_MODEL, Caption, SpeechTranscript, speech_captions
from .analyze import resolve_webcam
from .export import render_reel
from .gemini_ranker import GeminiClient, GeminiUnavailable, JsonGenerator, rank_with_gemini
from .highlights import Highlight, pick_local
from .settings import ReelSettings

log = logging.getLogger(__name__)

RANKERS = ("local", "gemini")
SUMMARY_NAME = "reels.json"
CACHE_VERSION = 1
# Progress shares: transcript + picking, then rendering every clip.
PICK_SHARE = 0.4

Progress = Callable[[float], None]


@dataclass(frozen=True)
class ReelResult:
    output: Path
    highlight: Highlight


@dataclass(frozen=True)
class ReelsRun:
    reels: tuple[ReelResult, ...]
    summary_path: Path
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
    if ranker not in RANKERS:
        raise ValueError(f"Ranker must be one of {', '.join(RANKERS)}.")
    input_path = Path(input_path)
    report = on_progress or (lambda _f: None)
    speech = _cached_transcript(input_path, model_size, cache_dir,
                                lambda f: report(0.9 * f), is_cancelled)
    _check(is_cancelled)
    duration_s = probe(input_path).duration_s
    profile = analyze(input_path, use_vad=False, is_cancelled=is_cancelled)
    _check(is_cancelled)
    captions = [c for c in speech.captions if c.end_s > c.start_s]
    if ranker == "gemini":
        if client is None:
            try:
                client = GeminiClient(model=gemini_model)
            except GeminiUnavailable as exc:
                log.warning("%s Using the local scorer instead.", exc)
        if client is not None:
            picks = rank_with_gemini(captions, settings, duration_s, client, profile)
            _check(is_cancelled)
            report(1.0)
            return picks
    picks = pick_local(captions, settings, duration_s, profile)
    report(1.0)
    return picks


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
    input_path, out_dir = Path(input_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = on_progress or (lambda _f: None)
    picks = find_highlights(input_path, settings, ranker, model_size, out_dir, gemini_model,
                            client, lambda f: report(PICK_SHARE * f), is_cancelled)
    if not picks:
        raise ValueError(
            f"No part of the video fits a {settings.min_length_s}-{settings.max_length_s} s "
            "clip of whole sentences. Try a shorter minimum length.")

    if settings.layout.tracked and settings.webcam is None:
        # Find the webcam once for all clips instead of once per clip.
        webcam = resolve_webcam(input_path, probe(input_path), settings)
        settings = dataclasses.replace(settings, webcam=webcam)

    results = []
    share = (1 - PICK_SHARE) / len(picks)
    for i, pick in enumerate(picks):
        out = out_dir / f"{input_path.stem}_reel{i + 1:02d}.mp4"
        base = PICK_SHARE + i * share
        render_reel(input_path, out, pick.start_s, pick.end_s, settings,
                    on_progress=lambda f, base=base: report(base + f * share),
                    is_cancelled=is_cancelled)
        results.append(ReelResult(out, pick))

    summary = _write_summary(out_dir, input_path, results, ranker)
    report(1.0)
    return ReelsRun(tuple(results), summary, ranker)


def _cached_transcript(input_path: Path, model_size: str, cache_dir: str | Path | None,
                       on_progress: Progress, is_cancelled) -> SpeechTranscript:
    cache = Path(cache_dir) / f"{input_path.stem}.transcript.json" if cache_dir else None
    key = _cache_key(input_path, model_size)
    if cache is not None and cache.is_file():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("key") == key:
                on_progress(1.0)
                return SpeechTranscript(
                    tuple(Caption(float(s), float(e), str(t)) for s, e, t in data["captions"]),
                    str(data["language"]), float(data["language_probability"]))
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            log.info("Ignoring unreadable transcript cache %s", cache)
    speech = speech_captions(input_path, model_size=model_size, on_progress=on_progress,
                             is_cancelled=is_cancelled)
    if cache is not None:
        payload = {"key": key, "language": speech.language,
                   "language_probability": speech.language_probability,
                   "captions": [[c.start_s, c.end_s, c.text] for c in speech.captions]}
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            _write_json(cache, payload)
        except OSError as exc:  # the transcript itself is still good
            log.warning("Could not save the transcript cache %s: %s", cache, exc)
    return speech


def _check(is_cancelled: Callable[[], bool] | None) -> None:
    if is_cancelled is not None and is_cancelled():
        raise RenderCancelled()


def _cache_key(input_path: Path, model_size: str) -> dict:
    stat = input_path.stat()
    return {"version": CACHE_VERSION, "name": input_path.name, "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "model": model_size}


def _write_summary(out_dir: Path, input_path: Path, results: list[ReelResult],
                   ranker: str) -> Path:
    clips = [{
        "file": r.output.name,
        "start_s": round(r.highlight.start_s, 3),
        "end_s": round(r.highlight.end_s, 3),
        "length_s": round(r.highlight.length_s, 3),
        "title": r.highlight.title,
        "hook": r.highlight.hook,
        "score": round(r.highlight.score, 2),
        "reason": r.highlight.reason,
    } for r in results]
    path = out_dir / SUMMARY_NAME
    _write_json(path, {"source": input_path.name, "ranker": ranker, "clips": clips})
    return path


def _write_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(f"{path.name}.part")
    try:
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
