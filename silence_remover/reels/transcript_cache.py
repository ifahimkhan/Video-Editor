"""Cache a video's transcript (with word timings) as JSON next to the reels.

Transcribing a long talk is the slowest step, so it is done once and reused
while the file, its size/modification time and the speech model match.
A broken or foreign cache file is ignored, and a failed write only logs.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Callable

from ..transcript import Caption, SpeechTranscript, Word, speech_captions

log = logging.getLogger(__name__)

CACHE_VERSION = 2  # 2: word timings included


def cache_path(cache_dir: str | Path, input_path: Path) -> Path:
    return Path(cache_dir) / f"{input_path.stem}.transcript.json"


def cached_transcript(input_path: Path, model_size: str, cache_dir: str | Path | None,
                      on_progress: Callable[[float], None],
                      is_cancelled: Callable[[], bool] | None) -> SpeechTranscript:
    """The transcript of `input_path` with word timings, from cache if possible."""
    cache = cache_path(cache_dir, input_path) if cache_dir else None
    key = _cache_key(input_path, model_size)
    if cache is not None:
        speech = _load(cache, key)
        if speech is not None:
            on_progress(1.0)
            return speech
    speech = speech_captions(input_path, model_size=model_size, on_progress=on_progress,
                             is_cancelled=is_cancelled, word_timestamps=True)
    if cache is not None:
        _save(cache, key, speech)
    return speech


def write_json(path: Path, payload: dict) -> None:
    """Write JSON atomically (never leaves a half-written file)."""
    tmp = path.with_name(f"{path.name}.part")
    try:
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _load(cache: Path, key: dict) -> SpeechTranscript | None:
    if not cache.is_file():
        return None
    try:
        data = json.loads(cache.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("key") != key:
            return None
        return SpeechTranscript(
            tuple(Caption(float(s), float(e), str(t)) for s, e, t in data["captions"]),
            str(data["language"]), float(data["language_probability"]),
            tuple(Word(float(s), float(e), str(t)) for s, e, t in data["words"]),
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        log.info("Ignoring unreadable transcript cache %s", cache)
        return None


def _save(cache: Path, key: dict, speech: SpeechTranscript) -> None:
    payload = {
        "key": key, "language": speech.language,
        "language_probability": speech.language_probability,
        "captions": [[c.start_s, c.end_s, c.text] for c in speech.captions],
        "words": [[w.start_s, w.end_s, w.text] for w in speech.words],
    }
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        write_json(cache, payload)
    except OSError as exc:  # the transcript itself is still good
        log.warning("Could not save the transcript cache %s: %s", cache, exc)


def _cache_key(input_path: Path, model_size: str) -> dict:
    stat = input_path.stat()
    return {"version": CACHE_VERSION, "name": input_path.name, "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "model": model_size}
