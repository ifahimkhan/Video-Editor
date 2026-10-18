"""Rank highlight clips with Google Gemini (optional, needs GEMINI_API_KEY).

The transcript is sent as numbered sentences; Gemini answers with JSON
(structured output) naming each clip as a *range of sentence indices*, plus
a title, a hook line and a score. Indices instead of seconds mean a clip can
only start and end on sentence boundaries. Every answer is still checked:
too-long clips are trimmed by whole sentences, invalid or overlapping ones
are dropped, and missing slots are filled by the local scorer. Any API
problem falls back to the local scorer entirely.

The REST API is called with the standard library (no SDK dependency); on
Windows `urllib` uses the system certificate store, so it also works behind
corporate TLS proxies.
"""

from __future__ import annotations

import http.client
import json
import logging
import math
import os
import re
import urllib.error
import urllib.request
from typing import Any, Protocol, Sequence

from ..analysis import LoudnessProfile
from ..transcript import Caption
from .highlights import (
    Highlight,
    clip_from_sentences,
    make_title,
    pick_local,
    score_candidates,
    select_non_overlapping,
)
from .settings import ReelSettings

log = logging.getLogger(__name__)

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
KEY_ENV_VAR = "GEMINI_API_KEY"
MODEL_ENV_VAR = "GEMINI_MODEL"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
TIMEOUT_S = 120
TEMPERATURE = 0.4
HOOK_MAX_CHARS = 200
MAX_REPLY_BYTES = 4_000_000  # a real answer is a few KB
_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")

_CLIP_FIELDS = ("start_index", "end_index", "title", "hook", "score", "reason")
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "clips": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "start_index": {"type": "INTEGER"},
                    "end_index": {"type": "INTEGER"},
                    "title": {"type": "STRING"},
                    "hook": {"type": "STRING"},
                    "score": {"type": "NUMBER"},
                    "reason": {"type": "STRING"},
                },
                "required": list(_CLIP_FIELDS),
                "propertyOrdering": list(_CLIP_FIELDS),
            },
        },
    },
    "required": ["clips"],
}


class GeminiUnavailable(RuntimeError):
    """No API key, network/API error, or an unusable reply."""


class JsonGenerator(Protocol):
    def generate_json(self, prompt: str, schema: dict[str, Any]) -> Any: ...


class GeminiClient:
    """Minimal Gemini `generateContent` client returning parsed JSON."""

    def __init__(self, api_key: str | None = None, model: str | None = None,
                 timeout_s: float = TIMEOUT_S) -> None:
        self._key = (api_key or os.environ.get(KEY_ENV_VAR, "")).strip()
        if not self._key:
            raise GeminiUnavailable(
                f"Set the {KEY_ENV_VAR} environment variable to use Gemini "
                "(get a key at https://aistudio.google.com/apikey).")
        if not self._key.isprintable() or any(ch.isspace() for ch in self._key):
            raise GeminiUnavailable(
                f"The Gemini API key contains spaces or control characters; check {KEY_ENV_VAR}.")
        self.model = model or os.environ.get(MODEL_ENV_VAR) or DEFAULT_GEMINI_MODEL
        if not _MODEL_NAME_RE.match(self.model):
            raise GeminiUnavailable(f"Invalid Gemini model name: {self.model!r}")
        self._timeout_s = timeout_s

    def generate_json(self, prompt: str, schema: dict[str, Any]) -> Any:
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": schema,
                "temperature": TEMPERATURE,
            },
        }
        request = urllib.request.Request(
            f"{API_ROOT}/models/{self.model}:generateContent",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": self._key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                raw = response.read(MAX_REPLY_BYTES + 1)
        except urllib.error.HTTPError as exc:
            raise GeminiUnavailable(
                f"Gemini request failed (HTTP {exc.code}): {self._error_message(exc)}") from None
        except (urllib.error.URLError, http.client.HTTPException, TimeoutError,
                OSError) as exc:
            raise GeminiUnavailable(f"Could not reach Gemini: {exc}") from None
        if len(raw) > MAX_REPLY_BYTES:
            raise GeminiUnavailable("Gemini's reply was too large.")
        try:
            data = json.loads(raw.decode("utf-8"))
        except ValueError:  # bad JSON or bad UTF-8
            raise GeminiUnavailable("Gemini sent a reply that is not JSON.") from None
        return _reply_json(data)

    def _error_message(self, exc: urllib.error.HTTPError) -> str:
        try:
            message = json.loads(exc.read().decode("utf-8"))["error"]["message"]
        except (ValueError, KeyError, TypeError, AttributeError):
            message = exc.reason
        return str(message).replace(self._key, "***")


def rank_with_gemini(captions: Sequence[Caption], settings: ReelSettings, duration_s: float,
                     client: JsonGenerator | None = None,
                     profile: LoudnessProfile | None = None) -> list[Highlight]:
    """Gemini's picks (checked), topped up by the local scorer; local on failure."""
    if not captions:
        return []
    try:
        client = client or GeminiClient()
        reply = client.generate_json(build_prompt(captions, settings), RESPONSE_SCHEMA)
    except GeminiUnavailable as exc:
        log.warning("Gemini unavailable, using the local scorer instead: %s", exc)
        return pick_local(captions, settings, duration_s, profile)

    chosen = select_non_overlapping(parse_picks(reply, captions, settings, duration_s),
                                    settings.clip_count)
    if len(chosen) < settings.clip_count:
        local = score_candidates(captions, settings, duration_s, profile)
        for clip in sorted(local, key=lambda h: -h.score):
            if len(chosen) == settings.clip_count:
                break
            if not any(clip.overlaps(c) for c in chosen):
                chosen.append(clip)
    return sorted(chosen, key=lambda h: h.start_s)


def build_prompt(captions: Sequence[Caption], settings: ReelSettings) -> str:
    lines = "\n".join(
        f"[{i}] {c.start_s:.1f}-{c.end_s:.1f}: {' '.join(c.text.split())}"
        for i, c in enumerate(captions)
    )
    return f"""You are an expert short-form video editor. Below is the transcript of a \
coding tutorial: one numbered sentence per line, with its start-end time in seconds.

Choose up to {settings.clip_count} clips that work as standalone YouTube Shorts / \
Instagram Reels:
- Each clip is a continuous run of sentences from start_index to end_index (inclusive).
- Each clip must be between {settings.min_length_s} and {settings.max_length_s} seconds \
long (end time of end_index minus start time of start_index).
- Clips must not overlap.
- Start on a strong hook (a question, a surprising claim, a common mistake) and end on \
a complete thought; the clip must make sense without the rest of the video.
- Prefer moments with a clear takeaway: a tip, a fix, a before/after, an explanation.

For each clip give: title (max 60 characters, in the transcript's language), hook \
(one punchy line to show on screen), score (1-10, how likely it is to perform well) \
and reason (one short sentence). List the best clip first.

Transcript:
{lines}"""


def parse_picks(reply: Any, captions: Sequence[Caption], settings: ReelSettings,
                duration_s: float) -> list[Highlight]:
    """Valid clips from a Gemini reply, trimmed to the length limit."""
    items = reply.get("clips", []) if isinstance(reply, dict) else []
    picks = []
    for item in items if isinstance(items, list) else []:
        clip = _pick_to_clip(item, captions, settings, duration_s)
        if clip is not None:
            picks.append(clip)
    return picks


def _pick_to_clip(item: Any, captions: Sequence[Caption], settings: ReelSettings,
                  duration_s: float) -> Highlight | None:
    try:
        first, last = _index(item["start_index"]), _index(item["end_index"])
        raw_score = item.get("score", 5)
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    score = min(10.0, max(0.0, float(raw_score))) if _is_number(raw_score) else 5.0
    if not 0 <= first <= last < len(captions):
        return None
    while last > first and \
            clip_from_sentences(captions, first, last, duration_s).length_s > settings.max_length_s:
        last -= 1
    base = clip_from_sentences(captions, first, last, duration_s)
    if not settings.min_length_s <= base.length_s <= settings.max_length_s:
        return None
    title = str(item.get("title") or "").strip()
    return Highlight(
        base.start_s, base.end_s, score,
        title=make_title(title) if title else base.title,
        hook=" ".join(str(item.get("hook") or "").split())[:HOOK_MAX_CHARS],
        reason=" ".join(str(item.get("reason") or "").split()),
        first=first, last=last,
    )


def _index(value: Any) -> int:
    """A sentence index: an int (or a whole-number float), never a bool or text."""
    if not _is_number(value) or (isinstance(value, float) and not value.is_integer()):
        raise ValueError(f"not a sentence index: {value!r}")
    return int(value)


def _is_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _reply_json(data: Any) -> Any:
    try:
        parts = data["candidates"][0]["content"]["parts"]
        text = "".join(str(p.get("text", "")) for p in parts)
    except (KeyError, IndexError, TypeError, AttributeError):
        feedback = data.get("promptFeedback") if isinstance(data, dict) else None
        reason = feedback.get("blockReason") if isinstance(feedback, dict) else None
        raise GeminiUnavailable(
            f"Gemini returned no answer{f' (blocked: {reason})' if reason else ''}.") from None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise GeminiUnavailable("Gemini's answer was not valid JSON.") from None
