"""Tests for the Gemini highlight ranker (no network: fake clients)."""

import io
import json
import urllib.error

import pytest

from silence_remover.reels import gemini_ranker
from silence_remover.reels.gemini_ranker import (
    RESPONSE_SCHEMA,
    GeminiClient,
    GeminiUnavailable,
    build_prompt,
    parse_picks,
    rank_with_gemini,
)
from silence_remover.reels.settings import ReelSettings

from .test_reels_highlights import FILLER, sentences

SETTINGS = ReelSettings(min_length_s=30, max_length_s=60, clip_count=2)
CAPS = sentences(FILLER)
DURATION = CAPS[-1].end_s


class FakeClient:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def generate_json(self, prompt, schema):
        self.prompts.append((prompt, schema))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def pick(first, last, score=9, title="T", hook="H"):
    return {"start_index": first, "end_index": last, "title": title, "hook": hook,
            "score": score, "reason": "because"}


def test_prompt_lists_numbered_sentences_and_rules():
    prompt = build_prompt(CAPS[:3], SETTINGS)
    assert "[0] 0.0-8.0" in prompt and "[2] 17.0-25.0" in prompt
    assert "30" in prompt and "60" in prompt and "2" in prompt
    assert FILLER[1] in prompt


def test_schema_asks_for_sentence_indices():
    item = RESPONSE_SCHEMA["properties"]["clips"]["items"]
    assert {"start_index", "end_index", "title", "score"} <= set(item["required"])


def test_valid_picks_become_highlights_with_titles():
    client = FakeClient({"clips": [pick(10, 14, title="Async tip", hook="Stop!")]})
    picks = rank_with_gemini(CAPS, SETTINGS, DURATION, client=client)
    gem = [p for p in picks if p.title == "Async tip"]
    assert len(gem) == 1
    assert gem[0].hook == "Stop!" and gem[0].first == 10 and gem[0].last == 14
    assert gem[0].start_s <= CAPS[10].start_s and gem[0].end_s >= CAPS[14].end_s


def test_too_long_pick_is_trimmed_to_max_length():
    picks = parse_picks({"clips": [pick(0, 20)]}, CAPS, SETTINGS, DURATION)
    assert len(picks) == 1 and picks[0].length_s <= 60 and picks[0].first == 0


def test_invalid_picks_are_dropped():
    reply = {"clips": [pick(5, 2), pick(-1, 3), pick(0, 999), pick(0, 1),  # bad/short
                       {"start_index": "x"}]}
    assert parse_picks(reply, CAPS, SETTINGS, DURATION) == []


def test_overlapping_picks_keep_the_better_one_and_top_up_locally():
    client = FakeClient({"clips": [pick(10, 14, score=6, title="B"),
                                   pick(12, 16, score=9, title="A")]})
    picks = rank_with_gemini(CAPS, SETTINGS, DURATION, client=client)
    titles = [p.title for p in picks]
    assert "A" in titles and "B" not in titles
    assert len(picks) == 2  # one slot filled by the local scorer
    assert picks[0].start_s < picks[1].start_s


def test_api_failure_falls_back_to_local():
    client = FakeClient(GeminiUnavailable("quota exceeded"))
    picks = rank_with_gemini(CAPS, SETTINGS, DURATION, client=client)
    assert len(picks) == 2


def test_missing_key_is_reported(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(GeminiUnavailable, match="GEMINI_API_KEY"):
        GeminiClient()


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_client_sends_schema_and_parses_reply(monkeypatch):
    sent = {}

    def fake_urlopen(request, timeout):
        sent["url"] = request.full_url
        sent["headers"] = dict(request.header_items())
        sent["body"] = json.loads(request.data)
        reply = {"candidates": [{"content": {"parts": [{"text": '{"clips": []}'}]}}]}
        return FakeResponse(json.dumps(reply).encode())

    monkeypatch.setattr(gemini_ranker.urllib.request, "urlopen", fake_urlopen)
    client = GeminiClient(api_key="test-key", model="gemini-2.5-flash")
    assert client.generate_json("hello", RESPONSE_SCHEMA) == {"clips": []}
    assert sent["url"].endswith("/models/gemini-2.5-flash:generateContent")
    assert sent["headers"]["X-goog-api-key"] == "test-key"
    assert "test-key" not in sent["url"]
    config = sent["body"]["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert config["responseSchema"] == RESPONSE_SCHEMA


def test_client_http_error_hides_key(monkeypatch):
    def fake_urlopen(request, timeout):
        body = io.BytesIO(b'{"error": {"message": "API key not valid"}}')
        raise urllib.error.HTTPError(request.full_url, 400, "Bad", {}, body)

    monkeypatch.setattr(gemini_ranker.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(GeminiUnavailable) as err:
        GeminiClient(api_key="secret-key").generate_json("x", RESPONSE_SCHEMA)
    assert "API key not valid" in str(err.value) and "secret-key" not in str(err.value)


def test_client_rejects_reply_without_text(monkeypatch):
    monkeypatch.setattr(gemini_ranker.urllib.request, "urlopen",
                        lambda request, timeout: FakeResponse(b'{"candidates": []}'))
    with pytest.raises(GeminiUnavailable):
        GeminiClient(api_key="k").generate_json("x", RESPONSE_SCHEMA)


def test_non_integer_indices_are_rejected():
    reply = {"clips": [pick(True, 5), pick(1.9, 5), pick("3", 7), pick(1e999, 5)]}
    assert parse_picks(reply, CAPS, SETTINGS, DURATION) == []


def test_whole_number_float_index_is_accepted():
    assert len(parse_picks({"clips": [pick(10.0, 14)]}, CAPS, SETTINGS, DURATION)) == 1


def test_garbage_reply_falls_back_to_local():
    client = FakeClient({"clips": [None, 5, {"start_index": [1]}]})
    assert len(rank_with_gemini(CAPS, SETTINGS, DURATION, client=client)) == 2


def test_oversized_reply_is_refused(monkeypatch):
    big = b"x" * (gemini_ranker.MAX_REPLY_BYTES + 10)
    monkeypatch.setattr(gemini_ranker.urllib.request, "urlopen",
                        lambda request, timeout: FakeResponse(big))
    with pytest.raises(GeminiUnavailable, match="too large"):
        GeminiClient(api_key="k").generate_json("x", RESPONSE_SCHEMA)


def test_key_with_control_characters_is_rejected():
    with pytest.raises(GeminiUnavailable) as err:
        GeminiClient(api_key="abc\ndef")
    assert "abc" not in str(err.value)
