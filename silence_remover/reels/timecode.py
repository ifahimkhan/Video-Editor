"""Parse and format timestamps typed by the user ("1:25", "1:02:05", "85.5")."""

from __future__ import annotations

import math


def parse_time(text: str) -> float:
    """Seconds from `[[h:]m:]s[.fraction]`; a comma also works as decimal mark."""
    cleaned = text.strip().replace(",", ".")
    parts = cleaned.split(":")
    if not cleaned or len(parts) > 3:
        raise ValueError(f"'{text}' is not a time. Use seconds, m:ss or h:mm:ss.")
    try:
        values = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"'{text}' is not a time. Use seconds, m:ss or h:mm:ss.") from None
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError(f"'{text}' is not a valid time.")
    if any(v >= 60 for v in values[1:]):
        raise ValueError(f"'{text}': minutes and seconds must be below 60.")
    if any(not v.is_integer() for v in values[:-1]):
        raise ValueError(f"'{text}': only the seconds may have a fraction.")
    seconds = 0.0
    for value in values:
        seconds = seconds * 60 + value
    return seconds


def format_time(seconds: float) -> str:
    """`m:ss` or `h:mm:ss`, with tenths when not a whole second."""
    tenths = round(seconds * 10)
    whole, frac = divmod(tenths, 10)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    text = f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"
    return f"{text}.{frac}" if frac else text
