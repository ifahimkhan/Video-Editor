"""Burned-in captions and title overlay as an ASS subtitle file. Pure logic.

Words are grouped into short lines (a few words, broken at pauses and
sentence ends). Each word gets its own event showing the whole line with
that word highlighted, so the highlight follows the speech karaoke-style.
Events are rendered by ffmpeg's libass ``subtitles`` filter.

When pauses are cut out of a clip, `remap_words` moves every word onto the
shortened timeline and drops words that fell inside a cut.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..segments import Segment
from ..transcript import Word
from .settings import CaptionStyle, Layout, ReelSettings, split_heights

MAX_LINE_WORDS = 4
MAX_LINE_CHARS = 22
MAX_GAP_S = 0.6        # a longer pause starts a new line
LINGER_S = 0.4         # keep a line up this long after its last word
TITLE_S = 4.0          # title overlay duration
CAPTION_HEIGHT_SHARE = 0.72
TITLE_HEIGHT_SHARE = 0.10
HIGHLIGHT = "&H00FFFF&"  # yellow, in ASS &HBBGGRR&
WHITE = "&HFFFFFF&"
SENTENCE_END = (".", "!", "?")
CLOSING_MARKS = "\"')]}»”’"
# Rough glyph width as a share of the font size (Arial bold), used to shrink a
# line that would not fit: capitals are wider than mixed case.
UPPER_WIDTH_SHARE = 0.68
MIXED_WIDTH_SHARE = 0.55


@dataclass(frozen=True)
class _Style:
    font_size: int
    outline_colour: str
    border_style: int   # 1 = outline + shadow, 3 = opaque box
    outline: int
    shadow: int
    upper: bool
    highlight: bool
    max_chars: int = MAX_LINE_CHARS


_STYLES = {
    CaptionStyle.BOLD: _Style(84, "&H00000000", 1, 6, 2, upper=True, highlight=True,
                              max_chars=16),
    # Opaque: the highlight splits a line into runs, each with its own box, and
    # semi-transparent boxes would show darker where they overlap.
    CaptionStyle.BOX: _Style(72, "&H00141414", 3, 16, 0, upper=False, highlight=True),
    CaptionStyle.MINIMAL: _Style(68, "&H00000000", 1, 3, 1, upper=False, highlight=False),
}
_STYLE_FORMAT = ("Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
                 "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, "
                 "Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, "
                 "Encoding")


def group_words(words: Sequence[Word],
                caption_style: CaptionStyle | None = None) -> list[list[Word]]:
    """Split words into caption lines (shorter lines for big styles)."""
    max_chars = _STYLES[caption_style].max_chars if caption_style else MAX_LINE_CHARS
    groups: list[list[Word]] = []
    line: list[Word] = []
    for word in words:
        if line and _starts_new_line(line, word, max_chars):
            groups.append(line)
            line = []
        line.append(word)
    if line:
        groups.append(line)
    return groups


def build_ass(words: Sequence[Word], settings: ReelSettings, duration_s: float,
              title: str | None = None) -> str:
    """The ASS file text for a clip (times relative to the clip start)."""
    width, height = settings.aspect.size
    style = _STYLES[settings.caption_style]
    events = []
    if settings.title_overlay and title:
        events.append(_dialogue(0.0, min(TITLE_S, duration_s), "Title",
                                f"{{\\an8\\pos({width // 2},{round(height * TITLE_HEIGHT_SHARE)})}}"
                                f"{_escape(title)}"))
    if settings.captions:
        position = f"{{\\an5\\pos({width // 2},{caption_y(settings)})}}"
        events += _word_events(group_words(words, settings.caption_style), style,
                               position, duration_s, width)
    return "\n".join([
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {width}", f"PlayResY: {height}",
        "WrapStyle: 0", "ScaledBorderAndShadow: yes", "",
        "[V4+ Styles]", f"Format: {_STYLE_FORMAT}",
        _style_line("Caption", style, width),
        _style_line("Title", _Style(60, "&H40000000", 3, 18, 0, False, False), width),
        "", "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        *events, "",
    ])


def caption_y(settings: ReelSettings) -> int:
    """Vertical centre of the caption line on the canvas."""
    height = settings.aspect.size[1]
    if settings.layout is Layout.SPLIT:
        return split_heights(height, settings.split_ratio)[0]  # on the divider
    return round(height * CAPTION_HEIGHT_SHARE)


def remap_words(words: Sequence[Word], clip_start_s: float, clip_end_s: float,
                keep: tuple[Segment, ...] | None = None) -> list[Word]:
    """Words inside the clip on its own timeline (and through its cuts, if any)."""
    length = clip_end_s - clip_start_s
    out = []
    for word in words:
        start, end = word.start_s - clip_start_s, word.end_s - clip_start_s
        if end <= 0 or start >= length:
            continue
        start, end = max(0.0, start), min(length, end)
        if keep is not None:
            mapped = _through_cuts(start, end, keep)
            if mapped is None:
                continue
            start, end = mapped
        out.append(Word(start, end, word.text))
    return out


def clip_keep(keep: Sequence[Segment], start_s: float, end_s: float,
              tolerance_ms: int = 10) -> tuple[Segment, ...] | None:
    """Kept parts of the whole file, cut to the clip and made clip-relative.

    None when nothing inside the clip is removed.
    """
    lo, hi = round(start_s * 1000), round(end_s * 1000)
    parts = tuple(Segment(max(s.start_ms, lo) - lo, min(s.end_ms, hi) - lo)
                  for s in keep if s.end_ms > lo and s.start_ms < hi)
    if len(parts) == 1 and parts[0].start_ms <= tolerance_ms \
            and parts[0].end_ms >= hi - lo - tolerance_ms:
        return None
    return parts


def ass_time(seconds: float) -> str:
    cs = max(0, round(seconds * 100))
    hours, rest = divmod(cs, 360_000)
    minutes, rest = divmod(rest, 6000)
    secs, cs = divmod(rest, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def _starts_new_line(line: list[Word], word: Word, max_chars: int) -> bool:
    chars = sum(len(w.text) + 1 for w in line) + len(word.text)
    return (len(line) >= MAX_LINE_WORDS or chars > max_chars
            or word.start_s - line[-1].end_s > MAX_GAP_S
            or line[-1].text.rstrip(CLOSING_MARKS).endswith(SENTENCE_END))


def _word_events(groups: list[list[Word]], style: _Style, position: str,
                 duration_s: float, width: int) -> list[str]:
    events = []
    for g, line in enumerate(groups):
        next_start = groups[g + 1][0].start_s if g + 1 < len(groups) else duration_s
        line_end = min(line[-1].end_s + LINGER_S, next_start, duration_s)
        texts = [_escape(w.text.upper() if style.upper else w.text) for w in line]
        fit = _fit_scale(" ".join(texts), style, width)
        for i, word in enumerate(line):
            start = word.start_s
            end = line[i + 1].start_s if i + 1 < len(line) else line_end
            if end <= start:
                continue
            shown = [f"{{\\c{HIGHLIGHT}}}{t}{{\\c{WHITE}}}" if j == i and style.highlight
                     else t for j, t in enumerate(texts)]
            events.append(_dialogue(start, end, "Caption", position + fit + " ".join(shown)))
    return events


def _fit_scale(text: str, style: _Style, width: int) -> str:
    """Override shrinking a line that would be wider than the canvas, else ""."""
    usable = width * (1 - 2 / 18) - 2 * style.outline
    share = UPPER_WIDTH_SHARE if style.upper else MIXED_WIDTH_SHARE
    estimate = len(text) * style.font_size * share
    if estimate <= usable:
        return ""
    scale = max(30, int(100 * usable / estimate))
    return f"{{\\fscx{scale}\\fscy{scale}}}"


def _dialogue(start: float, end: float, style: str, text: str) -> str:
    return f"Dialogue: 0,{ass_time(start)},{ass_time(end)},{style},,0,0,0,,{text}"


def _style_line(name: str, style: _Style, width: int) -> str:
    margin = width // 18
    return (f"Style: {name},Arial,{style.font_size},&H00FFFFFF,&H00FFFFFF,"
            f"{style.outline_colour},&H80000000,-1,0,0,0,100,100,0,0,{style.border_style},"
            f"{style.outline},{style.shadow},5,{margin},{margin},0,1")


def _escape(text: str) -> str:
    """Text safe for an ASS event: no override blocks or escape sequences."""
    return " ".join(text.replace("{", "").replace("}", "").replace("\\", "/").split())


def _through_cuts(start: float, end: float,
                  keep: tuple[Segment, ...]) -> tuple[float, float] | None:
    middle_ms = (start + end) * 500
    removed_ms = 0
    previous_end = 0
    for seg in keep:
        removed_ms += seg.start_ms - previous_end
        previous_end = seg.end_ms
        if seg.start_ms <= middle_ms < seg.end_ms:
            lo = max(start * 1000, seg.start_ms) - removed_ms
            hi = min(end * 1000, seg.end_ms) - removed_ms
            return lo / 1000, max(hi, lo + 1) / 1000
    return None
