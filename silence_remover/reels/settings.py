"""Reel settings: canvas, layout and clip-length rules. Pure, no I/O."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

LENGTH_RANGE_S = (5, 180)
CLIP_COUNT_RANGE = (1, 10)
SPLIT_RATIO_RANGE = (0.3, 0.8)


class Layout(str, Enum):
    FIT_BLUR = "fit-blur"        # whole video centred on a blurred copy of itself
    FIT_BARS = "fit-bars"        # whole video centred on black bars
    STATIC_CROP = "static-crop"  # fixed centre crop filling the canvas
    SPLIT = "split"              # screen on top, webcam overlay on the bottom


class Aspect(str, Enum):
    VERTICAL = "9:16"   # Shorts, Reels, TikTok
    SQUARE = "1:1"
    PORTRAIT = "4:5"    # Instagram feed

    @property
    def size(self) -> tuple[int, int]:
        """Output canvas (width, height) in pixels."""
        return _CANVAS[self]


_CANVAS = {
    Aspect.VERTICAL: (1080, 1920),
    Aspect.SQUARE: (1080, 1080),
    Aspect.PORTRAIT: (1080, 1350),
}


@dataclass(frozen=True)
class Rect:
    """Pixel rectangle in the source video."""

    x: int
    y: int
    w: int
    h: int

    def __post_init__(self) -> None:
        if self.w <= 0 or self.h <= 0 or self.x < 0 or self.y < 0:
            raise ValueError(f"Invalid rectangle {self}: size must be positive, "
                             "position not negative.")

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h

    def fits_inside(self, width: int, height: int) -> bool:
        return self.right <= width and self.bottom <= height

    @classmethod
    def parse(cls, text: str) -> Rect:
        """Parse "X,Y,W,H" (as typed on the command line)."""
        parts = text.split(",")
        if len(parts) != 4:
            raise ValueError(f"Expected X,Y,W,H but got '{text}'.")
        try:
            x, y, w, h = (int(p.strip()) for p in parts)
        except ValueError as exc:
            raise ValueError(f"Expected four whole numbers X,Y,W,H but got '{text}'.") from exc
        return cls(x, y, w, h)


@dataclass(frozen=True)
class ReelSettings:
    """How reels are cut and framed.

    layout:        how the landscape picture is placed on the canvas.
    aspect:        output canvas shape.
    min/max_length_s: allowed clip length when clips are picked automatically.
    clip_count:    how many clips to pick automatically.
    split_ratio:   SPLIT layout: share of the canvas height given to the screen.
    webcam:        SPLIT layout: webcam overlay position in the source,
                   or None to assume the bottom-right corner.
    """

    layout: Layout = Layout.FIT_BLUR
    aspect: Aspect = Aspect.VERTICAL
    min_length_s: int = 30
    max_length_s: int = 60
    clip_count: int = 3
    split_ratio: float = 0.6
    webcam: Rect | None = None

    def __post_init__(self) -> None:
        # Accept plain strings ("split", "9:16") from the CLI or saved settings.
        object.__setattr__(self, "layout", Layout(self.layout))
        object.__setattr__(self, "aspect", Aspect(self.aspect))
        _check_range("min_length_s", self.min_length_s, LENGTH_RANGE_S)
        _check_range("max_length_s", self.max_length_s, LENGTH_RANGE_S)
        if self.min_length_s > self.max_length_s:
            raise ValueError("min_length_s must not be longer than max_length_s.")
        _check_range("clip_count", self.clip_count, CLIP_COUNT_RANGE)
        _check_range("split_ratio", self.split_ratio, SPLIT_RATIO_RANGE)


def split_heights(canvas_h: int, ratio: float) -> tuple[int, int]:
    """Even (top, bottom) heights for the SPLIT layout, summing to `canvas_h`."""
    top = even(canvas_h * ratio)
    return top, canvas_h - top


def even(value: float) -> int:
    """Nearest even integer >= 2; yuv420p video needs even dimensions."""
    return max(2, int(round(value / 2)) * 2)


def _check_range(name: str, value: float, bounds: tuple[float, float]) -> None:
    low, high = bounds
    if not low <= value <= high:
        raise ValueError(f"{name}={value} is outside {low}..{high}")
