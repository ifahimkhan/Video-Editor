"""FFmpeg filter graphs that place a landscape video on a reel canvas.

Every graph reads ``[0:v:0]`` and produces ``[v]`` at the canvas size.
Crop rectangles are computed here in Python (not as ffmpeg expressions) so
they are easy to test and later easy to animate for speaker/code tracking.
"""

from __future__ import annotations

from .settings import Layout, Rect, ReelSettings, even, split_heights

# The blurred background is drawn at 1/8 size: blurring is far cheaper and
# looks the same once scaled back up.
BLUR_DOWNSCALE = 8
BLUR_RADIUS = 10
# Default webcam overlay guess: bottom-right corner, a quarter of the width.
WEBCAM_WIDTH_SHARE = 0.25
WEBCAM_ASPECT = 16 / 9


def cover_crop(region: Rect, target_w: int, target_h: int) -> Rect:
    """Largest centred sub-rectangle of `region` with the target's aspect ratio."""
    ratio = target_w / target_h
    if region.w / region.h > ratio:  # region is wider: trim the sides
        h = _even_floor(region.h)
        w = min(even(h * ratio), _even_floor(region.w))
    else:                            # region is taller: trim top and bottom
        w = _even_floor(region.w)
        h = min(even(w / ratio), _even_floor(region.h))
    return Rect(region.x + (region.w - w) // 2, region.y + (region.h - h) // 2, w, h)


def default_webcam(src_w: int, src_h: int) -> Rect:
    """Assumed webcam overlay position: bottom-right corner."""
    w = even(src_w * WEBCAM_WIDTH_SHARE)
    h = min(even(w / WEBCAM_ASPECT), _even_floor(src_h))
    return Rect(src_w - w, src_h - h, w, h)


def build_layout_graph(src_w: int, src_h: int, settings: ReelSettings) -> str:
    """Filter graph turning the source picture into `[v]` at the canvas size.

    SPLIT here is the static version (fixed centre crop); tracked layouts
    are built by `reels.motion`.
    """
    if src_w <= 0 or src_h <= 0:
        raise ValueError("Could not read the video's picture size.")
    canvas_w, canvas_h = settings.aspect.size
    frame = Rect(0, 0, src_w, src_h)
    builders = {
        Layout.FIT_BARS: lambda: _fit_bars(canvas_w, canvas_h),
        Layout.FIT_BLUR: lambda: _fit_blur(frame, canvas_w, canvas_h),
        Layout.STATIC_CROP: lambda: _static_crop(frame, canvas_w, canvas_h),
        Layout.SPLIT: lambda: _split(frame, canvas_w, canvas_h, settings),
    }
    if settings.layout not in builders:
        raise ValueError(f"Layout '{settings.layout.value}' needs clip analysis; "
                         "use reels.motion.build_tracked_graph.")
    return builders[settings.layout]()


def _fit_bars(canvas_w: int, canvas_h: int) -> str:
    return (f"[0:v:0]{_fit_scale(canvas_w, canvas_h)},"
            f"pad={canvas_w}:{canvas_h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1[v]")


def _fit_blur(frame: Rect, canvas_w: int, canvas_h: int) -> str:
    small_w, small_h = even(canvas_w / BLUR_DOWNSCALE), even(canvas_h / BLUR_DOWNSCALE)
    return ";\n".join([
        "[0:v:0]split=2[bg][fg]",
        f"[bg]{_crop(cover_crop(frame, canvas_w, canvas_h))},scale={small_w}:{small_h},"
        f"boxblur={BLUR_RADIUS}:2,scale={canvas_w}:{canvas_h},setsar=1[bgb]",
        f"[fg]{_fit_scale(canvas_w, canvas_h)},setsar=1[fgs]",
        "[bgb][fgs]overlay=(main_w-overlay_w)/2:(main_h-overlay_h)/2[v]",
    ])


def _static_crop(frame: Rect, canvas_w: int, canvas_h: int) -> str:
    crop = cover_crop(frame, canvas_w, canvas_h)
    return f"[0:v:0]{_crop(crop)},scale={canvas_w}:{canvas_h},setsar=1[v]"


def _split(frame: Rect, canvas_w: int, canvas_h: int, settings: ReelSettings) -> str:
    webcam = settings.webcam or default_webcam(frame.w, frame.h)
    if not webcam.fits_inside(frame.w, frame.h):
        raise ValueError(f"Webcam area {webcam} lies outside the "
                         f"{frame.w}x{frame.h} video.")
    top_h, bottom_h = split_heights(canvas_h, settings.split_ratio)
    return ";\n".join([
        "[0:v:0]split=2[scr][cam]",
        f"[scr]{_crop(cover_crop(frame, canvas_w, top_h))},"
        f"scale={canvas_w}:{top_h},setsar=1[top]",
        f"[cam]{_crop(cover_crop(webcam, canvas_w, bottom_h))},"
        f"scale={canvas_w}:{bottom_h},setsar=1[bot]",
        "[top][bot]vstack=inputs=2[v]",
    ])


def _fit_scale(canvas_w: int, canvas_h: int) -> str:
    return (f"scale={canvas_w}:{canvas_h}:force_original_aspect_ratio=decrease"
            ":force_divisible_by=2")


def _crop(rect: Rect) -> str:
    return f"crop={rect.w}:{rect.h}:{rect.x}:{rect.y}"


def _even_floor(value: int) -> int:
    return max(2, value - value % 2)
