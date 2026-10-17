"""Pure-logic tests for reel settings and filter-graph layouts."""

import pytest

from silence_remover.reels.layout import build_layout_graph, cover_crop, default_webcam
from silence_remover.reels.settings import Aspect, Layout, Rect, ReelSettings, split_heights


def test_aspect_canvas_sizes():
    assert Aspect.VERTICAL.size == (1080, 1920)
    assert Aspect.SQUARE.size == (1080, 1080)
    assert Aspect.PORTRAIT.size == (1080, 1350)


def test_settings_defaults_are_vertical_30_to_60s():
    s = ReelSettings()
    assert s.aspect is Aspect.VERTICAL
    assert (s.min_length_s, s.max_length_s) == (30, 60)


def test_settings_accept_string_enums():
    s = ReelSettings(layout="split", aspect="1:1")
    assert s.layout is Layout.SPLIT and s.aspect is Aspect.SQUARE


@pytest.mark.parametrize("kwargs", [
    {"min_length_s": 70, "max_length_s": 60},
    {"min_length_s": 2},
    {"max_length_s": 500},
    {"clip_count": 0},
    {"split_ratio": 0.95},
    {"layout": "nope"},
])
def test_settings_reject_invalid_values(kwargs):
    with pytest.raises(ValueError):
        ReelSettings(**kwargs)


def test_rect_rejects_non_positive_size():
    with pytest.raises(ValueError):
        Rect(0, 0, 0, 10)


def test_rect_parse():
    assert Rect.parse("10,20,300,200") == Rect(10, 20, 300, 200)
    with pytest.raises(ValueError):
        Rect.parse("10,20,300")


def test_cover_crop_trims_sides_of_landscape_source():
    crop = cover_crop(Rect(0, 0, 1920, 1080), 1080, 1920)
    assert crop.h == 1080
    assert crop.w == 608  # 1080 * 9/16 = 607.5, rounded to even
    assert crop.x == (1920 - 608) // 2 and crop.y == 0


def test_cover_crop_trims_top_and_bottom_of_tall_source():
    crop = cover_crop(Rect(0, 0, 1080, 1920), 1080, 1080)
    assert (crop.w, crop.h) == (1080, 1080)
    assert crop.y == 420


def test_cover_crop_stays_inside_offset_region():
    region = Rect(1400, 760, 480, 270)
    crop = cover_crop(region, 1080, 768)
    assert crop.x >= region.x and crop.right <= region.right
    assert crop.y >= region.y and crop.bottom <= region.bottom
    assert crop.w % 2 == 0 and crop.h % 2 == 0


def test_default_webcam_is_bottom_right():
    cam = default_webcam(1920, 1080)
    assert cam.right == 1920 and cam.bottom == 1080
    assert cam.w < 1920 / 2 and cam.h < 1080 / 2


def test_split_heights_sum_to_canvas_and_are_even():
    top, bottom = split_heights(1920, 0.6)
    assert top + bottom == 1920
    assert top % 2 == 0 and bottom % 2 == 0
    assert top == 1152


@pytest.mark.parametrize("layout", [m for m in Layout if not m.tracked])
def test_every_static_layout_outputs_v_label_at_canvas_size(layout):
    graph = build_layout_graph(1920, 1080, ReelSettings(layout=layout))
    assert graph.startswith("[0:v:0]")
    assert graph.rstrip().endswith("[v]")


def test_fit_bars_pads_to_canvas():
    graph = build_layout_graph(1920, 1080, ReelSettings(layout=Layout.FIT_BARS))
    assert "pad=1080:1920" in graph
    assert "force_original_aspect_ratio=decrease" in graph


def test_fit_blur_overlays_on_blurred_background():
    graph = build_layout_graph(1920, 1080, ReelSettings(layout=Layout.FIT_BLUR))
    assert "boxblur" in graph and "overlay" in graph


def test_static_crop_uses_centre_crop():
    graph = build_layout_graph(1920, 1080, ReelSettings(layout=Layout.STATIC_CROP))
    assert "crop=608:1080:656:0" in graph
    assert "scale=1080:1920" in graph


def test_tracked_layouts_are_flagged():
    assert {m for m in Layout if m.tracked} == {
        Layout.SPLIT, Layout.SPEAKER, Layout.CODE, Layout.AUTO}


def test_code_zoom_range_is_checked():
    with pytest.raises(ValueError):
        ReelSettings(code_zoom=5)


def test_split_stacks_screen_above_webcam():
    cam = Rect(1440, 810, 480, 270)
    graph = build_layout_graph(1920, 1080, ReelSettings(layout=Layout.SPLIT, webcam=cam))
    assert "vstack" in graph
    assert "scale=1080:1152" in graph and "scale=1080:768" in graph
    # Webcam crop must lie inside the given webcam rectangle.
    webcam_crop = cover_crop(cam, 1080, 768)
    assert (f"crop={webcam_crop.w}:{webcam_crop.h}:{webcam_crop.x}:{webcam_crop.y}"
            in graph)


def test_split_rejects_webcam_outside_frame():
    settings = ReelSettings(layout=Layout.SPLIT, webcam=Rect(1800, 1000, 480, 270))
    with pytest.raises(ValueError):
        build_layout_graph(1920, 1080, settings)


def test_layout_rejects_unknown_source_size():
    with pytest.raises(ValueError):
        build_layout_graph(0, 0, ReelSettings())
