"""Tests for the crop boxes drawn on the reels preview."""

import pytest

from silence_remover.reels.preview import preview_boxes
from silence_remover.reels.settings import Layout, Rect, ReelSettings

CAM = Rect(1440, 810, 480, 270)


def names(layout, **kw):
    return [b.name for b in preview_boxes(ReelSettings(layout=layout, **kw), 1920, 1080, CAM)]


def test_fit_layouts_show_the_whole_frame():
    boxes = preview_boxes(ReelSettings(layout=Layout.FIT_BLUR), 1920, 1080, CAM)
    assert [(b.name, b.rect) for b in boxes] == [("Whole video", Rect(0, 0, 1920, 1080))]


def test_static_crop_box_is_centred():
    (box,) = preview_boxes(ReelSettings(layout=Layout.STATIC_CROP), 1920, 1080, CAM)
    assert box.rect == Rect(656, 0, 608, 1080)


def test_speaker_layout_shows_webcam_and_crop_inside_it():
    boxes = preview_boxes(ReelSettings(layout=Layout.SPEAKER), 1920, 1080, CAM)
    assert [b.name for b in boxes] == ["Webcam", "Speaker crop (follows face)"]
    crop = boxes[1].rect
    assert crop.x >= CAM.x and crop.right <= CAM.right


def test_code_zoom_shrinks_code_box():
    wide = preview_boxes(ReelSettings(layout=Layout.CODE), 1920, 1080, CAM)[-1].rect
    zoomed = preview_boxes(ReelSettings(layout=Layout.CODE, code_zoom=2.0),
                           1920, 1080, CAM)[-1].rect
    assert zoomed.w == pytest.approx(wide.w / 2, abs=2)


def test_split_and_auto_show_both_crops():
    assert names(Layout.SPLIT) == ["Webcam", "Code panel (follows typing)",
                                   "Speaker panel (follows face)"]
    assert names(Layout.AUTO) == ["Webcam", "Code crop (follows typing)",
                                  "Speaker crop (follows face)"]


def test_without_webcam_tracked_layout_assumes_corner():
    boxes = preview_boxes(ReelSettings(layout=Layout.SPEAKER), 1920, 1080, None)
    assert boxes[0].rect.right == 1920 and boxes[0].rect.bottom == 1080
