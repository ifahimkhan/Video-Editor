"""A video frame with the reel's crop boxes drawn on top."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy

from ..reels.preview import PreviewBox

BOX_COLOURS = {"frame": QColor("#4fc3f7"), "webcam": QColor("#66bb6a"),
               "crop": QColor("#ffca28")}
PEN_WIDTH = 3


class FramePreview(QLabel):
    """Shows a BGR frame scaled to fit, with labelled boxes."""

    def __init__(self) -> None:
        super().__init__("Select a clip to preview it")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._frame: np.ndarray | None = None
        self._boxes: list[PreviewBox] = []

    def set_frame(self, bgr: np.ndarray | None, boxes: Sequence[PreviewBox] = ()) -> None:
        self._frame = None if bgr is None else np.ascontiguousarray(bgr[:, :, ::-1])
        self._boxes = list(boxes)
        self._redraw()

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        super().resizeEvent(event)
        self._redraw()

    def _redraw(self) -> None:
        if self._frame is None:
            self.setPixmap(QPixmap())
            return
        h, w = self._frame.shape[:2]
        image = QImage(self._frame.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image).scaled(
            max(1, self.width()), max(1, self.height()),
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        scale = pixmap.width() / w
        painter = QPainter(pixmap)
        try:
            for box in self._boxes:
                colour = BOX_COLOURS.get(box.kind, QColor("white"))
                painter.setPen(QPen(colour, PEN_WIDTH))
                r = box.rect
                rect = QRectF(r.x * scale, r.y * scale, r.w * scale, r.h * scale)
                painter.drawRect(rect.adjusted(1, 1, -1, -1))
                painter.drawText(rect.adjusted(6, 4, -4, -4),
                                 Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, box.name)
        finally:
            painter.end()
        self.setPixmap(pixmap)
