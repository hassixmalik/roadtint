"""Synthetic fixtures.

Tests deliberately do not use the bundled map screenshots: those are
third-party imagery that cannot be redistributed, and a test suite that
depends on them cannot be run by a contributor who only has the source.
Everything here is drawn from scratch with known ground truth.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

# Colours sampled from a real Google Maps traffic layer.
GREEN = (52, 168, 83)
ORANGE = (249, 171, 0)
RED = (232, 69, 60)
DARK_RED = (139, 26, 16)
GREY = (128, 128, 128)
WHITE = (255, 255, 255)
PIN_BLUE = (66, 133, 244)


def canvas(w: int = 200, h: int = 200, fill=GREY) -> np.ndarray:
    return np.full((h, w, 3), fill, dtype=np.uint8)


def hline(arr: np.ndarray, y: int, color, x0: int = 0, x1: int | None = None, width: int = 4):
    """Draw a horizontal stroke of a given pixel width."""
    x1 = arr.shape[1] if x1 is None else x1
    arr[y : y + width, x0:x1] = color
    return arr


def vline(arr: np.ndarray, x: int, color, y0: int = 0, y1: int | None = None, width: int = 4):
    y1 = arr.shape[0] if y1 is None else y1
    arr[y0:y1, x : x + width] = color
    return arr


def to_image(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def all_green():
    """One green road across a grey background."""
    a = canvas()
    hline(a, 100, GREEN)
    return to_image(a)


@pytest.fixture
def all_red():
    a = canvas()
    hline(a, 100, RED)
    return to_image(a)


@pytest.fixture
def half_red_half_green():
    """Two strokes of identical length and width: one green, one red."""
    a = canvas()
    hline(a, 60, GREEN, width=4)
    hline(a, 140, RED, width=4)
    return to_image(a)


@pytest.fixture
def wide_green_thin_red():
    """Equal-length roads, very different stroke widths.

    Area-based measurement reports mostly green here; length-based
    measurement should report roughly half and half.
    """
    a = canvas()
    hline(a, 60, GREEN, width=12)
    hline(a, 140, RED, width=3)
    return to_image(a)


@pytest.fixture
def zoomed_pair():
    """The same scene at two scales, for scale-invariance checks."""
    small = canvas(200, 200)
    hline(small, 60, GREEN, width=3)
    hline(small, 140, RED, width=3)

    big = canvas(400, 400)
    hline(big, 120, GREEN, width=6)
    hline(big, 280, RED, width=6)

    return to_image(small), to_image(big)


@pytest.fixture
def with_ui_chrome():
    """A red road plus blue map pins and white label text."""
    a = canvas()
    hline(a, 100, RED)
    a[10:30, 10:30] = PIN_BLUE      # map pin
    a[170:185, 20:120] = WHITE      # label text block
    return to_image(a)


@pytest.fixture
def blank_grey():
    return to_image(canvas())
