"""Turn a traffic-layer image into comparable congestion numbers.

Why not just count contours
---------------------------
The original 2023 implementation reported ``len(contours)`` of the red mask
as "road blockage points". That number is not comparable between two
screenshots: it scales with zoom, with how badly the overlay is broken up by
labels and junctions, and with JPEG noise. Two captures of the same road at
different zoom levels produce different counts for identical traffic.

What this module measures instead is the *share of detected road* in each
severity class. Because it is a ratio over the road actually visible in that
frame, it stays stable when the zoom or crop changes.

Length vs area
--------------
Measuring coloured *area* biases towards wide strokes: a motorway drawn 8px
wide outweighs a side street drawn 3px wide even when both are equally long.
When ``measure="length"`` (the default) each class mask is thinned to a
one-pixel skeleton first, so every road contributes in proportion to its
length. ``measure="area"`` is kept for callers who want raw coverage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from PIL import Image

from .profile import Profile, load_profile

__all__ = ["Measurement", "ExtractionError", "extract", "extract_file", "rgb_to_hsv_array"]


class ExtractionError(RuntimeError):
    """Raised when an image cannot be analysed."""


@dataclass
class Measurement:
    """Result of analysing one frame."""

    source: str
    width: int
    height: int
    measure: str
    profile: str
    #: pixels (or skeleton pixels) per class
    counts: Dict[str, int] = field(default_factory=dict)
    #: fraction of detected road per class, sums to ~1.0
    shares: Dict[str, float] = field(default_factory=dict)
    #: total road detected across all classes
    road_total: int = 0
    #: weighted 0-1 index, higher means worse
    congestion_index: float = 0.0
    #: fraction of the frame that was classified as road at all
    road_coverage: float = 0.0

    def to_row(self) -> Dict[str, Any]:
        """Flatten to a single record for CSV/JSON output."""
        row: Dict[str, Any] = {
            "source": self.source,
            "width": self.width,
            "height": self.height,
            "profile": self.profile,
            "measure": self.measure,
            "road_total": self.road_total,
            "road_coverage": round(self.road_coverage, 6),
            "congestion_index": round(self.congestion_index, 6),
        }
        for name, share in self.shares.items():
            row[f"share_{name}"] = round(share, 6)
        for name, count in self.counts.items():
            row[f"count_{name}"] = count
        return row


def rgb_to_hsv_array(rgb: np.ndarray) -> np.ndarray:
    """Vectorised RGB->HSV for a uint8 HxWx3 array.

    Returns float32 HxWx3 with hue in degrees (0-360) and s/v in 0-1.
    Written out rather than looping pixels through ``colorsys``, which is
    orders of magnitude slower on megapixel frames.
    """
    arr = rgb.astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    cmax = arr.max(axis=-1)
    cmin = arr.min(axis=-1)
    delta = cmax - cmin

    hue = np.zeros_like(cmax)
    # Guard against divide-by-zero on greys; those keep hue 0 and are
    # excluded later by the saturation floor anyway.
    nonzero = delta > 1e-12
    with np.errstate(invalid="ignore", divide="ignore"):
        rmax = nonzero & (cmax == r)
        gmax = nonzero & (cmax == g) & ~rmax
        bmax = nonzero & (cmax == b) & ~rmax & ~gmax
        hue[rmax] = ((g[rmax] - b[rmax]) / delta[rmax]) % 6.0
        hue[gmax] = ((b[gmax] - r[gmax]) / delta[gmax]) + 2.0
        hue[bmax] = ((r[bmax] - g[bmax]) / delta[bmax]) + 4.0
    hue *= 60.0

    sat = np.zeros_like(cmax)
    np.divide(delta, cmax, out=sat, where=cmax > 1e-12)

    return np.stack([hue, sat, cmax], axis=-1)


def _band_mask(hsv: np.ndarray, band) -> np.ndarray:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    ok = (s >= band.s_min) & (s <= band.s_max) & (v >= band.v_min) & (v <= band.v_max)
    if band.wraps:
        ok &= (h >= band.h_min) | (h <= band.h_max)
    else:
        ok &= (h >= band.h_min) & (h <= band.h_max)
    return ok


def _binary_erode(mask: np.ndarray) -> np.ndarray:
    """3x3 erosion with no SciPy dependency."""
    p = np.pad(mask, 1, mode="constant", constant_values=False)
    out = mask.copy()
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            out &= p[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out


def _binary_dilate(mask: np.ndarray) -> np.ndarray:
    p = np.pad(mask, 1, mode="constant", constant_values=False)
    out = np.zeros_like(mask)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            out |= p[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out


def denoise(mask: np.ndarray, open_iters: int = 1, close_iters: int = 1) -> np.ndarray:
    """Morphological opening then closing.

    Opening drops isolated speckle (antialiasing, JPEG ringing); closing
    re-joins a stroke broken by a street label drawn over it.
    """
    out = mask
    for _ in range(open_iters):
        out = _binary_erode(out)
    for _ in range(open_iters):
        out = _binary_dilate(out)
    for _ in range(close_iters):
        out = _binary_dilate(out)
    for _ in range(close_iters):
        out = _binary_erode(out)
    return out


def skeleton_length(mask: np.ndarray, max_iters: int = 64) -> int:
    """Approximate the centre-line length of a binary mask, in pixels.

    Uses iterative thinning: repeatedly erode and count what survives, which
    estimates length as area divided by mean stroke width. This avoids a
    SciPy/scikit-image dependency while removing most of the stroke-width
    bias that makes raw area unusable for comparing wide and narrow roads.
    """
    if not mask.any():
        return 0

    total_area = int(mask.sum())
    layers = 0
    current = mask
    # Count how many erosion layers the shape survives -> mean half-width.
    for _ in range(max_iters):
        current = _binary_erode(current)
        if not current.any():
            break
        layers += 1

    # A stroke of width w survives about w/2 erosions, so width ~ 2*layers+1.
    width = 2 * layers + 1
    return max(1, int(round(total_area / width)))


def extract(
    image: Image.Image | np.ndarray,
    profile: Profile | str = "google_maps",
    *,
    measure: str = "length",
    roi: Sequence[int] | None = None,
    denoise_mask: bool = True,
    source: str = "<memory>",
) -> Measurement:
    """Analyse one frame and return per-class road shares.

    Args:
        image: PIL image or HxWx3 uint8 array.
        profile: a Profile, or the name/path of one.
        measure: ``"length"`` (skeletonised, default) or ``"area"``.
        roi: optional ``(left, top, right, bottom)`` crop applied first,
            used to exclude UI chrome, legends and watermarks.
        denoise_mask: apply morphological open/close before measuring.
        source: label recorded on the result.
    """
    if measure not in {"length", "area"}:
        raise ExtractionError(f"measure must be 'length' or 'area', got {measure!r}")

    prof = profile if isinstance(profile, Profile) else load_profile(profile)

    if isinstance(image, Image.Image):
        if roi is not None:
            image = image.crop(tuple(roi))
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        rgb = np.asarray(image, dtype=np.uint8)
        if roi is not None:
            left, top, right, bottom = roi
            rgb = rgb[top:bottom, left:right]
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ExtractionError(f"expected HxWx3 array, got shape {rgb.shape}")
        rgb = rgb[:, :, :3]

    if rgb.size == 0:
        raise ExtractionError("image is empty (check the roi bounds)")

    height, width = rgb.shape[:2]
    hsv = rgb_to_hsv_array(rgb)

    counts: Dict[str, int] = {}
    for cls_ in prof.classes:
        mask = np.zeros((height, width), dtype=bool)
        for band in cls_.bands:
            mask |= _band_mask(hsv, band)
        if denoise_mask:
            mask = denoise(mask)
        counts[cls_.name] = (
            skeleton_length(mask) if measure == "length" else int(mask.sum())
        )

    road_total = sum(counts.values())
    shares = {
        name: (count / road_total if road_total else 0.0) for name, count in counts.items()
    }

    weights = {c.name: c.weight for c in prof.classes}
    max_weight = max(weights.values()) or 1.0
    index = sum(shares[n] * weights[n] for n in shares) / max_weight

    return Measurement(
        source=source,
        width=width,
        height=height,
        measure=measure,
        profile=prof.name,
        counts=counts,
        shares=shares,
        road_total=road_total,
        congestion_index=index,
        road_coverage=road_total / float(height * width) if height * width else 0.0,
    )


def extract_file(
    path: str | Path,
    profile: Profile | str = "google_maps",
    **kwargs: Any,
) -> Measurement:
    """Analyse an image file."""
    path = Path(path)
    if not path.is_file():
        raise ExtractionError(f"no such image: {path}")
    try:
        with Image.open(path) as im:
            im.load()
            kwargs.setdefault("source", path.name)
            return extract(im, profile, **kwargs)
    except OSError as exc:
        raise ExtractionError(f"cannot read {path}: {exc}") from exc
