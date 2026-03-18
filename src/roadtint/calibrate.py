"""Build a colour profile from a user's own screenshot.

Bundled profiles only cover renderers we happen to have looked at. Anyone
using a regional map service, a custom Mapbox style, or a dark theme needs
their own palette, and hand-writing HSV bands is tedious and error-prone.

Two ways in:

``sample_points``  - the user names a pixel per severity class and the band
is built around the colour found there.

``suggest_palette`` - no input beyond the image: cluster the saturated
pixels by hue and report the dominant groups, so the user can see what the
renderer actually paints before deciding which cluster is which class.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image

from .extract import rgb_to_hsv_array
from .profile import HSVBand, Profile, ProfileError, SeverityClass

__all__ = ["HueCluster", "profile_from_samples", "sample_color", "suggest_palette"]

# Reasonable defaults for "this pixel is part of a coloured overlay"
# rather than basemap, terrain or text.
MIN_SAT = 0.35
MIN_VAL = 0.25


@dataclass
class HueCluster:
    """A group of similarly-hued pixels found in an image."""

    hue_center: float
    hue_min: float
    hue_max: float
    pixel_count: int
    share: float
    mean_sat: float
    mean_val: float
    example_rgb: Tuple[int, int, int]

    def to_band(self, hue_pad: float = 6.0, s_pad: float = 0.25, v_pad: float = 0.25) -> HSVBand:
        """Convert the cluster into a band with a little headroom."""
        return HSVBand(
            h_min=(self.hue_min - hue_pad) % 360.0,
            h_max=(self.hue_max + hue_pad) % 360.0,
            s_min=max(0.0, self.mean_sat - s_pad),
            s_max=1.0,
            v_min=max(0.0, self.mean_val - v_pad),
            v_max=1.0,
        )

    def describe(self) -> str:
        r, g, b = self.example_rgb
        return (
            f"hue {self.hue_center:6.1f}deg  "
            f"[{self.hue_min:5.1f}-{self.hue_max:5.1f}]  "
            f"{self.share * 100:5.2f}% of saturated px  "
            f"rgb({r},{g},{b})  #{r:02x}{g:02x}{b:02x}"
        )


def _load_rgb(image: Image.Image | np.ndarray | str | Path) -> np.ndarray:
    if isinstance(image, (str, Path)):
        with Image.open(image) as im:
            im.load()
            return np.asarray(im.convert("RGB"), dtype=np.uint8)
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ProfileError(f"expected HxWx3 image, got shape {arr.shape}")
    return arr[:, :, :3]


def sample_color(
    image: Image.Image | np.ndarray | str | Path,
    x: int,
    y: int,
    patch: int = 3,
) -> Tuple[int, int, int]:
    """Median colour of a small patch centred on ``(x, y)``.

    A median over a patch rather than a single pixel, because overlays are
    antialiased and the exact pixel a user clicks is often a blend with the
    basemap underneath.
    """
    rgb = _load_rgb(image)
    h, w = rgb.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        raise ProfileError(f"point ({x},{y}) outside image bounds {w}x{h}")

    half = max(0, patch // 2)
    y0, y1 = max(0, y - half), min(h, y + half + 1)
    x0, x1 = max(0, x - half), min(w, x + half + 1)
    region = rgb[y0:y1, x0:x1].reshape(-1, 3)
    med = np.median(region, axis=0).astype(int)
    return (int(med[0]), int(med[1]), int(med[2]))


def suggest_palette(
    image: Image.Image | np.ndarray | str | Path,
    *,
    bin_width: float = 5.0,
    min_share: float = 0.01,
    min_sat: float = MIN_SAT,
    min_val: float = MIN_VAL,
    max_clusters: int = 8,
    bin_share_floor: float = 0.005,
) -> List[HueCluster]:
    """Find the dominant hue clusters among saturated pixels.

    Returns clusters sorted by size, largest first. Adjacent populated hue
    bins are merged, so one overlay colour spread over a few degrees of
    antialiasing comes back as a single cluster rather than three.

    ``bin_share_floor`` is the fraction of saturated pixels a hue bin must
    hold before it counts as populated. It exists because antialiasing puts
    a handful of pixels in nearly every bin, and without a floor the merge
    step walks straight through them and returns one cluster covering the
    entire hue circle.
    """
    rgb = _load_rgb(image)
    hsv = rgb_to_hsv_array(rgb)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    keep = (s >= min_sat) & (v >= min_val)
    total = int(keep.sum())
    if total == 0:
        return []

    hues = h[keep]
    sats = s[keep]
    vals = v[keep]
    flat_rgb = rgb[keep]

    nbins = int(np.ceil(360.0 / bin_width))
    idx = np.minimum((hues / bin_width).astype(int), nbins - 1)
    counts = np.bincount(idx, minlength=nbins)

    # Only treat a bin as populated if it holds a meaningful slice of the
    # saturated pixels. Real screenshots are antialiased, so almost every
    # hue bin holds a few stray pixels; without this floor the runs below
    # chain through those stragglers and merge every colour into one
    # cluster spanning the whole circle.
    bin_floor = max(1, int(total * bin_share_floor))
    populated = counts >= bin_floor
    groups: List[List[int]] = []
    visited = np.zeros(nbins, dtype=bool)
    for start in range(nbins):
        if not populated[start] or visited[start]:
            continue
        # Walk backwards to the run's true start (handles wrap).
        s_idx = start
        while populated[(s_idx - 1) % nbins] and not visited[(s_idx - 1) % nbins]:
            s_idx = (s_idx - 1) % nbins
            if s_idx == start:
                break
        run: List[int] = []
        cur = s_idx
        while populated[cur] and not visited[cur]:
            visited[cur] = True
            run.append(cur)
            cur = (cur + 1) % nbins
            if cur == s_idx:
                break
        if run:
            groups.append(run)

    clusters: List[HueCluster] = []
    for run in groups:
        member = np.isin(idx, run)
        n = int(member.sum())
        share = n / total
        if share < min_share:
            continue
        group_hues = hues[member]
        # Unwrap for a sane centre when the run crosses 0/360.
        if max(run) - min(run) > nbins // 2:
            adj = np.where(group_hues > 180, group_hues - 360, group_hues)
            center = float(np.mean(adj)) % 360.0
        else:
            center = float(np.mean(group_hues))
        mean_rgb = np.median(flat_rgb[member], axis=0).astype(int)
        clusters.append(
            HueCluster(
                hue_center=center,
                hue_min=float(min(run) * bin_width),
                hue_max=float((max(run) + 1) * bin_width),
                pixel_count=n,
                share=share,
                mean_sat=float(np.mean(sats[member])),
                mean_val=float(np.mean(vals[member])),
                example_rgb=(int(mean_rgb[0]), int(mean_rgb[1]), int(mean_rgb[2])),
            )
        )

    clusters.sort(key=lambda c: c.pixel_count, reverse=True)
    return clusters[:max_clusters]


def profile_from_samples(
    image: Image.Image | np.ndarray | str | Path,
    samples: Mapping[str, Sequence[int]],
    *,
    name: str = "custom",
    description: str = "",
    weights: Mapping[str, float] | None = None,
    hue_tol: float = 8.0,
) -> Profile:
    """Build a profile from one sampled point per severity class.

    Args:
        image: the screenshot to sample.
        samples: ``{"free": (x, y), "congested": (x, y), ...}``.
        weights: optional per-class weight; sensible defaults are used for
            the conventional free/slow/congested/blocked names.
    """
    if not samples:
        raise ProfileError("need at least one sample point")

    default_weights: Dict[str, float] = {
        "free": 0.0,
        "slow": 0.5,
        "congested": 1.0,
        "blocked": 1.5,
    }
    weights = dict(default_weights, **(weights or {}))

    rgb = _load_rgb(image)
    classes: List[SeverityClass] = []
    for cls_name, point in samples.items():
        if len(point) != 2:
            raise ProfileError(f"sample for '{cls_name}' must be (x, y), got {point!r}")
        x, y = int(point[0]), int(point[1])
        color = sample_color(rgb, x, y)
        band = HSVBand.around_rgb(color, hue_tol=hue_tol)
        classes.append(
            SeverityClass(
                name=cls_name,
                bands=[band],
                weight=float(weights.get(cls_name, 0.0)),
                color="#{:02x}{:02x}{:02x}".format(*color),
            )
        )

    return Profile(
        name=name,
        description=description or "Generated by `roadtint calibrate`.",
        classes=classes,
    )
