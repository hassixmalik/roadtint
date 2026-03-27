"""Render what the extractor matched, for checking a profile by eye.

Numbers alone do not tell you whether a profile is right. A free share of
0.9 looks plausible until you see that the "free" band is matching parkland
rather than roads. This draws the masks back over the frame so a bad band
is obvious immediately.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
from PIL import Image

from .extract import _band_mask, denoise, rgb_to_hsv_array
from .profile import Profile, load_profile

__all__ = ["class_masks", "render_overlay", "save_overlay"]


def _hex_to_rgb(value: str) -> Tuple[int, int, int]:
    text = value.lstrip("#")
    if len(text) == 3:
        text = "".join(c * 2 for c in text)
    if len(text) != 6:
        return (136, 136, 136)
    try:
        return tuple(int(text[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    except ValueError:
        return (136, 136, 136)


def class_masks(
    image: Image.Image,
    profile: Profile | str = "google_maps",
    *,
    roi: Sequence[int] | None = None,
    denoise_mask: bool = True,
) -> Tuple[Image.Image, Dict[str, np.ndarray]]:
    """Return the (possibly cropped) frame and one boolean mask per class."""
    prof = profile if isinstance(profile, Profile) else load_profile(profile)

    if roi is not None:
        image = image.crop(tuple(roi))
    rgb_img = image.convert("RGB")
    rgb = np.asarray(rgb_img, dtype=np.uint8)
    hsv = rgb_to_hsv_array(rgb)

    masks: Dict[str, np.ndarray] = {}
    for cls_ in prof.classes:
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        for band in cls_.bands:
            mask |= _band_mask(hsv, band)
        masks[cls_.name] = denoise(mask) if denoise_mask else mask
    return rgb_img, masks


def render_overlay(
    image: Image.Image,
    profile: Profile | str = "google_maps",
    *,
    roi: Sequence[int] | None = None,
    dim: float = 0.65,
    alpha: float = 0.85,
    denoise_mask: bool = True,
) -> Image.Image:
    """Dim the frame and paint each matched class in its profile colour.

    Args:
        dim: how far to fade the underlying frame (0 = black, 1 = untouched).
        alpha: opacity of the painted masks.
    """
    prof = profile if isinstance(profile, Profile) else load_profile(profile)
    base, masks = class_masks(image, prof, roi=roi, denoise_mask=denoise_mask)

    arr = np.asarray(base, dtype=np.float32)
    # Fade toward grey rather than black so unmatched roads stay legible.
    out = arr * dim + (255.0 * (1.0 - dim) * 0.25)

    for cls_ in prof.classes:
        mask = masks[cls_.name]
        if not mask.any():
            continue
        colour = np.array(_hex_to_rgb(cls_.color), dtype=np.float32)
        out[mask] = out[mask] * (1.0 - alpha) + colour * alpha

    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def save_overlay(
    src: str | Path,
    dest: str | Path,
    profile: Profile | str = "google_maps",
    **kwargs,
) -> Path:
    """Render an overlay for one file and write it to ``dest``."""
    src, dest = Path(src), Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im.load()
        render_overlay(im, profile, **kwargs).save(dest)
    return dest
