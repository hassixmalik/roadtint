"""Detect duplicate and near-duplicate frames.

Motivating case: the original sample set in this repository advertises 19
half-hourly observations, but seven of the files are byte-identical copies
of three images (10:00/10:30/11:00, 12:00/12:30, 13:00/13:30). Treating
those as independent readings inflates the apparent size of the dataset and
silently flattens the time series.

Exact duplicates are caught by content hash. Near-duplicates - a frame
re-screenshotted a second later, or the same scene nudged by a pixel - need
a perceptual hash, here the classic difference hash (dHash): downscale to a
small grid and record whether each pixel is brighter than its right-hand
neighbour. Resilient to small shifts and compression, cheap to compare by
Hamming distance.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image

__all__ = ["DuplicateGroup", "FrameHash", "dhash", "find_duplicates", "hamming", "hash_file"]


@dataclass(frozen=True)
class FrameHash:
    path: Path
    content_md5: str
    phash: int

    @property
    def name(self) -> str:
        return self.path.name


@dataclass
class DuplicateGroup:
    """A set of frames considered the same image."""

    members: List[Path] = field(default_factory=list)
    exact: bool = False

    @property
    def keeper(self) -> Path:
        """The member to keep: the first in sorted order, for stability."""
        return sorted(self.members)[0]

    @property
    def redundant(self) -> List[Path]:
        return [p for p in sorted(self.members) if p != self.keeper]

    def describe(self) -> str:
        kind = "identical" if self.exact else "near-identical"
        names = ", ".join(sorted(p.name for p in self.members))
        return f"{kind}: {names}"


def dhash(image: Image.Image | str | Path, hash_size: int = 8) -> int:
    """Perceptual hash of an image, as an integer of ``4 * hash_size**2`` bits.

    Classic dHash compares each pixel only with its right-hand neighbour,
    which makes it blind to purely horizontal structure: a frame containing
    one long horizontal road hashes to all zeros, and every such frame
    collides with every other regardless of colour or position. Road
    networks contain exactly that kind of structure, so this adds a
    vertical pass.

    Greyscale also throws away the thing we care most about here - a road
    painted green and the same road painted red are the same shape. A
    third block hashes the per-cell hue, so recolouring changes the hash.
    """
    if isinstance(image, (str, Path)):
        with Image.open(image) as im:
            im.load()
            return dhash(im, hash_size)

    rgb = image.convert("RGB")

    # Gradients are computed over SATURATION, not luminance.
    #
    # A traffic overlay is defined by being colourful, not by being bright:
    # a green road at luminance 124 over a grey basemap at 128 is almost
    # invisible in greyscale, and after an 8x8 downscale the whole frame
    # flattens to a spread of 1. Frames with roads in completely different
    # places then hash within a bit of each other. In saturation the same
    # road is unmissable.
    arr_full = np.asarray(rgb, dtype=np.float32)
    mx = arr_full.max(axis=-1)
    mn = arr_full.min(axis=-1)
    sat_full = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    sat_img = Image.fromarray((sat_full * 255).astype(np.uint8), mode="L")

    # A dead-band keeps near-equal cells from flipping under recompression,
    # scaled to the frame's own range so a low-contrast frame is not
    # flattened entirely.
    spread = int(np.asarray(sat_img).max()) - int(np.asarray(sat_img).min())
    grad_margin = max(1, int(spread * 0.05))

    # Horizontal gradient: one extra column -> hash_size comparisons per row.
    h_small = sat_img.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    h_diff = np.asarray(h_small, dtype=np.int16)
    h_bits = (h_diff[:, 1:] > h_diff[:, :-1] + grad_margin).flatten()

    # Vertical gradient: one extra row.
    v_small = sat_img.resize((hash_size, hash_size + 1), Image.Resampling.LANCZOS)
    v_diff = np.asarray(v_small, dtype=np.int16)
    v_bits = (v_diff[1:, :] > v_diff[:-1, :] + grad_margin).flatten()

    # Colour signature: hue of each cell, above/below the frame median.
    c_small = rgb.resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    arr = np.asarray(c_small, dtype=np.float32) / 255.0
    cmax = arr.max(axis=-1)
    cmin = arr.min(axis=-1)
    delta = cmax - cmin
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    hue = np.zeros_like(cmax)
    nz = delta > 1e-12
    with np.errstate(invalid="ignore", divide="ignore"):
        rm = nz & (cmax == r)
        gm = nz & (cmax == g) & ~rm
        bm = nz & (cmax == b) & ~rm & ~gm
        hue[rm] = ((g[rm] - b[rm]) / delta[rm]) % 6.0
        hue[gm] = ((b[gm] - r[gm]) / delta[gm]) + 2.0
        hue[bm] = ((r[bm] - g[bm]) / delta[bm]) + 4.0
    # Channel-dominance bits. Comparing channels against each other is
    # stable under recompression (a red cell stays red-dominant at JPEG
    # q55) while still separating a green road from a red one, which a
    # greyscale-only hash cannot do. A margin keeps near-grey cells, where
    # the ordering is arbitrary, from flipping on noise.
    margin = 0.06
    c_bits = np.concatenate(
        [
            (r > g + margin).flatten(),
            (g > b + margin).flatten(),
        ]
    )

    value = 0
    for bit in np.concatenate([h_bits, v_bits, c_bits]):
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    """Number of differing bits between two hashes."""
    return bin(a ^ b).count("1")


def hash_file(path: str | Path, hash_size: int = 8) -> FrameHash:
    path = Path(path)
    data = path.read_bytes()
    with Image.open(path) as im:
        im.load()
        ph = dhash(im, hash_size)
    return FrameHash(path=path, content_md5=hashlib.md5(data).hexdigest(), phash=ph)


def find_duplicates(
    paths: Iterable[str | Path],
    *,
    threshold: int = 4,
    hash_size: int = 8,
) -> List[DuplicateGroup]:
    """Group frames that are the same image.

    Args:
        paths: image files to compare.
        threshold: maximum Hamming distance still considered a duplicate.
            0 means perceptually identical; 4 tolerates light recompression.

    Returns groups with two or more members, largest first.
    """
    hashes: List[FrameHash] = []
    for p in paths:
        try:
            hashes.append(hash_file(p, hash_size))
        except (OSError, ValueError):
            # Unreadable files are skipped rather than aborting a batch.
            continue

    groups: List[DuplicateGroup] = []
    claimed: set[Path] = set()

    # Exact content matches first - cheaper and unambiguous.
    by_md5: Dict[str, List[FrameHash]] = {}
    for h in hashes:
        by_md5.setdefault(h.content_md5, []).append(h)
    for bucket in by_md5.values():
        if len(bucket) > 1:
            groups.append(DuplicateGroup(members=[h.path for h in bucket], exact=True))
            claimed.update(h.path for h in bucket)

    # Then perceptual near-matches among whatever is left.
    remaining = [h for h in hashes if h.path not in claimed]
    for i, h in enumerate(remaining):
        if h.path in claimed:
            continue
        cluster = [h.path]
        for other in remaining[i + 1 :]:
            if other.path in claimed:
                continue
            if hamming(h.phash, other.phash) <= threshold:
                cluster.append(other.path)
                claimed.add(other.path)
        if len(cluster) > 1:
            claimed.add(h.path)
            groups.append(DuplicateGroup(members=cluster, exact=False))

    groups.sort(key=lambda g: len(g.members), reverse=True)
    return groups


def unique_paths(
    paths: Sequence[str | Path],
    *,
    threshold: int = 4,
) -> Tuple[List[Path], List[DuplicateGroup]]:
    """Return ``(kept, groups)`` with one representative per duplicate set."""
    ordered = [Path(p) for p in paths]
    groups = find_duplicates(ordered, threshold=threshold)
    drop = {p for g in groups for p in g.redundant}
    return [p for p in ordered if p not in drop], groups
