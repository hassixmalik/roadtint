"""Color profiles describing how a map renderer paints traffic.

The 2023 version of this project hardcoded a single pair of HSV ranges for
"red" and nothing else, which tied it to one renderer and one colour. A
profile makes that configurable: each severity class owns a list of HSV
bands, so a palette can be swapped without touching code.

Hue is stored in degrees (0-360) because that is how colour pickers report
it; saturation and value are 0-1. The extractor converts to whatever scale
the backend needs.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import yaml

__all__ = [
    "HSVBand",
    "SeverityClass",
    "Profile",
    "ProfileError",
    "builtin_profile_names",
    "load_profile",
]

# Profiles shipped with the package.
_BUILTIN_DIR = Path(__file__).resolve().parent / "profiles"


class ProfileError(ValueError):
    """Raised when a profile file is malformed or semantically invalid."""


@dataclass(frozen=True)
class HSVBand:
    """One contiguous HSV region.

    A band wraps around the hue circle when ``h_min > h_max``; red normally
    needs this (e.g. 345 -> 10) and getting it wrong is the single most common
    palette mistake, so it is handled here rather than by each caller.
    """

    h_min: float
    h_max: float
    s_min: float = 0.35
    s_max: float = 1.0
    v_min: float = 0.25
    v_max: float = 1.0

    def __post_init__(self) -> None:
        for name, value, hi in (
            ("h_min", self.h_min, 360.0),
            ("h_max", self.h_max, 360.0),
            ("s_min", self.s_min, 1.0),
            ("s_max", self.s_max, 1.0),
            ("v_min", self.v_min, 1.0),
            ("v_max", self.v_max, 1.0),
        ):
            if not 0.0 <= float(value) <= hi:
                raise ProfileError(f"{name}={value} out of range 0..{hi}")
        if self.s_min > self.s_max:
            raise ProfileError(f"s_min ({self.s_min}) exceeds s_max ({self.s_max})")
        if self.v_min > self.v_max:
            raise ProfileError(f"v_min ({self.v_min}) exceeds v_max ({self.v_max})")

    @property
    def wraps(self) -> bool:
        """True when this band crosses the 0/360 hue boundary."""
        return self.h_min > self.h_max

    def contains(self, h: float, s: float, v: float) -> bool:
        """Test a single HSV triple (hue in degrees, s/v in 0-1)."""
        if not (self.s_min <= s <= self.s_max and self.v_min <= v <= self.v_max):
            return False
        if self.wraps:
            return h >= self.h_min or h <= self.h_max
        return self.h_min <= h <= self.h_max

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "HSVBand":
        unknown = set(raw) - {"h_min", "h_max", "s_min", "s_max", "v_min", "v_max"}
        if unknown:
            raise ProfileError(f"unknown band keys: {sorted(unknown)}")
        if "h_min" not in raw or "h_max" not in raw:
            raise ProfileError("band requires both h_min and h_max")
        return cls(**{k: float(v) for k, v in raw.items()})

    def to_mapping(self) -> Dict[str, float]:
        return {
            "h_min": self.h_min,
            "h_max": self.h_max,
            "s_min": self.s_min,
            "s_max": self.s_max,
            "v_min": self.v_min,
            "v_max": self.v_max,
        }

    @classmethod
    def around_rgb(
        cls,
        rgb: Sequence[int],
        hue_tol: float = 8.0,
        s_tol: float = 0.30,
        v_tol: float = 0.30,
    ) -> "HSVBand":
        """Build a band centred on one RGB colour, with tolerances.

        Used by calibration to turn sampled pixels into a usable band.
        """
        r, g, b = (float(c) / 255.0 for c in rgb[:3])
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        h_deg = h * 360.0
        return cls(
            h_min=(h_deg - hue_tol) % 360.0,
            h_max=(h_deg + hue_tol) % 360.0,
            s_min=max(0.0, s - s_tol),
            s_max=min(1.0, s + s_tol),
            v_min=max(0.0, v - v_tol),
            v_max=min(1.0, v + v_tol),
        )


@dataclass(frozen=True)
class SeverityClass:
    """A named congestion level and the colours that represent it.

    ``weight`` lets callers collapse the classes into a single congestion
    index; it is advisory and not used by the mask builder.
    """

    name: str
    bands: List[HSVBand]
    weight: float = 0.0
    color: str = "#888888"

    @classmethod
    def from_mapping(cls, name: str, raw: Mapping[str, Any]) -> "SeverityClass":
        if "bands" not in raw:
            raise ProfileError(f"class '{name}' has no bands")
        bands_raw = raw["bands"]
        if not isinstance(bands_raw, list) or not bands_raw:
            raise ProfileError(f"class '{name}' bands must be a non-empty list")
        return cls(
            name=name,
            bands=[HSVBand.from_mapping(b) for b in bands_raw],
            weight=float(raw.get("weight", 0.0)),
            color=str(raw.get("color", "#888888")),
        )

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "weight": self.weight,
            "color": self.color,
            "bands": [b.to_mapping() for b in self.bands],
        }


@dataclass(frozen=True)
class Profile:
    """A full renderer palette: every severity class it can paint."""

    name: str
    classes: List[SeverityClass]
    description: str = ""
    source: Path | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if not self.classes:
            raise ProfileError(f"profile '{self.name}' defines no classes")
        seen = set()
        for c in self.classes:
            if c.name in seen:
                raise ProfileError(f"duplicate class '{c.name}' in profile '{self.name}'")
            seen.add(c.name)

    @property
    def class_names(self) -> List[str]:
        return [c.name for c in self.classes]

    def get(self, name: str) -> SeverityClass:
        for c in self.classes:
            if c.name == name:
                return c
        raise KeyError(f"profile '{self.name}' has no class '{name}'")

    # -- serialisation ---------------------------------------------------

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], source: Path | None = None) -> "Profile":
        if not isinstance(raw, Mapping):
            raise ProfileError("profile must be a mapping at the top level")
        missing = {"name", "classes"} - set(raw)
        if missing:
            raise ProfileError(f"profile missing required keys: {sorted(missing)}")
        classes_raw = raw["classes"]
        if not isinstance(classes_raw, Mapping) or not classes_raw:
            raise ProfileError("'classes' must be a non-empty mapping")
        return cls(
            name=str(raw["name"]),
            description=str(raw.get("description", "")),
            classes=[SeverityClass.from_mapping(n, c) for n, c in classes_raw.items()],
            source=source,
        )

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "classes": {c.name: c.to_mapping() for c in self.classes},
        }

    def dump_yaml(self) -> str:
        return yaml.safe_dump(self.to_mapping(), sort_keys=False)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.dump_yaml(), encoding="utf-8")
        return path


def builtin_profile_names() -> List[str]:
    """Names of the profiles bundled with the package."""
    if not _BUILTIN_DIR.is_dir():
        return []
    return sorted(p.stem for p in _BUILTIN_DIR.glob("*.yaml"))


def load_profile(ref: str | Path) -> Profile:
    """Load a profile by builtin name or filesystem path.

    A bare name resolves against the bundled profiles first, so
    ``load_profile("google_maps")`` works without knowing the install
    location; anything containing a separator or ending in .yaml/.yml is
    treated as a path.
    """
    text_ref = str(ref)
    candidate = Path(text_ref)
    looks_like_path = (
        candidate.suffix.lower() in {".yaml", ".yml"}
        or "/" in text_ref
        or "\\" in text_ref
    )

    if not looks_like_path:
        builtin = _BUILTIN_DIR / f"{text_ref}.yaml"
        if builtin.is_file():
            candidate = builtin
        else:
            available = builtin_profile_names()
            raise ProfileError(
                f"unknown profile '{text_ref}'. Available: {', '.join(available) or 'none'}"
            )

    if not candidate.is_file():
        raise ProfileError(f"profile file not found: {candidate}")

    try:
        raw = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ProfileError(f"invalid YAML in {candidate}: {exc}") from exc

    return Profile.from_mapping(raw, source=candidate)


def iter_bands(profile: Profile) -> Iterable[tuple[str, HSVBand]]:
    """Yield ``(class_name, band)`` for every band in the profile."""
    for cls_ in profile.classes:
        for band in cls_.bands:
            yield cls_.name, band
