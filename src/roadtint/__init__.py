"""roadtint - quantify road congestion from traffic-layer map imagery.

Typical use:

    from roadtint import extract_file, build_series

    m = extract_file("14_30.png")
    print(m.congestion_index, m.shares)

Video support lives in ``roadtint.video`` and is imported separately so
that the still-image path does not require opencv.
"""

from .calibrate import profile_from_samples, suggest_palette
from .dedup import find_duplicates, unique_paths
from .extract import ExtractionError, Measurement, extract, extract_file
from .overlay import render_overlay, save_overlay
from .profile import (
    HSVBand,
    Profile,
    ProfileError,
    SeverityClass,
    builtin_profile_names,
    load_profile,
)
from .series import Series, TimePoint, build_series, write_csv, write_json

__version__ = "0.1.0"

__all__ = [
    # profiles
    "HSVBand",
    "Profile",
    "ProfileError",
    "SeverityClass",
    "builtin_profile_names",
    "load_profile",
    # measurement
    "ExtractionError",
    "Measurement",
    "extract",
    "extract_file",
    # calibration
    "profile_from_samples",
    "suggest_palette",
    # batching
    "Series",
    "TimePoint",
    "build_series",
    "write_csv",
    "write_json",
    # duplicates
    "find_duplicates",
    "unique_paths",
    # visual check
    "render_overlay",
    "save_overlay",
    "__version__",
]
