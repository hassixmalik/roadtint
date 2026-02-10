"""Batch a folder of frames (or a video) into a congestion time series.

A single measurement is rarely the interesting thing. What people want is
how congestion moves across a morning, a week, or a recording - so this
turns many frames into one ordered table.

Timestamps come from filenames where possible. The sample set here is named
``08_00am.png`` / ``04_30pm.png``, which is a common enough convention to
support directly, but the parser is a list of patterns rather than one
hardcoded format so other layouts can be added without touching callers.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

from .dedup import unique_paths
from .extract import Measurement, extract_file
from .profile import Profile

__all__ = [
    "TimePoint",
    "Series",
    "parse_time_from_name",
    "build_series",
    "write_csv",
    "write_json",
]

# Ordered patterns; first match wins.
_TIME_PATTERNS: List[tuple[re.Pattern, Callable[[re.Match], time]]] = [
    # 08_00am, 12_30pm, 4_05pm
    (
        re.compile(r"(?P<h>\d{1,2})[_\-.](?P<m>\d{2})\s*(?P<ap>am|pm)", re.I),
        lambda m: _mk_time(int(m["h"]), int(m["m"]), m["ap"]),
    ),
    # 08am, 5pm
    (
        re.compile(r"(?P<h>\d{1,2})\s*(?P<ap>am|pm)", re.I),
        lambda m: _mk_time(int(m["h"]), 0, m["ap"]),
    ),
    # 24-hour: 0800, 1430 (anchored so it does not eat a date)
    (
        re.compile(r"(?<!\d)(?P<h>[01]\d|2[0-3])(?P<m>[0-5]\d)(?!\d)"),
        lambda m: time(int(m["h"]), int(m["m"])),
    ),
    # 14-30, 14_30, 14:30
    (
        re.compile(r"(?<!\d)(?P<h>[01]?\d|2[0-3])[_\-:](?P<m>[0-5]\d)(?!\d)"),
        lambda m: time(int(m["h"]), int(m["m"])),
    ),
]


def _mk_time(hour: int, minute: int, ampm: str) -> time:
    ampm = ampm.lower()
    if not 1 <= hour <= 12:
        raise ValueError(f"hour {hour} invalid for 12-hour clock")
    if ampm == "am":
        hour = 0 if hour == 12 else hour
    else:
        hour = 12 if hour == 12 else hour + 12
    return time(hour, minute)


def parse_time_from_name(name: str) -> time | None:
    """Best-effort clock time from a filename, or None.

    Note the sample set labels midday as ``12_00am``, which strictly means
    midnight. We honour the literal reading rather than guessing; a caller
    who knows better can pass explicit timestamps.
    """
    stem = Path(name).stem
    for pattern, build in _TIME_PATTERNS:
        m = pattern.search(stem)
        if m:
            try:
                return build(m)
            except ValueError:
                continue
    return None


@dataclass
class TimePoint:
    """One measurement, optionally placed on a clock."""

    measurement: Measurement
    timestamp: time | datetime | None = None

    @property
    def label(self) -> str:
        if isinstance(self.timestamp, datetime):
            return self.timestamp.isoformat(sep=" ", timespec="minutes")
        if isinstance(self.timestamp, time):
            return self.timestamp.strftime("%H:%M")
        return self.measurement.source

    def to_row(self) -> Dict[str, Any]:
        row = {"label": self.label, "timestamp": self.label if self.timestamp else ""}
        row.update(self.measurement.to_row())
        return row


@dataclass
class Series:
    """An ordered set of measurements."""

    points: List[TimePoint]
    profile: str = ""
    measure: str = ""
    #: duplicate groups removed during the build, for reporting
    duplicates_removed: int = 0

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self):
        return iter(self.points)

    def rows(self) -> List[Dict[str, Any]]:
        return [p.to_row() for p in self.points]

    def column(self, key: str) -> List[float]:
        """Pull one numeric column, e.g. ``"congestion_index"``."""
        out = []
        for p in self.points:
            row = p.to_row()
            if key not in row:
                raise KeyError(f"no column '{key}'; have {sorted(row)}")
            out.append(float(row[key]))
        return out

    def peak(self) -> TimePoint | None:
        """The most congested point."""
        if not self.points:
            return None
        return max(self.points, key=lambda p: p.measurement.congestion_index)

    def quietest(self) -> TimePoint | None:
        if not self.points:
            return None
        return min(self.points, key=lambda p: p.measurement.congestion_index)

    def summary(self) -> Dict[str, Any]:
        if not self.points:
            return {"frames": 0}
        idx = [p.measurement.congestion_index for p in self.points]
        peak, quiet = self.peak(), self.quietest()
        return {
            "frames": len(self.points),
            "duplicates_removed": self.duplicates_removed,
            "mean_index": round(sum(idx) / len(idx), 4),
            "min_index": round(min(idx), 4),
            "max_index": round(max(idx), 4),
            "peak_at": peak.label if peak else None,
            "quietest_at": quiet.label if quiet else None,
        }


def build_series(
    paths: Iterable[str | Path],
    profile: Profile | str = "google_maps",
    *,
    measure: str = "length",
    roi: Sequence[int] | None = None,
    drop_duplicates: bool = True,
    sort_by_time: bool = True,
    timestamps: Dict[str, time] | None = None,
) -> Series:
    """Measure many frames and order them into a series.

    Args:
        paths: image files.
        drop_duplicates: remove repeated frames before measuring, so a
            copy-pasted file does not read as a real observation.
        sort_by_time: order by parsed timestamp where available.
        timestamps: explicit ``{filename: time}`` overrides.
    """
    ordered = [Path(p) for p in paths]
    removed = 0
    if drop_duplicates and len(ordered) > 1:
        kept, groups = unique_paths(ordered)
        removed = sum(len(g.redundant) for g in groups)
        ordered = kept

    overrides = timestamps or {}
    points: List[TimePoint] = []
    for path in ordered:
        m = extract_file(path, profile, measure=measure, roi=roi)
        ts = overrides.get(path.name) or parse_time_from_name(path.name)
        points.append(TimePoint(measurement=m, timestamp=ts))

    if sort_by_time:
        # Frames without a parseable time keep their input order, after
        # the timed ones, rather than being dropped.
        timed = [p for p in points if p.timestamp is not None]
        untimed = [p for p in points if p.timestamp is None]
        timed.sort(key=lambda p: (p.timestamp.hour, p.timestamp.minute))
        points = timed + untimed

    prof_name = points[0].measurement.profile if points else ""
    return Series(
        points=points,
        profile=prof_name,
        measure=measure,
        duplicates_removed=removed,
    )


def write_csv(series: Series, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = series.rows()
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    # Union of keys, stable order, so a profile with extra classes still
    # produces a rectangular file.
    fields: List[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    return path


def write_json(series: Series, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "profile": series.profile,
        "measure": series.measure,
        "summary": series.summary(),
        "points": series.rows(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
