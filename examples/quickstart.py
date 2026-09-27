#!/usr/bin/env python3
"""Measure the bundled Islamabad capture set and print a daily profile.

Run from the repository root:

    python examples/quickstart.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from roadtint.series import build_series, write_csv

CAPTURES = Path(__file__).parent / "islamabad-f8"


def main() -> int:
    frames = sorted(CAPTURES.glob("*.png"))
    if not frames:
        print(f"no frames in {CAPTURES}", file=sys.stderr)
        print("(the sample imagery may have been removed from this fork)", file=sys.stderr)
        return 1

    series = build_series(frames, "google_maps")
    summary = series.summary()

    print(f"{len(frames)} files -> {summary['frames']} distinct frames")
    if summary["duplicates_removed"]:
        print(f"({summary['duplicates_removed']} duplicate(s) dropped)")
    print()

    for point in series:
        m = point.measurement
        bar = "#" * round(m.congestion_index * 50)
        print(
            f"{point.label:<12} {m.congestion_index:.3f} "
            f"free={m.shares['free']:.2f} slow={m.shares['slow']:.2f} "
            f"cong={m.shares['congested']:.2f}  {bar}"
        )

    print()
    print(f"busiest:  {summary['peak_at']} ({summary['max_index']:.3f})")
    print(f"quietest: {summary['quietest_at']} ({summary['min_index']:.3f})")

    if summary["low_coverage_frames"]:
        print(
            f"\nlow road coverage in: {', '.join(summary['low_coverage_frames'])}"
            "\n(these shares rest on very few pixels - treat with care)"
        )

    out = write_csv(series, "islamabad.csv")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
