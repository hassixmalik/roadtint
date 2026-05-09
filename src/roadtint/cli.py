"""Command line interface.

Subcommands:
    extract    measure one or more frames
    series     batch a folder into a time series (CSV/JSON)
    calibrate  inspect an image's palette, or build a profile from it
    dedup      report duplicate frames
    profiles   list bundled profiles
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

from . import __version__
from .calibrate import profile_from_samples, suggest_palette
from .dedup import find_duplicates
from .extract import ExtractionError, extract_file
from .overlay import save_overlay
from .profile import ProfileError, builtin_profile_names, load_profile
from .series import build_series, write_csv, write_json
from .video import VideoUnavailable, series_from_video

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _collect_images(inputs: Sequence[str], recursive: bool = False) -> List[Path]:
    """Expand files and directories into a sorted list of image paths."""
    found: List[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            globber = p.rglob("*") if recursive else p.glob("*")
            found.extend(
                f for f in globber if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES
            )
        elif p.is_file():
            found.append(p)
        else:
            print(f"warning: no such path: {p}", file=sys.stderr)
    return sorted(set(found))


def _parse_roi(value: str | None) -> tuple[int, int, int, int] | None:
    if not value:
        return None
    parts = value.split(",")
    if len(parts) != 4:
        raise SystemExit("--roi needs four comma-separated integers: left,top,right,bottom")
    try:
        left, top, right, bottom = (int(p.strip()) for p in parts)
    except ValueError:
        raise SystemExit("--roi values must be integers") from None
    if right <= left or bottom <= top:
        raise SystemExit("--roi must satisfy left<right and top<bottom")
    return left, top, right, bottom


def _parse_point(value: str) -> tuple[str, int, int]:
    """Parse ``class=x,y``."""
    if "=" not in value:
        raise SystemExit(f"--point needs the form class=x,y (got {value!r})")
    name, coords = value.split("=", 1)
    parts = coords.split(",")
    if len(parts) != 2:
        raise SystemExit(f"--point needs the form class=x,y (got {value!r})")
    try:
        return name.strip(), int(parts[0]), int(parts[1])
    except ValueError:
        raise SystemExit(f"--point coordinates must be integers (got {value!r})") from None


# -- commands -----------------------------------------------------------


def cmd_extract(args: argparse.Namespace) -> int:
    images = _collect_images(args.inputs, args.recursive)
    if not images:
        print("no images found", file=sys.stderr)
        return 1

    roi = _parse_roi(args.roi)
    rows = []
    for path in images:
        try:
            m = extract_file(path, args.profile, measure=args.measure, roi=roi)
        except (ExtractionError, ProfileError) as exc:
            print(f"error: {path.name}: {exc}", file=sys.stderr)
            if args.strict:
                return 1
            continue
        rows.append(m.to_row())

        if args.overlay:
            try:
                dest = Path(args.overlay) / f"{path.stem}.overlay.png"
                save_overlay(path, dest, args.profile, roi=roi)
            except (ExtractionError, ProfileError, OSError) as exc:
                print(f"warning: overlay for {path.name}: {exc}", file=sys.stderr)

        if not args.json:
            shares = "  ".join(f"{k}={v:.3f}" for k, v in m.shares.items())
            print(f"{path.name:<22} index={m.congestion_index:.3f}  {shares}")

    if args.json:
        print(json.dumps(rows, indent=2))
    return 0 if rows else 1


def cmd_series(args: argparse.Namespace) -> int:
    images = _collect_images(args.inputs, args.recursive)
    if not images:
        print("no images found", file=sys.stderr)
        return 1

    try:
        series = build_series(
            images,
            args.profile,
            measure=args.measure,
            roi=_parse_roi(args.roi),
            drop_duplicates=not args.keep_duplicates,
        )
    except (ExtractionError, ProfileError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.csv:
        print(f"wrote {write_csv(series, args.csv)}")
    if args.json_out:
        print(f"wrote {write_json(series, args.json_out)}")

    if not (args.csv or args.json_out):
        width = max((len(p.label) for p in series), default=5)
        for p in series:
            m = p.measurement
            bar = "#" * int(round(m.congestion_index * 40))
            print(f"{p.label:<{width}}  {m.congestion_index:.3f}  {bar}")

    s = series.summary()
    if s.get("frames"):
        print(
            f"\n{s['frames']} frames"
            + (f" ({s['duplicates_removed']} duplicate(s) dropped)" if s["duplicates_removed"] else "")
            + f"  mean={s['mean_index']:.3f}"
            f"  peak={s['max_index']:.3f} at {s['peak_at']}"
            f"  quietest={s['min_index']:.3f} at {s['quietest_at']}"
        )
    return 0


def cmd_video(args: argparse.Namespace) -> int:
    try:
        series = series_from_video(
            args.video,
            args.profile,
            every_seconds=args.every,
            measure=args.measure,
            roi=_parse_roi(args.roi),
            max_frames=args.max_frames,
        )
    except (VideoUnavailable, ExtractionError, ProfileError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not len(series):
        print("no frames sampled", file=sys.stderr)
        return 1

    if args.csv:
        print(f"wrote {write_csv(series, args.csv)}")
    if args.json_out:
        print(f"wrote {write_json(series, args.json_out)}")

    if not (args.csv or args.json_out):
        for p in series:
            m = p.measurement
            bar = "#" * int(round(m.congestion_index * 40))
            print(f"{p.label:<34}  {m.congestion_index:.3f}  {bar}")

    s = series.summary()
    print(
        f"\n{s['frames']} frames  mean={s['mean_index']:.3f}"
        f"  peak={s['max_index']:.3f} at {s['peak_at']}"
    )
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    image = Path(args.image)
    if not image.is_file():
        print(f"no such image: {image}", file=sys.stderr)
        return 1

    if args.point:
        samples = {}
        for raw in args.point:
            name, x, y = _parse_point(raw)
            samples[name] = (x, y)
        try:
            prof = profile_from_samples(image, samples, name=args.name)
        except ProfileError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if args.out:
            print(f"wrote {prof.save(args.out)}")
        else:
            print(prof.dump_yaml())
        return 0

    clusters = suggest_palette(image)
    if not clusters:
        print("no saturated colours found - is this a traffic overlay?", file=sys.stderr)
        return 1
    print(f"Dominant colours in {image.name}:\n")
    for i, c in enumerate(clusters, 1):
        print(f"  {i}. {c.describe()}")
    print(
        "\nPick the ones that are roads, then build a profile:\n"
        f"  roadtint calibrate {image} --point free=X,Y --point congested=X,Y -o my.yaml"
    )
    return 0


def cmd_dedup(args: argparse.Namespace) -> int:
    images = _collect_images(args.inputs, args.recursive)
    if not images:
        print("no images found", file=sys.stderr)
        return 1

    groups = find_duplicates(images, threshold=args.threshold)
    if not groups:
        print(f"{len(images)} frames, no duplicates found")
        return 0

    redundant = 0
    for g in groups:
        print(g.describe())
        print(f"    keep: {g.keeper.name}")
        redundant += len(g.redundant)
    print(f"\n{len(images)} frames -> {len(images) - redundant} distinct")
    return 0


def cmd_profiles(args: argparse.Namespace) -> int:
    names = builtin_profile_names()
    if not names:
        print("no bundled profiles found", file=sys.stderr)
        return 1
    for name in names:
        prof = load_profile(name)
        desc = " ".join(prof.description.split())
        print(f"{name}")
        print(f"    classes: {', '.join(prof.class_names)}")
        if desc:
            print(f"    {desc[:100]}{'...' if len(desc) > 100 else ''}")
    return 0


# -- parser -------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="roadtint",
        description="Quantify road congestion from traffic-layer map imagery.",
    )
    parser.add_argument("--version", action="version", version=f"roadtint {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("-p", "--profile", default="google_maps",
                       help="builtin profile name or path to a YAML profile")
        p.add_argument("-m", "--measure", choices=["length", "area"], default="length",
                       help="length (default) thins roads to a centre line first")
        p.add_argument("--roi", help="crop as left,top,right,bottom")
        p.add_argument("-r", "--recursive", action="store_true",
                       help="recurse into subdirectories")

    pe = sub.add_parser("extract", help="measure one or more frames")
    pe.add_argument("inputs", nargs="+", help="image files or directories")
    pe.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    pe.add_argument("--strict", action="store_true", help="stop on the first unreadable frame")
    pe.add_argument("--overlay", metavar="DIR",
                    help="also write an annotated overlay per frame into DIR")
    add_common(pe)
    pe.set_defaults(func=cmd_extract)

    ps = sub.add_parser("series", help="batch frames into a time series")
    ps.add_argument("inputs", nargs="+", help="image files or directories")
    ps.add_argument("--csv", help="write CSV here")
    ps.add_argument("--json-out", dest="json_out", help="write JSON here")
    ps.add_argument("--keep-duplicates", action="store_true",
                    help="do not drop repeated frames")
    add_common(ps)
    ps.set_defaults(func=cmd_series)

    pv = sub.add_parser("video", help="sample a video into a time series")
    pv.add_argument("video", help="video file")
    pv.add_argument("--every", type=float, default=60.0,
                    help="seconds of video between samples (default 60)")
    pv.add_argument("--max-frames", dest="max_frames", type=int,
                    help="stop after this many samples")
    pv.add_argument("--csv", help="write CSV here")
    pv.add_argument("--json-out", dest="json_out", help="write JSON here")
    pv.add_argument("-p", "--profile", default="google_maps")
    pv.add_argument("-m", "--measure", choices=["length", "area"], default="length")
    pv.add_argument("--roi", help="crop as left,top,right,bottom")
    pv.set_defaults(func=cmd_video)

    pc = sub.add_parser("calibrate", help="inspect a palette or build a profile")
    pc.add_argument("image", help="a screenshot from your renderer")
    pc.add_argument("--point", action="append", metavar="CLASS=X,Y",
                    help="sample a class colour at a pixel; repeatable")
    pc.add_argument("--name", default="custom", help="name for the generated profile")
    pc.add_argument("-o", "--out", help="write the profile here instead of stdout")
    pc.set_defaults(func=cmd_calibrate)

    pd = sub.add_parser("dedup", help="report duplicate frames")
    pd.add_argument("inputs", nargs="+", help="image files or directories")
    pd.add_argument("-t", "--threshold", type=int, default=4,
                    help="max Hamming distance treated as duplicate (default 4)")
    pd.add_argument("-r", "--recursive", action="store_true")
    pd.set_defaults(func=cmd_dedup)

    pp = sub.add_parser("profiles", help="list bundled profiles")
    pp.set_defaults(func=cmd_profiles)

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130
    except (ExtractionError, ProfileError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
