# Changelog

## 0.1.0

First release as a general tool. The repository previously held a 2023
university lab notebook; this is a rewrite, not an increment.

### Added

- **`roadtint` CLI** with `extract`, `series`, `calibrate`, `dedup`,
  `video` and `profiles` subcommands.
- **Configurable colour profiles** (YAML) for Google Maps, Mapbox and
  HERE, plus `calibrate` to generate one for any other renderer.
- **Share-based measurement.** Each severity class is reported as a
  share of the road detected in that frame, so results are comparable
  between captures at different zoom levels.
- **Length-based measurement** (default). Masks are thinned to a centre
  line before measuring, removing the stroke-width bias that makes wide
  roads dominate. `--measure area` keeps the old behaviour.
- **Duplicate detection** by content hash and a perceptual hash computed
  over saturation.
- **Time series** output to CSV and JSON, ordered by timestamps parsed
  from filenames, with a warning for frames whose road coverage falls
  well below the rest of the set.
- **Overlay rendering** (`--overlay`) to check a profile visually.
- **Video input** via the optional `video` extra.
- Test suite of ~190 tests using only synthetic imagery, CI across
  Python 3.9-3.13 on Linux, macOS and Windows.

### Changed from the 2023 notebook

- Contour counting is gone. `len(contours)` of a red mask varied with
  zoom and overlay fragmentation, so it could not be compared between
  frames.
- Orange is measured. The original looked only for red, which is scarce
  in the sample captures (0.00-0.2% of pixels) while orange covers 2-4%
  — the dominant congestion signal was being discarded.
- The README no longer claims a genetic algorithm optimises the
  morphological kernel. No such code existed; the kernel was a
  hardcoded 5x5.

### Known limitations

See the Limitations section of the README. In short: absent traffic data
reads as absent rather than free; shares are relative to visible road;
basemap colours can collide with overlay colours; road length is
estimated by thinning rather than traced; strokes below 3px are removed
with the speckle.
