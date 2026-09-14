# roadtint

Quantify road congestion from traffic-layer map imagery.

Point it at screenshots, exported map tiles, or a screen recording of a
traffic layer, and get numbers you can actually compare: what share of the
visible road is free-flowing, slow, congested or blocked, as CSV or JSON.

```console
$ roadtint series ./captures --csv morning.csv
09:00  0.166  #######
09:30  0.123  #####
11:30  0.222  #########
14:30  0.352  ##############
17:00  0.171  #######

16 frames (4 duplicate(s) dropped)  mean=0.196  peak=0.352 at 14:30
```

## Why this exists

Lots of traffic data is locked up in pictures — a map service with no API
in your region, an archived screenshot, a dashcam frame, a recording of a
traffic dashboard. The colour is right there on the road; the hard part is
turning it into a number that means the same thing in two different frames.

That is the whole job here.

## What makes the numbers comparable

**Shares, not counts.** Counting coloured blobs sounds reasonable and does
not work: the count scales with zoom level and with how badly labels and
junctions chop the overlay into pieces. Two captures of the same road at
different zoom report different numbers for identical traffic. roadtint
reports each severity class as a *share of the road detected in that
frame*, which stays put when the crop or zoom changes.

**Length, not area.** Measuring coloured area over-weights wide strokes: a
motorway drawn 8px wide outranks a side street drawn 3px wide even when
both are equally long. By default each mask is thinned to a one-pixel
centre line first, so roads contribute in proportion to their length.
Pass `--measure area` if you want raw coverage instead.

**Duplicates removed.** Repeated frames quietly flatten a time series.
`roadtint dedup` finds them, and `series` drops them by default.

## Install

```bash
pip install -e .
```

Core install needs only `numpy`, `pillow` and `pyyaml`. Optional extras:

```bash
pip install -e ".[video]"   # video input (opencv)
pip install -e ".[plot]"    # chart output (matplotlib)
pip install -e ".[dev]"     # tests and linting
```

Python 3.9+.

## Usage

### Measure some frames

```bash
roadtint extract ./captures
roadtint extract frame.png --json
```

### Build a time series

```bash
roadtint series ./captures --csv out.csv --json-out out.json
```

Timestamps are read from filenames where possible (`08_00am.png`,
`1430.png`, `cap-14-30.png`). Frames without a recognisable time keep
their input order and are listed last.

### Work with your own renderer

The bundled profiles cover Google Maps, Mapbox and HERE. If your map looks
different — a custom style, a dark theme, a regional provider — build your
own profile instead of fighting the defaults.

First, see what is actually in your image:

```console
$ roadtint calibrate my-screenshot.png
Dominant colours in my-screenshot.png:

  1. hue   25.3deg  [ 20.0- 30.0]  71.30% of saturated px  rgb(253,150,77)  #fd964d
  2. hue  122.6deg  [120.0-125.0]  12.58% of saturated px  rgb(99,214,104)  #63d668
  3. hue    3.2deg  [  0.0-  5.0]  10.41% of saturated px  rgb(242,60,50)  #f23c32
  4. hue  230.5deg  [230.0-235.0]   1.63% of saturated px  rgb(121,134,203)  #7986cb
```

Then sample one pixel per class and save a profile:

```bash
roadtint calibrate my-screenshot.png \
  --point free=410,220 \
  --point slow=180,340 \
  --point congested=295,118 \
  --name my_city -o my_city.yaml

roadtint extract ./captures -p my_city.yaml
```

### Check a profile by eye

Numbers look plausible long before they are right. Render what actually
matched:

```bash
roadtint extract frame.png --overlay ./overlays
```

Roads should be painted in their class colour; parkland, water, labels and
map pins should not be.

### Read a video

```bash
roadtint video recording.mp4 --every 60 --csv out.csv
```

Needs the `video` extra. Frames are labelled by elapsed video time.

### Find duplicate frames

```bash
roadtint dedup ./captures
```

### Exclude map furniture

Pins, legends and watermarks are coloured too. Crop them out:

```bash
roadtint series ./captures --roi 0,60,1200,780
```

## Profiles

A profile maps severity classes to HSV bands:

```yaml
name: my_city
classes:
  free:
    weight: 0.0
    bands:
      - { h_min: 110, h_max: 140, s_min: 0.35, v_min: 0.30 }
  congested:
    weight: 1.0
    bands:
      # Red wraps the hue origin - h_min > h_max is intentional.
      - { h_min: 345, h_max: 12, s_min: 0.45, v_min: 0.35 }
```

`weight` drives the single-number `congestion_index`; the class names are
yours to choose. Red is the usual trap: its hue straddles 0/360, so a band
written `h_min: 0, h_max: 12` silently misses half the pixels. Write it as
`345 -> 12` and the wrap is handled for you.

## Python API

```python
from roadtint import load_profile
from roadtint.extract import extract_file
from roadtint.series import build_series, write_csv

m = extract_file("14_30.png", "google_maps")
print(m.congestion_index, m.shares)

series = build_series(sorted(Path("captures").glob("*.png")))
write_csv(series, "out.csv")
print(series.summary())
```

## Live data

roadtint is offline by design: it reads images you already have, so there
are no API keys, rate limits or provider terms to worry about in the core.

If you want to collect frames automatically, write a small fetcher that
saves images to a folder on whatever schedule you like, then point
`roadtint series` at that folder. See [docs/fetchers.md](docs/fetchers.md)
for the interface and a worked example.

## Limitations

Worth knowing before you trust the output:

- **It measures what the renderer drew.** If the map has no traffic data
  for a road, that road is just absent from the numbers — not free-flowing.
- **Shares are relative.** A frame where every road is red reports
  `congested ≈ 1.0`, the same as a frame with one red street and nothing
  else visible. Read `road_coverage` alongside it.
- **Basemap colour can collide with overlay colour.** Green parkland under
  a green "free" band inflates the free share. Use `--roi`, or tighten the
  saturation floor in your profile.
- **Length is estimated, not traced.** Thinning approximates centre-line
  length by area over mean stroke width. Good enough to compare frames;
  not a substitute for real road-network geometry.
- **Thin roads are dropped.** Strokes of 1–2px are removed by the
  denoising pass along with the speckle; 3px and wider survive. Capture
  at a zoom where the roads you care about are clearly drawn, or pass
  `denoise_mask=False` through the Python API.

`roadtint series` warns when a frame's road coverage falls well below the
rest of the set, which usually means the capture drifted or the overlay
failed to load.

## History

This started in May 2023 as a university AI lab submission that counted red
contours in Google Maps screenshots of Islamabad and printed a "number of
blockage points". That number was not comparable between frames, and the
accompanying README described a genetic algorithm that was not present in
the code.

It was rebuilt in 2026 as a general tool. The original screenshots are kept
under [examples/](examples/) as a case study, including the duplicate files
that motivated `roadtint dedup`.

## Contributing

Issues and pull requests welcome — especially profiles for renderers not
covered yet. Tests use synthetic images only, so you can run the suite
without any third-party map data:

```bash
pytest
```

## License

MIT — see [LICENSE](LICENSE).

Sample map screenshots under `examples/` are **not** covered by the MIT
licence. They are derived from third-party map services and are included
for documentation only. See [examples/README.md](examples/README.md).
