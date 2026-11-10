# Contributing

Thanks for taking a look. Issues and pull requests are welcome.

## Getting set up

```bash
git clone https://github.com/hassixmalik/roadtint.git
cd roadtint
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,video]"
pytest
```

The suite uses only synthetic images, so it runs without any third-party
map data. If a test needs a picture, draw it in `tests/conftest.py` rather
than committing a screenshot.

## Before you open a PR

```bash
pytest
ruff check src tests
```

CI runs both across Python 3.9–3.13 on Linux, macOS and Windows.

## Adding a renderer profile

This is the most useful contribution and the easiest to get right.

1. Capture a screenshot from the renderer showing a mix of congestion.
2. See what is actually there:
   ```bash
   roadtint calibrate my-shot.png
   ```
3. Build a draft profile from sampled points:
   ```bash
   roadtint calibrate my-shot.png \
     --point free=X,Y --point slow=X,Y --point congested=X,Y \
     --name my_renderer -o src/roadtint/profiles/my_renderer.yaml
   ```
4. **Check it by eye.** Numbers look reasonable long before they are right:
   ```bash
   roadtint extract my-shot.png -p my_renderer --overlay ./ov
   ```
   Open the overlay. Roads should be painted; parkland, water, labels and
   map pins should not be.
5. Add the profile name to the parametrised builtin tests in
   `tests/test_profile.py`.

Please say in the PR which renderer and style the profile came from, and
do not commit the screenshot you used.

### The red trap

Red straddles the 0/360 hue boundary. A band written `h_min: 0, h_max: 12`
silently misses everything from 348–360, which is most of the red in a
typical palette. Write it as `h_min: 345, h_max: 12` — `h_min > h_max` is
how a wrapping band is expressed, and the test suite checks every bundled
profile for this.

## Design notes

Three decisions that are easy to undo by accident:

**Shares, not counts.** Blob counts scale with zoom and fragmentation, so
they cannot be compared across frames. Everything reports a share of the
road detected in that frame.

**Length, not area.** Area over-weights wide strokes. Masks are thinned to
a centre line before measuring. Tests pin both properties — if you change
the measurement path, `test_length_measure_corrects_stroke_width_bias` and
`test_shares_are_stable_across_zoom` are the ones that will tell you.

**No required opencv.** The still-image path runs on numpy and pillow
alone. Morphology is implemented directly in `extract.py` for this reason.
Keep video-only imports inside `video.py`.

## Scope

Things that fit: more profiles, better road/basemap separation, output
formats, geometry-aware measurement, performance.

Things that do not: bundled API clients for specific map providers. The
tool stays offline so it carries no keys and no provider terms — see
[docs/fetchers.md](docs/fetchers.md) for wiring up collection yourself.
