# Wiring up your own data source

roadtint is offline by design. The core reads images from disk, so it needs
no API keys, has no rate limits to respect, and carries no provider terms
into your project. That is deliberate: the analysis is the hard part and it
should not be coupled to anyone's REST API.

Collecting frames is the easy part, and it is yours to own.

## The contract

There isn't a plugin system to learn. A fetcher is anything that writes
image files into a directory:

```
captures/
  0800.png
  0830.png
  0900.png
```

Then:

```bash
roadtint series ./captures --csv out.csv
```

That is the whole interface. A shell script with `curl` is a valid fetcher.

## Naming frames

`roadtint series` reads timestamps from filenames. Recognised layouts:

| Pattern | Example | Parsed as |
|---|---|---|
| `H_MMam/pm` | `08_30am.png` | 08:30 |
| `Ham/pm` | `5pm.png` | 17:00 |
| `HHMM` | `1430.png` | 14:30 |
| `HH-MM`, `HH_MM`, `HH:MM` | `cap-14-30.png` | 14:30 |

Frames with no recognisable time keep their input order and sort last.
Zero-padded 24-hour names (`0830.png`) are the least ambiguous choice.

To span more than one day, put the date in the directory and run one series
per day, or pass explicit timestamps through the Python API:

```python
from datetime import time
from roadtint.series import build_series

series = build_series(paths, timestamps={"frame_a.png": time(8, 30)})
```

## Example: static map tiles on a schedule

Most providers expose a static-image endpoint. This fetches one frame and
names it by clock time:

```bash
#!/usr/bin/env bash
# fetch.sh - grab one traffic frame
set -euo pipefail

: "${MAP_TOKEN:?set MAP_TOKEN in your environment}"
OUT="${1:-./captures}"
mkdir -p "$OUT"

BBOX="73.02,33.68,73.08,33.72"     # lon,lat,lon,lat
SIZE="1024x768"
STAMP="$(date +%H%M)"

curl -fsS -o "$OUT/${STAMP}.png" \
  "https://api.example-maps.com/static/traffic?bbox=${BBOX}&size=${SIZE}&token=${MAP_TOKEN}"

echo "saved $OUT/${STAMP}.png"
```

Run it every half hour:

```cron
*/30 8-18 * * 1-5  /path/to/fetch.sh /path/to/captures
```

Then analyse the day:

```bash
roadtint series /path/to/captures --csv "$(date +%F).csv"
```

Keep your key in the environment, not in the script.

## Example: a Python fetcher

```python
import os
from datetime import datetime
from pathlib import Path

import requests

OUT = Path("captures")
OUT.mkdir(exist_ok=True)

def fetch(bbox: str, size: str = "1024x768") -> Path:
    resp = requests.get(
        "https://api.example-maps.com/static/traffic",
        params={"bbox": bbox, "size": size, "token": os.environ["MAP_TOKEN"]},
        timeout=30,
    )
    resp.raise_for_status()
    path = OUT / f"{datetime.now():%H%M}.png"
    path.write_bytes(resp.content)
    return path
```

Then drive roadtint directly:

```python
from roadtint.series import build_series, write_csv

series = build_series(sorted(OUT.glob("*.png")), "google_maps")
write_csv(series, "today.csv")
print(series.summary())
```

## From a screen recording

If you already have video, sample frames with ffmpeg — one frame per
minute here — and treat the output as a capture folder:

```bash
mkdir -p frames
ffmpeg -i recording.mp4 -vf fps=1/60 frames/%04d.png
roadtint series ./frames --csv recording.csv
```

Those names carry no clock time, so the series keeps file order. That is
usually what you want for a recording.

## Things to get right

**Keep the viewport fixed.** This matters more than anything else. Shares
are computed over the road visible in each frame, so if the map pans or
zooms between captures you are measuring different places and the series
is not a series. Pin the bbox and the image size.

**Crop the furniture once.** Find the right `--roi` for your viewport and
reuse it. Legends, attribution strips and search bars are coloured and
will otherwise be counted.

**Check your palette.** Run `roadtint calibrate` on one frame before
collecting hundreds. If the dominant hues do not match your profile, the
numbers will be confidently wrong.

**Respect the provider's terms.** Automated capture is not allowed
everywhere, and terms differ sharply between providers and between their
free and paid tiers. That is on you, which is part of why none of this
ships in the tool.
