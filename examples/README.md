# Examples

## islamabad-f8

Twenty screenshots of the Google Maps traffic layer covering the F-8 / F-9
sector of Islamabad, around the Main Margalla Road and 9th Avenue
intersection near Air University. Captured on a weekday in May 2023,
roughly half-hourly from 08:00 to 17:00, plus one wider `monday.png` frame.

Also here: `original_notebook_2023.ipynb`, the university lab submission
this project grew out of, kept for reference.

### ⚠ Licensing

**These images are not covered by the repository's MIT licence.** They are
screenshots of Google Maps and remain the property of their copyright
holders. They are included so the documentation can show the tool working
on real input, and so the duplicate-detection behaviour can be reproduced.

**Do not redistribute them.** If you fork this repository for anything
beyond personal study, delete this directory. The test suite does not touch
these files — it builds its own synthetic images — so removing them breaks
nothing.

### Known problems with this dataset

Kept deliberately, because they are instructive and they motivated real
features:

**Seven files are three images.** These groups are byte-identical:

| Group | Files |
|---|---|
| A | `10_00am.png`, `10_30am.png`, `11_00am.png` |
| B | `12_00am.png`, `12_30am.png` |
| C | `01_00pm.png`, `01_30pm.png` |

So what looks like 19 half-hourly observations is 15 distinct ones. This is
why `roadtint series` drops duplicates by default:

```console
$ roadtint dedup examples/islamabad-f8
identical: 10_00am.png, 10_30am.png, 11_00am.png
    keep: 10_00am.png
identical: 01_00pm.png, 01_30pm.png
    keep: 01_00pm.png
identical: 12_00am.png, 12_30am.png
    keep: 12_00am.png

20 frames -> 16 distinct
```

**Sixteen different resolutions.** Every frame was screenshotted by hand at
a different zoom and pan, from 826×648 to 1213×648. This is precisely why
counting coloured blobs cannot work and why the tool reports shares.

**Midday is labelled `12_00am`.** Strictly that means midnight, and the
time parser reads it literally, so those frames sort to the start of the
series. Pass explicit timestamps if it matters to you.

**Mislabelled severity.** The original code looked only for red. In this
capture set red is scarce — most frames are 0.00–0.2% red pixels — while
orange covers 2–4%. The dominant congestion signal was being ignored
entirely.

### Removed coursework files

The original submission also carried `dataset.docx` (17 MB),
`Project_Presentation.pptx` (5.7 MB) and `Project_recording.mp4` (2.3 MB).
These were dropped from the working tree: the docx contained no text at
all, only 15 embedded copies of the screenshots already in this directory.

They remain reachable in git history, so a full clone still pays for them.
Fetch a shallow copy if that matters:

```bash
git clone --depth 1 https://github.com/hassixmalik/Trafic-Prediction-System.git
```

### Try it

```bash
roadtint series examples/islamabad-f8 --csv /tmp/islamabad.csv
roadtint calibrate examples/islamabad-f8/02_30pm.png
```
