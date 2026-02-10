import csv
import json
from datetime import time

import pytest

from roadtint.series import (
    build_series,
    parse_time_from_name,
    write_csv,
    write_json,
)

from .conftest import GREEN, ORANGE, RED, canvas, hline, to_image


@pytest.fixture
def frame_dir(tmp_path):
    """A small morning->afternoon run with a duplicate pair."""
    def scene(red_len):
        """Fixed total road; `red_len` of it is congested.

        Severity is varied by road LENGTH, not stroke width - the
        extractor measures centre-line length precisely so that width
        does not affect the result.
        """
        a = canvas(200, 200)
        hline(a, 100, GREEN, x0=0, x1=200, width=5)
        if red_len:
            hline(a, 100, RED, x0=0, x1=red_len, width=5)
        return to_image(a)

    quiet = scene(0)
    quiet.save(tmp_path / "09_00am.png")
    quiet.save(tmp_path / "09_30am.png")   # exact duplicate
    scene(90).save(tmp_path / "11_00am.png")
    scene(180).save(tmp_path / "01_00pm.png")
    return tmp_path


class TestParseTime:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("08_00am.png", time(8, 0)),
            ("08_30am.png", time(8, 30)),
            ("12_00am.png", time(0, 0)),   # literal reading of 12am
            ("12_30pm.png", time(12, 30)),
            ("01_00pm.png", time(13, 0)),
            ("04_30pm.png", time(16, 30)),
            ("5pm.png", time(17, 0)),
            ("frame_1430.png", time(14, 30)),
            ("cap-14-30.png", time(14, 30)),
        ],
    )
    def test_parses_known_layouts(self, name, expected):
        assert parse_time_from_name(name) == expected

    @pytest.mark.parametrize("name", ["scene.png", "monday.png", "IMG.png"])
    def test_returns_none_when_absent(self, name):
        assert parse_time_from_name(name) is None

    def test_rejects_impossible_clock_time(self):
        assert parse_time_from_name("99_99am.png") is None


class TestBuildSeries:
    def test_drops_duplicate_frames(self, frame_dir):
        paths = sorted(frame_dir.glob("*.png"))
        s = build_series(paths)
        assert len(paths) == 4
        assert len(s) == 3
        assert s.duplicates_removed == 1

    def test_can_keep_duplicates(self, frame_dir):
        paths = sorted(frame_dir.glob("*.png"))
        s = build_series(paths, drop_duplicates=False)
        assert len(s) == 4

    def test_orders_by_clock_not_filename(self, frame_dir):
        s = build_series(sorted(frame_dir.glob("*.png")))
        labels = [p.label for p in s]
        assert labels == sorted(labels), "01_00pm must sort after 11_00am"
        assert labels[-1] == "13:00"

    def test_untimed_frames_are_kept_at_the_end(self, tmp_path):
        img = to_image(hline(canvas(), 100, RED, width=6))
        img.save(tmp_path / "09_00am.png")
        other = to_image(hline(canvas(), 40, GREEN, width=9))
        other.save(tmp_path / "scratch.png")

        s = build_series(sorted(tmp_path.glob("*.png")))
        assert len(s) == 2
        assert s.points[-1].label == "scratch.png"

    def test_explicit_timestamps_win(self, tmp_path):
        to_image(hline(canvas(), 100, RED, width=6)).save(tmp_path / "frame.png")
        s = build_series([tmp_path / "frame.png"], timestamps={"frame.png": time(6, 15)})
        assert s.points[0].label == "06:15"

    def test_peak_and_quietest_are_identified(self, frame_dir):
        s = build_series(sorted(frame_dir.glob("*.png")))
        assert s.peak().measurement.congestion_index >= s.quietest().measurement.congestion_index
        assert s.peak().label == "13:00"

    def test_summary_reports_counts_and_extremes(self, frame_dir):
        s = build_series(sorted(frame_dir.glob("*.png")))
        summary = s.summary()
        assert summary["frames"] == 3
        assert summary["duplicates_removed"] == 1
        assert summary["min_index"] <= summary["mean_index"] <= summary["max_index"]

    def test_empty_input_yields_empty_series(self):
        s = build_series([])
        assert len(s) == 0
        assert s.summary() == {"frames": 0}
        assert s.peak() is None

    def test_column_extracts_a_numeric_series(self, frame_dir):
        s = build_series(sorted(frame_dir.glob("*.png")))
        col = s.column("congestion_index")
        assert len(col) == 3 and all(isinstance(v, float) for v in col)

    def test_column_rejects_unknown_key(self, frame_dir):
        s = build_series(sorted(frame_dir.glob("*.png")))
        with pytest.raises(KeyError, match="no column"):
            s.column("nope")

    def test_roi_is_applied_to_every_frame(self, frame_dir):
        full = build_series(sorted(frame_dir.glob("*.png")))
        crop = build_series(sorted(frame_dir.glob("*.png")), roi=(0, 100, 200, 200))
        assert crop.points[0].measurement.height == 100
        assert full.points[0].measurement.height == 200


class TestOutputs:
    def test_csv_is_rectangular_with_a_header(self, frame_dir, tmp_path):
        s = build_series(sorted(frame_dir.glob("*.png")))
        out = write_csv(s, tmp_path / "out.csv")

        with out.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 3
        assert "congestion_index" in rows[0]
        assert "share_congested" in rows[0]
        assert len({len(r) for r in rows}) == 1

    def test_empty_series_writes_empty_csv(self, tmp_path):
        out = write_csv(build_series([]), tmp_path / "empty.csv")
        assert out.read_text(encoding="utf-8") == ""

    def test_json_carries_summary_and_points(self, frame_dir, tmp_path):
        s = build_series(sorted(frame_dir.glob("*.png")))
        out = write_json(s, tmp_path / "out.json")

        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["profile"] == "google_maps"
        assert payload["measure"] == "length"
        assert payload["summary"]["frames"] == 3
        assert len(payload["points"]) == 3

    def test_output_dirs_are_created(self, frame_dir, tmp_path):
        s = build_series(sorted(frame_dir.glob("*.png")))
        out = write_csv(s, tmp_path / "nested" / "deep" / "out.csv")
        assert out.is_file()
