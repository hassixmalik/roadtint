import json

import pytest

from roadtint.cli import main

from .conftest import GREEN, RED, canvas, hline, to_image


@pytest.fixture
def run_dir(tmp_path):
    def scene(red_len):
        a = canvas(200, 200)
        hline(a, 100, GREEN, x0=0, x1=200, width=5)
        if red_len:
            hline(a, 100, RED, x0=0, x1=red_len, width=5)
        return to_image(a)

    scene(0).save(tmp_path / "09_00am.png")
    scene(0).save(tmp_path / "09_30am.png")    # duplicate
    scene(120).save(tmp_path / "01_00pm.png")
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
    return tmp_path


class TestExtract:
    def test_prints_a_row_per_frame(self, run_dir, capsys):
        assert main(["extract", str(run_dir)]) == 0
        out = capsys.readouterr().out
        assert "09_00am.png" in out and "index=" in out

    def test_ignores_non_image_files(self, run_dir, capsys):
        main(["extract", str(run_dir)])
        assert "notes.txt" not in capsys.readouterr().out

    def test_json_output_is_valid(self, run_dir, capsys):
        assert main(["extract", str(run_dir), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert len(payload) == 3
        assert "congestion_index" in payload[0]

    def test_roi_is_honoured(self, run_dir, capsys):
        assert main(["extract", str(run_dir), "--roi", "0,0,200,100", "--json"]) == 0
        assert json.loads(capsys.readouterr().out)[0]["height"] == 100

    def test_rejects_malformed_roi(self, run_dir):
        with pytest.raises(SystemExit, match="four comma-separated"):
            main(["extract", str(run_dir), "--roi", "1,2,3"])

    def test_rejects_inverted_roi(self, run_dir):
        with pytest.raises(SystemExit, match="left<right"):
            main(["extract", str(run_dir), "--roi", "100,0,10,50"])

    def test_missing_path_returns_error(self, tmp_path, capsys):
        assert main(["extract", str(tmp_path / "nope")]) == 1
        assert "no images found" in capsys.readouterr().err

    def test_unreadable_frame_is_skipped_by_default(self, run_dir, capsys):
        (run_dir / "broken.png").write_text("not a png", encoding="utf-8")
        assert main(["extract", str(run_dir)]) == 0
        assert "error:" in capsys.readouterr().err

    def test_strict_stops_on_bad_frame(self, run_dir):
        (run_dir / "broken.png").write_text("not a png", encoding="utf-8")
        assert main(["extract", str(run_dir), "--strict"]) == 1

    def test_unknown_profile_is_reported(self, run_dir, capsys):
        assert main(["extract", str(run_dir), "-p", "nope"]) == 1
        assert "unknown profile" in capsys.readouterr().err


class TestSeries:
    def test_prints_a_bar_chart_and_summary(self, run_dir, capsys):
        assert main(["series", str(run_dir)]) == 0
        out = capsys.readouterr().out
        assert "#" in out
        assert "duplicate(s) dropped" in out

    def test_writes_csv(self, run_dir, tmp_path, capsys):
        out = tmp_path / "s.csv"
        assert main(["series", str(run_dir), "--csv", str(out)]) == 0
        text = out.read_text(encoding="utf-8")
        assert "congestion_index" in text.splitlines()[0]
        assert len(text.strip().splitlines()) == 3  # header + 2 unique frames

    def test_writes_json(self, run_dir, tmp_path):
        out = tmp_path / "s.json"
        assert main(["series", str(run_dir), "--json-out", str(out)]) == 0
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["summary"]["frames"] == 2
        assert payload["summary"]["duplicates_removed"] == 1

    def test_keep_duplicates_flag(self, run_dir, tmp_path):
        out = tmp_path / "s.json"
        main(["series", str(run_dir), "--keep-duplicates", "--json-out", str(out)])
        assert json.loads(out.read_text(encoding="utf-8"))["summary"]["frames"] == 3

    def test_orders_output_by_clock(self, run_dir, tmp_path):
        out = tmp_path / "s.json"
        main(["series", str(run_dir), "--json-out", str(out)])
        labels = [p["label"] for p in json.loads(out.read_text(encoding="utf-8"))["points"]]
        assert labels == ["09:00", "13:00"]


class TestCalibrate:
    def test_reports_dominant_colours(self, run_dir, capsys):
        assert main(["calibrate", str(run_dir / "01_00pm.png")]) == 0
        out = capsys.readouterr().out
        assert "Dominant colours" in out and "hue" in out

    def test_builds_a_profile_from_points(self, run_dir, tmp_path, capsys):
        out = tmp_path / "custom.yaml"
        code = main([
            "calibrate", str(run_dir / "01_00pm.png"),
            "--point", "congested=50,102",
            "--point", "free=180,102",
            "--name", "mine", "-o", str(out),
        ])
        assert code == 0 and out.is_file()
        text = out.read_text(encoding="utf-8")
        assert "name: mine" in text and "congested" in text

    def test_generated_profile_is_usable(self, run_dir, tmp_path):
        prof = tmp_path / "custom.yaml"
        main([
            "calibrate", str(run_dir / "01_00pm.png"),
            "--point", "congested=50,102", "--point", "free=180,102",
            "-o", str(prof),
        ])
        assert main(["extract", str(run_dir / "01_00pm.png"), "-p", str(prof)]) == 0

    def test_rejects_malformed_point(self, run_dir):
        with pytest.raises(SystemExit, match="class=x,y"):
            main(["calibrate", str(run_dir / "01_00pm.png"), "--point", "congested"])

    def test_missing_image_returns_error(self, tmp_path, capsys):
        assert main(["calibrate", str(tmp_path / "nope.png")]) == 1
        assert "no such image" in capsys.readouterr().err

    def test_blank_image_reports_no_colours(self, tmp_path, capsys):
        to_image(canvas()).save(tmp_path / "blank.png")
        assert main(["calibrate", str(tmp_path / "blank.png")]) == 1
        assert "no saturated colours" in capsys.readouterr().err


class TestDedup:
    def test_reports_duplicate_groups(self, run_dir, capsys):
        assert main(["dedup", str(run_dir)]) == 0
        out = capsys.readouterr().out
        assert "identical" in out and "keep:" in out
        assert "3 frames -> 2 distinct" in out

    def test_clean_set_reports_none(self, tmp_path, capsys):
        to_image(hline(canvas(), 40, GREEN, width=10)).save(tmp_path / "a.png")
        to_image(hline(canvas(), 160, RED, width=3)).save(tmp_path / "b.png")
        assert main(["dedup", str(tmp_path)]) == 0
        assert "no duplicates" in capsys.readouterr().out


class TestProfiles:
    def test_lists_bundled_profiles(self, capsys):
        assert main(["profiles"]) == 0
        out = capsys.readouterr().out
        assert "google_maps" in out and "classes:" in out


class TestTopLevel:
    def test_version_flag(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
        assert exc.value.code == 0
        assert "roadtint" in capsys.readouterr().out

    def test_subcommand_required(self):
        with pytest.raises(SystemExit):
            main([])
