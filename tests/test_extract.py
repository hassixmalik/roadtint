import numpy as np
import pytest

from roadtint.extract import (
    ExtractionError,
    extract,
    extract_file,
    rgb_to_hsv_array,
    skeleton_length,
)


class TestRGBtoHSV:
    def test_matches_colorsys_on_known_colours(self):
        import colorsys

        samples = [(232, 69, 60), (52, 168, 83), (249, 171, 0), (255, 255, 255), (0, 0, 0)]
        arr = np.array([samples], dtype=np.uint8)
        got = rgb_to_hsv_array(arr)[0]

        for i, (r, g, b) in enumerate(samples):
            h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
            assert got[i][0] == pytest.approx(h * 360, abs=0.5)
            assert got[i][1] == pytest.approx(s, abs=0.01)
            assert got[i][2] == pytest.approx(v, abs=0.01)

    def test_greys_have_zero_saturation(self):
        arr = np.full((4, 4, 3), 128, dtype=np.uint8)
        assert rgb_to_hsv_array(arr)[..., 1].max() == pytest.approx(0.0)


class TestSkeletonLength:
    def test_empty_mask_is_zero(self):
        assert skeleton_length(np.zeros((50, 50), dtype=bool)) == 0

    def test_length_is_independent_of_stroke_width(self):
        thin = np.zeros((80, 200), dtype=bool)
        thin[40:43, 10:190] = True          # 180 long, 3 wide

        thick = np.zeros((80, 200), dtype=bool)
        thick[35:47, 10:190] = True         # 180 long, 12 wide

        # Areas differ 4x; estimated lengths should stay close.
        assert thick.sum() / thin.sum() > 3.5
        assert skeleton_length(thick) == pytest.approx(skeleton_length(thin), rel=0.45)

    def test_longer_road_measures_longer(self):
        short = np.zeros((80, 200), dtype=bool)
        short[40:44, 10:60] = True
        long_ = np.zeros((80, 200), dtype=bool)
        long_[40:44, 10:190] = True
        assert skeleton_length(long_) > skeleton_length(short) * 2


class TestExtract:
    def test_all_green_scene_is_all_free(self, all_green):
        m = extract(all_green)
        assert m.shares["free"] == pytest.approx(1.0, abs=0.02)
        assert m.shares["congested"] == pytest.approx(0.0, abs=0.02)
        assert m.congestion_index == pytest.approx(0.0, abs=0.02)

    def test_all_red_scene_is_all_congested(self, all_red):
        m = extract(all_red)
        assert m.shares["congested"] == pytest.approx(1.0, abs=0.02)
        assert m.congestion_index > 0.6

    def test_equal_roads_split_evenly(self, half_red_half_green):
        m = extract(half_red_half_green)
        assert m.shares["free"] == pytest.approx(0.5, abs=0.08)
        assert m.shares["congested"] == pytest.approx(0.5, abs=0.08)

    def test_shares_sum_to_one(self, half_red_half_green):
        m = extract(half_red_half_green)
        assert sum(m.shares.values()) == pytest.approx(1.0, abs=1e-6)

    def test_blank_scene_reports_no_road(self, blank_grey):
        m = extract(blank_grey)
        assert m.road_total == 0
        assert m.congestion_index == 0.0
        assert all(v == 0.0 for v in m.shares.values())

    def test_length_measure_corrects_stroke_width_bias(self, wide_green_thin_red):
        """The headline fix: equal-length roads of unequal width."""
        by_area = extract(wide_green_thin_red, measure="area")
        by_length = extract(wide_green_thin_red, measure="length")

        # Area is dominated by the wide green stroke...
        assert by_area.shares["free"] > 0.70
        # ...while length sees two roads of comparable extent.
        assert by_length.shares["free"] < by_area.shares["free"]
        assert by_length.shares["congested"] > by_area.shares["congested"]

    def test_shares_are_stable_across_zoom(self, zoomed_pair):
        """A raw count would change with zoom; a share should not."""
        small, big = zoomed_pair
        ms, mb = extract(small), extract(big)
        assert ms.shares["congested"] == pytest.approx(mb.shares["congested"], abs=0.10)
        assert ms.road_total != mb.road_total, "absolute totals do differ"

    def test_roi_excludes_ui_chrome(self, with_ui_chrome):
        full = extract(with_ui_chrome)
        cropped = extract(with_ui_chrome, roi=(0, 50, 200, 160))
        assert cropped.road_total <= full.road_total
        assert cropped.shares["congested"] == pytest.approx(1.0, abs=0.05)

    def test_blue_pins_are_not_counted_as_road(self, with_ui_chrome):
        m = extract(with_ui_chrome)
        # Blue is outside every band in the profile.
        assert m.shares["congested"] == pytest.approx(1.0, abs=0.05)

    def test_accepts_numpy_array(self, all_red):
        arr = np.asarray(all_red)
        assert extract(arr).shares["congested"] == pytest.approx(1.0, abs=0.02)

    def test_records_dimensions_and_profile(self, all_red):
        m = extract(all_red, profile="google_maps")
        assert (m.width, m.height) == (200, 200)
        assert m.profile == "google_maps"

    def test_to_row_is_flat_and_serialisable(self, half_red_half_green):
        row = extract(half_red_half_green).to_row()
        assert "share_congested" in row and "count_free" in row
        assert all(not isinstance(v, dict) for v in row.values())


class TestErrors:
    def test_rejects_unknown_measure(self, all_red):
        with pytest.raises(ExtractionError, match="measure must be"):
            extract(all_red, measure="volume")

    def test_rejects_empty_roi(self, all_red):
        with pytest.raises(ExtractionError, match="empty"):
            extract(all_red, roi=(10, 10, 10, 10))

    def test_missing_file_is_reported(self, tmp_path):
        with pytest.raises(ExtractionError, match="no such image"):
            extract_file(tmp_path / "nope.png")

    def test_unreadable_file_is_reported(self, tmp_path):
        bad = tmp_path / "bad.png"
        bad.write_text("not a png", encoding="utf-8")
        with pytest.raises(ExtractionError, match="cannot read"):
            extract_file(bad)

    def test_extract_file_roundtrip(self, tmp_path, all_red):
        p = tmp_path / "frame.png"
        all_red.save(p)
        m = extract_file(p)
        assert m.source == "frame.png"
        assert m.shares["congested"] == pytest.approx(1.0, abs=0.02)
