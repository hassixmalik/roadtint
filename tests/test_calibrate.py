import numpy as np
import pytest

from roadtint.calibrate import profile_from_samples, sample_color, suggest_palette
from roadtint.extract import extract
from roadtint.profile import ProfileError

from .conftest import GREEN, ORANGE, RED, canvas, hline, to_image


class TestSampleColor:
    def test_reads_the_colour_at_a_point(self, all_red):
        assert sample_color(all_red, 100, 101) == pytest.approx(RED, abs=2)

    def test_median_patch_survives_an_antialiased_edge(self):
        a = canvas()
        hline(a, 100, RED, width=6)
        a[100, 100] = (200, 120, 110)  # one blended pixel
        assert sample_color(a, 100, 102, patch=5) == pytest.approx(RED, abs=3)

    def test_rejects_out_of_bounds_point(self, all_red):
        with pytest.raises(ProfileError, match="outside image bounds"):
            sample_color(all_red, 9999, 0)


class TestSuggestPalette:
    def test_finds_the_three_overlay_colours(self):
        a = canvas(300, 300)
        hline(a, 60, GREEN, width=6)
        hline(a, 150, ORANGE, width=6)
        hline(a, 240, RED, width=6)

        clusters = suggest_palette(to_image(a))
        centers = sorted(c.hue_center for c in clusters)
        assert len(clusters) == 3
        # green ~136, orange ~41, red ~3
        assert centers[0] == pytest.approx(3, abs=8)
        assert centers[1] == pytest.approx(41, abs=8)
        assert centers[2] == pytest.approx(136, abs=8)

    def test_ignores_the_unsaturated_basemap(self, blank_grey):
        assert suggest_palette(blank_grey) == []

    def test_shares_sum_to_about_one(self):
        a = canvas(300, 300)
        hline(a, 80, GREEN, width=6)
        hline(a, 200, RED, width=6)
        total = sum(c.share for c in suggest_palette(to_image(a)))
        assert total == pytest.approx(1.0, abs=0.02)

    def test_merges_adjacent_bins_into_one_cluster(self):
        """Antialiasing spreads one colour over a few degrees of hue."""
        a = canvas(300, 300)
        for i, shade in enumerate([(230, 69, 60), (235, 72, 58), (228, 66, 63)]):
            hline(a, 60 + i * 40, shade, width=6)
        clusters = suggest_palette(to_image(a))
        assert len(clusters) == 1, "near-identical reds should merge"

    def test_scattered_noise_does_not_bridge_distinct_colours(self):
        """Regression: antialiasing used to merge every hue into one cluster.

        A real screenshot has a few stray pixels in almost every hue bin.
        Merging adjacent *non-empty* bins therefore walked straight through
        the noise and returned a single cluster spanning 0-360 degrees.
        """
        rng = np.random.default_rng(1234)
        a = canvas(300, 300)
        hline(a, 80, GREEN, width=8)
        hline(a, 220, RED, width=8)
        # Sprinkle saturated single pixels across the whole hue circle.
        for _ in range(900):
            y = int(rng.integers(0, 300))
            x = int(rng.integers(0, 300))
            if a[y, x].tolist() != list(canvas(1, 1)[0, 0]):
                continue
            hue = float(rng.uniform(0, 360))
            import colorsys

            r, g, b = colorsys.hsv_to_rgb(hue / 360, 0.9, 0.9)
            a[y, x] = (int(r * 255), int(g * 255), int(b * 255))

        clusters = suggest_palette(to_image(a))
        assert clusters, "expected to find the two roads"
        widest = max(c.hue_max - c.hue_min for c in clusters)
        assert widest < 180, f"a cluster spanned {widest:.0f} degrees of hue"
        centers = sorted(c.hue_center for c in clusters[:2])
        assert centers[0] == pytest.approx(3, abs=10)
        assert centers[1] == pytest.approx(136, abs=10)

    def test_larger_feature_ranks_first(self):
        a = canvas(300, 300)
        hline(a, 60, GREEN, width=20)
        hline(a, 200, RED, width=4)
        clusters = suggest_palette(to_image(a))
        assert clusters[0].example_rgb == pytest.approx(GREEN, abs=3)

    def test_cluster_describes_itself_readably(self):
        a = canvas(200, 200)
        hline(a, 100, RED, width=6)
        text = suggest_palette(to_image(a))[0].describe()
        assert "hue" in text and "rgb(" in text and "#" in text


class TestProfileFromSamples:
    def test_builds_a_working_profile(self):
        a = canvas(300, 300)
        hline(a, 60, GREEN, width=6)
        hline(a, 150, ORANGE, width=6)
        hline(a, 240, RED, width=6)
        img = to_image(a)

        prof = profile_from_samples(
            img,
            {"free": (150, 62), "slow": (150, 152), "congested": (150, 242)},
            name="synthetic",
        )
        assert prof.name == "synthetic"
        assert prof.class_names == ["free", "slow", "congested"]

        # The generated profile must actually classify the image it came from.
        m = extract(img, prof)
        for cls in ("free", "slow", "congested"):
            assert m.shares[cls] == pytest.approx(1 / 3, abs=0.12), cls

    def test_applies_conventional_default_weights(self):
        a = canvas()
        hline(a, 100, RED, width=6)
        prof = profile_from_samples(to_image(a), {"congested": (100, 102)})
        assert prof.get("congested").weight == 1.0

    def test_accepts_custom_weights(self):
        a = canvas()
        hline(a, 100, RED, width=6)
        prof = profile_from_samples(
            to_image(a), {"congested": (100, 102)}, weights={"congested": 4.0}
        )
        assert prof.get("congested").weight == 4.0

    def test_red_sample_produces_a_wrapping_band(self):
        a = canvas()
        hline(a, 100, (255, 0, 0), width=6)
        prof = profile_from_samples(to_image(a), {"congested": (100, 102)})
        assert prof.get("congested").bands[0].wraps

    def test_rejects_empty_samples(self, all_red):
        with pytest.raises(ProfileError, match="at least one sample"):
            profile_from_samples(all_red, {})

    def test_rejects_malformed_point(self, all_red):
        with pytest.raises(ProfileError, match=r"must be \(x, y\)"):
            profile_from_samples(all_red, {"congested": (1, 2, 3)})

    def test_profile_survives_a_yaml_roundtrip(self, tmp_path):
        a = canvas()
        hline(a, 100, RED, width=6)
        prof = profile_from_samples(to_image(a), {"congested": (100, 102)}, name="rt")

        path = prof.save(tmp_path / "rt.yaml")
        from roadtint.profile import load_profile

        reloaded = load_profile(path)
        assert reloaded.name == "rt"
        assert reloaded.get("congested").bands[0].wraps == prof.get("congested").bands[0].wraps
