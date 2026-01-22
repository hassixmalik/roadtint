import pytest

from roadtint.profile import (
    HSVBand,
    Profile,
    ProfileError,
    builtin_profile_names,
    load_profile,
)


class TestHSVBand:
    def test_plain_band_contains_hue_inside_range(self):
        band = HSVBand(h_min=100, h_max=140)
        assert band.contains(120, 0.8, 0.8)
        assert not band.contains(160, 0.8, 0.8)

    def test_wrapping_band_spans_hue_origin(self):
        band = HSVBand(h_min=345, h_max=12)
        assert band.wraps
        assert band.contains(350, 0.8, 0.8)
        assert band.contains(5, 0.8, 0.8)
        assert not band.contains(180, 0.8, 0.8)

    def test_saturation_and_value_gate_membership(self):
        band = HSVBand(h_min=100, h_max=140, s_min=0.5, v_min=0.5)
        assert not band.contains(120, 0.2, 0.9), "low saturation must be excluded"
        assert not band.contains(120, 0.9, 0.2), "low value must be excluded"

    def test_rejects_out_of_range_values(self):
        with pytest.raises(ProfileError):
            HSVBand(h_min=-5, h_max=10)
        with pytest.raises(ProfileError):
            HSVBand(h_min=0, h_max=400)

    def test_rejects_inverted_s_and_v_bounds(self):
        with pytest.raises(ProfileError):
            HSVBand(h_min=0, h_max=10, s_min=0.9, s_max=0.1)
        with pytest.raises(ProfileError):
            HSVBand(h_min=0, h_max=10, v_min=0.9, v_max=0.1)

    def test_around_rgb_centres_on_the_sampled_colour(self):
        band = HSVBand.around_rgb((232, 69, 60), hue_tol=10)
        assert band.contains(3, 0.74, 0.91)

    def test_around_rgb_wraps_for_red(self):
        band = HSVBand.around_rgb((255, 0, 0), hue_tol=12)
        assert band.wraps
        assert band.contains(355, 1.0, 1.0)
        assert band.contains(5, 1.0, 1.0)


class TestProfileParsing:
    def test_rejects_profile_without_classes(self):
        with pytest.raises(ProfileError, match="missing required keys"):
            Profile.from_mapping({"name": "x"})

    def test_rejects_class_without_bands(self):
        with pytest.raises(ProfileError, match="no bands"):
            Profile.from_mapping({"name": "x", "classes": {"free": {}}})

    def test_rejects_unknown_band_key(self):
        with pytest.raises(ProfileError, match="unknown band keys"):
            Profile.from_mapping(
                {"name": "x", "classes": {"free": {"bands": [{"h_min": 1, "h_max": 2, "hue": 3}]}}}
            )

    def test_roundtrips_through_yaml(self):
        src = load_profile("google_maps")
        import yaml

        clone = Profile.from_mapping(yaml.safe_load(src.dump_yaml()))
        assert clone.class_names == src.class_names
        assert clone.get("congested").bands[0].wraps


class TestBuiltins:
    def test_ships_expected_profiles(self):
        names = builtin_profile_names()
        assert {"google_maps", "mapbox_traffic", "here_traffic"} <= set(names)

    @pytest.mark.parametrize("name", ["google_maps", "mapbox_traffic", "here_traffic"])
    def test_every_builtin_parses_and_has_four_classes(self, name):
        p = load_profile(name)
        assert p.class_names == ["free", "slow", "congested", "blocked"]

    @pytest.mark.parametrize("name", ["google_maps", "mapbox_traffic", "here_traffic"])
    def test_red_classes_wrap_the_hue_origin(self, name):
        p = load_profile(name)
        assert any(b.wraps for b in p.get("congested").bands)

    def test_unknown_name_lists_alternatives(self):
        with pytest.raises(ProfileError, match="google_maps"):
            load_profile("not_a_real_profile")

    def test_missing_path_is_reported(self, tmp_path):
        with pytest.raises(ProfileError, match="not found"):
            load_profile(tmp_path / "nope.yaml")

    def test_invalid_yaml_is_reported(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("name: x\n  bad indent: [", encoding="utf-8")
        with pytest.raises(ProfileError, match="invalid YAML"):
            load_profile(bad)
