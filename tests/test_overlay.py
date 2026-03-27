import numpy as np
import pytest
from PIL import Image

from roadtint.overlay import class_masks, render_overlay, save_overlay


class TestClassMasks:
    def test_returns_a_mask_per_class(self, half_red_half_green):
        _, masks = class_masks(half_red_half_green)
        assert set(masks) == {"free", "slow", "congested", "blocked"}

    def test_masks_land_on_the_right_rows(self, half_red_half_green):
        _, masks = class_masks(half_red_half_green)
        green_rows = np.where(masks["free"].any(axis=1))[0]
        red_rows = np.where(masks["congested"].any(axis=1))[0]
        assert 55 <= green_rows.mean() <= 70
        assert 135 <= red_rows.mean() <= 150

    def test_blank_frame_matches_nothing(self, blank_grey):
        _, masks = class_masks(blank_grey)
        assert not any(m.any() for m in masks.values())

    def test_roi_crops_before_masking(self, half_red_half_green):
        base, masks = class_masks(half_red_half_green, roi=(0, 100, 200, 200))
        assert base.size == (200, 100)
        assert not masks["free"].any(), "green row is outside the roi"
        assert masks["congested"].any()


class TestRenderOverlay:
    def test_output_matches_input_size(self, half_red_half_green):
        out = render_overlay(half_red_half_green)
        assert out.size == half_red_half_green.size
        assert out.mode == "RGB"

    def test_matched_pixels_take_the_profile_colour(self, all_red):
        out = np.asarray(render_overlay(all_red, alpha=1.0))
        # google_maps paints "congested" #e8453c
        row = out[101]
        painted = row[(row[:, 0] > 200) & (row[:, 1] < 120)]
        assert len(painted) > 100

    def test_unmatched_background_is_dimmed(self, all_red):
        before = np.asarray(all_red, dtype=float)
        after = np.asarray(render_overlay(all_red, dim=0.4), dtype=float)
        # Row 5 is plain background in both.
        assert after[5].mean() < before[5].mean()

    def test_dim_one_leaves_background_untouched(self, all_red):
        before = np.asarray(all_red, dtype=float)
        after = np.asarray(render_overlay(all_red, dim=1.0), dtype=float)
        assert after[5].mean() == pytest.approx(before[5].mean(), abs=1.0)

    def test_blank_frame_renders_without_error(self, blank_grey):
        assert render_overlay(blank_grey).size == blank_grey.size

    def test_accepts_a_custom_profile_object(self, all_red):
        from roadtint.calibrate import profile_from_samples

        prof = profile_from_samples(all_red, {"congested": (100, 101)})
        assert render_overlay(all_red, prof).size == all_red.size


class TestSaveOverlay:
    def test_writes_a_readable_image(self, tmp_path, all_red):
        src = tmp_path / "in.png"
        all_red.save(src)
        out = save_overlay(src, tmp_path / "nested" / "out.png")
        assert out.is_file()
        with Image.open(out) as im:
            assert im.size == (200, 200)

    def test_creates_missing_parent_dirs(self, tmp_path, all_red):
        src = tmp_path / "in.png"
        all_red.save(src)
        out = save_overlay(src, tmp_path / "a" / "b" / "c.png")
        assert out.is_file()
