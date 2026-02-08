import numpy as np
import pytest
from PIL import Image

from roadtint.dedup import dhash, find_duplicates, hamming, hash_file, unique_paths

from .conftest import GREEN, RED, canvas, hline, to_image


def save(img: Image.Image, path, **kw):
    img.save(path, **kw)
    return path


class TestDHash:
    def test_identical_images_hash_the_same(self, all_red):
        assert dhash(all_red) == dhash(all_red.copy())

    def test_different_images_hash_differently(self, all_red, all_green):
        assert dhash(all_red) != dhash(all_green)

    def test_hash_size_controls_bit_width(self, all_red):
        # four blocks of hash_size**2 bits: h-grad, v-grad, 2x colour sector
        assert dhash(all_red, hash_size=4).bit_length() <= 4 * 16
        assert dhash(all_red, hash_size=8).bit_length() <= 4 * 64

    def test_horizontal_stripes_do_not_all_collide(self):
        """Regression: plain dHash hashed every horizontal road to zero."""
        a = to_image(hline(canvas(), 40, GREEN, width=10))
        b = to_image(hline(canvas(), 160, RED, width=3))
        assert dhash(a) != 0
        assert hamming(dhash(a), dhash(b)) > 4

    def test_recolouring_the_same_shape_changes_the_hash(self):
        shape_green = to_image(hline(canvas(), 100, GREEN, width=6))
        shape_red = to_image(hline(canvas(), 100, RED, width=6))
        assert hamming(dhash(shape_green), dhash(shape_red)) > 0

    def test_survives_jpeg_recompression(self, tmp_path):
        a = canvas(200, 200)
        hline(a, 60, GREEN, width=6)
        hline(a, 140, RED, width=6)
        img = to_image(a)

        clean = save(img, tmp_path / "a.png")
        lossy = save(img, tmp_path / "a.jpg", quality=55)

        assert hamming(dhash(clean), dhash(lossy)) <= 4

    def test_small_shift_stays_close(self):
        a = canvas(200, 200)
        hline(a, 100, RED, width=6)
        shifted = canvas(200, 200)
        hline(shifted, 102, RED, width=6)
        assert hamming(dhash(to_image(a)), dhash(to_image(shifted))) <= 8


class TestHamming:
    def test_zero_for_equal(self):
        assert hamming(0b1011, 0b1011) == 0

    def test_counts_differing_bits(self):
        assert hamming(0b0000, 0b1011) == 3


class TestFindDuplicates:
    def test_detects_byte_identical_copies(self, tmp_path):
        img = to_image(hline(canvas(), 100, RED))
        paths = [save(img, tmp_path / f"{n}.png") for n in ("a", "b", "c")]

        groups = find_duplicates(paths)
        assert len(groups) == 1
        assert groups[0].exact
        assert len(groups[0].members) == 3

    def test_reproduces_the_repository_duplicate_pattern(self, tmp_path):
        """The real sample set has 7 files that are only 3 images."""
        scene_a = to_image(hline(canvas(), 60, GREEN, width=6))
        scene_b = to_image(hline(canvas(), 120, RED, width=6))
        scene_c = to_image(hline(canvas(), 180, GREEN, width=10))

        paths = []
        for name in ("10_00", "10_30", "11_00"):
            paths.append(save(scene_a, tmp_path / f"{name}.png"))
        for name in ("12_00", "12_30"):
            paths.append(save(scene_b, tmp_path / f"{name}.png"))
        for name in ("13_00", "13_30"):
            paths.append(save(scene_c, tmp_path / f"{name}.png"))

        groups = find_duplicates(paths)
        assert sorted(len(g.members) for g in groups) == [2, 2, 3]

        kept, _ = unique_paths(paths)
        assert len(kept) == 3, "7 files collapse to 3 distinct scenes"

    def test_distinct_images_are_not_grouped(self, tmp_path):
        a = save(to_image(hline(canvas(), 40, GREEN, width=10)), tmp_path / "a.png")
        b = save(to_image(hline(canvas(), 160, RED, width=3)), tmp_path / "b.png")
        assert find_duplicates([a, b]) == []

    def test_threshold_zero_requires_near_exact_match(self, tmp_path):
        a = canvas(200, 200)
        hline(a, 100, RED, width=6)
        shifted = canvas(200, 200)
        hline(shifted, 140, RED, width=6)

        p1 = save(to_image(a), tmp_path / "a.png")
        p2 = save(to_image(shifted), tmp_path / "b.png")
        assert find_duplicates([p1, p2], threshold=0) == []

    def test_keeper_is_stable_and_redundant_excludes_it(self, tmp_path):
        img = to_image(hline(canvas(), 100, RED))
        paths = [save(img, tmp_path / f"{n}.png") for n in ("z", "a", "m")]

        group = find_duplicates(paths)[0]
        assert group.keeper.name == "a.png"
        assert sorted(p.name for p in group.redundant) == ["m.png", "z.png"]

    def test_unreadable_files_are_skipped_not_fatal(self, tmp_path):
        img = to_image(hline(canvas(), 100, RED))
        good1 = save(img, tmp_path / "a.png")
        good2 = save(img, tmp_path / "b.png")
        bad = tmp_path / "broken.png"
        bad.write_text("not an image", encoding="utf-8")

        groups = find_duplicates([good1, bad, good2])
        assert len(groups) == 1 and len(groups[0].members) == 2

    def test_empty_input_is_handled(self):
        assert find_duplicates([]) == []

    def test_group_describes_itself(self, tmp_path):
        img = to_image(hline(canvas(), 100, RED))
        paths = [save(img, tmp_path / f"{n}.png") for n in ("a", "b")]
        text = find_duplicates(paths)[0].describe()
        assert "identical" in text and "a.png" in text


class TestUniquePaths:
    def test_preserves_input_order_of_kept_frames(self, tmp_path):
        a = to_image(hline(canvas(), 40, GREEN, width=10))
        b = to_image(hline(canvas(), 160, RED, width=3))
        p1 = save(a, tmp_path / "1.png")
        p2 = save(b, tmp_path / "2.png")
        p3 = save(a, tmp_path / "3.png")  # duplicate of p1

        kept, groups = unique_paths([p1, p2, p3])
        assert [p.name for p in kept] == ["1.png", "2.png"]
        assert len(groups) == 1

    def test_no_duplicates_keeps_everything(self, tmp_path):
        p1 = save(to_image(hline(canvas(), 40, GREEN, width=10)), tmp_path / "1.png")
        p2 = save(to_image(hline(canvas(), 160, RED, width=3)), tmp_path / "2.png")
        kept, groups = unique_paths([p1, p2])
        assert len(kept) == 2 and groups == []
