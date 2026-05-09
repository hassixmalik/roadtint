import pytest

from roadtint.video import VideoUnavailable, iter_video_frames, series_from_video

from .conftest import GREEN, RED, canvas, hline

cv2 = pytest.importorskip("cv2", reason="video tests need the [video] extra")


@pytest.fixture
def clip(tmp_path):
    """A 6-second clip at 10fps that gets busier over time."""
    path = tmp_path / "clip.mp4"
    fps = 10
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (200, 200)
    )
    assert writer.isOpened(), "cannot open an mp4 writer in this environment"

    for second in range(6):
        red_len = second * 36  # 0 -> 180
        a = canvas(200, 200)
        hline(a, 100, GREEN, x0=0, x1=200, width=5)
        if red_len:
            hline(a, 100, RED, x0=0, x1=red_len, width=5)
        bgr = a[:, :, ::-1].copy()
        for _ in range(fps):
            writer.write(bgr)
    writer.release()
    return path


class TestIterFrames:
    def test_samples_at_the_requested_interval(self, clip):
        frames = list(iter_video_frames(clip, every_seconds=1.0))
        assert len(frames) == pytest.approx(6, abs=1)

    def test_respects_max_frames(self, clip):
        assert len(list(iter_video_frames(clip, every_seconds=1.0, max_frames=2))) == 2

    def test_labels_are_elapsed_time(self, clip):
        labels = [f.label for f in iter_video_frames(clip, every_seconds=1.0)]
        assert labels[0] == "00:00:00"
        assert all(lbl.count(":") == 2 for lbl in labels)

    def test_frames_are_rgb_images(self, clip):
        frame = next(iter_video_frames(clip, every_seconds=1.0))
        assert frame.image.mode == "RGB"
        assert frame.image.size == (200, 200)

    def test_missing_file_is_reported(self, tmp_path):
        with pytest.raises(VideoUnavailable, match="no such video"):
            list(iter_video_frames(tmp_path / "nope.mp4"))

    def test_unopenable_file_is_reported(self, tmp_path):
        bad = tmp_path / "bad.mp4"
        bad.write_text("not a video", encoding="utf-8")
        with pytest.raises(VideoUnavailable, match="cannot open"):
            list(iter_video_frames(bad))


class TestSeriesFromVideo:
    def test_builds_a_series(self, clip):
        s = series_from_video(clip, every_seconds=1.0)
        assert len(s) >= 5
        assert s.profile == "google_maps"

    def test_congestion_rises_through_the_clip(self, clip):
        idx = series_from_video(clip, every_seconds=1.0).column("congestion_index")
        assert idx[0] < idx[-1], "clip was authored to get busier"

    def test_labels_identify_the_source_and_offset(self, clip):
        s = series_from_video(clip, every_seconds=1.0)
        assert "clip.mp4@" in s.points[0].label

    def test_roi_is_applied(self, clip):
        s = series_from_video(clip, every_seconds=2.0, roi=(0, 90, 200, 120))
        assert s.points[0].measurement.height == 30

    def test_max_frames_limits_work(self, clip):
        assert len(series_from_video(clip, every_seconds=1.0, max_frames=2)) == 2
