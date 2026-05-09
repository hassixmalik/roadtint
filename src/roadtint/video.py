"""Sample frames from a video and measure them.

A screen recording of a traffic layer is a congestion time series that
nobody has extracted yet. This walks a video at a chosen interval and
hands the frames to the normal extraction path.

OpenCV is an optional dependency: the still-image path - which is what
most people use - should not require it. Import errors are turned into a
clear instruction rather than a traceback.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterator, List, Sequence

import numpy as np
from PIL import Image

from .extract import extract
from .profile import Profile
from .series import Series, TimePoint

__all__ = ["VideoUnavailable", "iter_video_frames", "series_from_video"]


class VideoUnavailable(RuntimeError):
    """Raised when video support is requested but unusable."""


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on install
        raise VideoUnavailable(
            "video support needs opencv. Install it with:\n"
            '    pip install "roadtint[video]"'
        ) from exc
    return cv2


@dataclass
class VideoFrame:
    index: int
    seconds: float
    image: Image.Image

    @property
    def label(self) -> str:
        td = timedelta(seconds=self.seconds)
        total = int(td.total_seconds())
        return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def iter_video_frames(
    path: str | Path,
    *,
    every_seconds: float = 60.0,
    max_frames: int | None = None,
) -> Iterator[VideoFrame]:
    """Yield frames sampled at a fixed wall-clock interval.

    Args:
        every_seconds: interval between sampled frames, in video time.
        max_frames: stop after this many samples.
    """
    cv2 = _require_cv2()
    path = Path(path)
    if not path.is_file():
        raise VideoUnavailable(f"no such video: {path}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoUnavailable(f"cannot open video: {path}")

    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0:
            # Some containers do not report fps; fall back to frame numbers.
            fps = 25.0
        stride = max(1, int(round(fps * every_seconds)))

        produced = 0
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % stride == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield VideoFrame(
                    index=index,
                    seconds=index / fps,
                    image=Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB"),
                )
                produced += 1
                if max_frames is not None and produced >= max_frames:
                    break
            index += 1
    finally:
        capture.release()


def series_from_video(
    path: str | Path,
    profile: Profile | str = "google_maps",
    *,
    every_seconds: float = 60.0,
    measure: str = "length",
    roi: Sequence[int] | None = None,
    max_frames: int | None = None,
) -> Series:
    """Measure a video at intervals and return it as a Series.

    Labels are elapsed video time (``HH:MM:SS``), not clock time, so the
    points stay in capture order.
    """
    points: List[TimePoint] = []
    for frame in iter_video_frames(
        path, every_seconds=every_seconds, max_frames=max_frames
    ):
        m = extract(
            frame.image,
            profile,
            measure=measure,
            roi=roi,
            source=f"{Path(path).name}@{frame.label}",
        )
        points.append(TimePoint(measurement=m, timestamp=None))

    return Series(
        points=points,
        profile=points[0].measurement.profile if points else "",
        measure=measure,
    )
