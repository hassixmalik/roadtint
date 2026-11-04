"""Guards on the public surface."""

import importlib

import pytest

import roadtint


def test_version_is_exposed():
    assert roadtint.__version__ == "0.1.0"


@pytest.mark.parametrize("name", roadtint.__all__)
def test_everything_in_all_is_importable(name):
    assert hasattr(roadtint, name), f"__all__ lists {name} but it is not exported"


def test_top_level_api_is_usable_without_deep_imports():
    from roadtint import build_series, extract_file, load_profile  # noqa: F401


def test_importing_the_package_does_not_require_opencv(monkeypatch):
    """The still-image path must work without the video extra."""
    import sys

    monkeypatch.setitem(sys.modules, "cv2", None)
    importlib.reload(roadtint)
    assert roadtint.extract_file is not None


def test_video_module_is_not_imported_eagerly():
    import sys

    sys.modules.pop("roadtint.video", None)
    importlib.reload(roadtint)
    assert "roadtint.video" not in sys.modules, "video must stay opt-in"
