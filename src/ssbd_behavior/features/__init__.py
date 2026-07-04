"""Feature extraction foundations for numeric pose data."""

from .windowing import VideoWindow, WindowSpec, generate_windows_for_video

__all__ = ["VideoWindow", "WindowSpec", "generate_windows_for_video"]
