"""Feature extraction foundations for numeric pose data."""

from .engineering import (
    LEFT_WRIST,
    NOSE,
    RIGHT_WRIST,
    dominant_frequency_hz,
    inter_landmark_distance_series,
    landmark_axis_series,
    periodicity_strength,
    window_feature_dict,
)
from .windowing import VideoWindow, WindowSpec, generate_windows_for_video

__all__ = [
    "LEFT_WRIST",
    "NOSE",
    "RIGHT_WRIST",
    "VideoWindow",
    "WindowSpec",
    "dominant_frequency_hz",
    "generate_windows_for_video",
    "inter_landmark_distance_series",
    "landmark_axis_series",
    "periodicity_strength",
    "window_feature_dict",
]
