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
from .pipeline import (
    FEATURE_TABLE_BASE_COLUMNS,
    FeatureRow,
    build_feature_rows_for_video,
    read_feature_table_csv,
    write_feature_table_csv,
)
from .windowing import VideoWindow, WindowSpec, generate_windows_for_video

__all__ = [
    "LEFT_WRIST",
    "NOSE",
    "RIGHT_WRIST",
    "FEATURE_TABLE_BASE_COLUMNS",
    "FeatureRow",
    "VideoWindow",
    "WindowSpec",
    "build_feature_rows_for_video",
    "dominant_frequency_hz",
    "generate_windows_for_video",
    "inter_landmark_distance_series",
    "landmark_axis_series",
    "periodicity_strength",
    "read_feature_table_csv",
    "window_feature_dict",
    "write_feature_table_csv",
]
