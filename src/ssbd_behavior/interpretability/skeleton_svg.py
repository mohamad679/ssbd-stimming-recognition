"""Render abstract SVG stick figures from numeric pose keypoints only."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math
from numbers import Integral, Real

from ssbd_behavior.pose.keypoints import (
    PoseKeypoint,
    validate_confidence,
    validate_coordinate,
    validate_landmark_index,
)


DEFAULT_SKELETON_EDGES = (
    (0, 11),
    (0, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
)

_SVG_NAMESPACE = "http://www.w3.org/2000/svg"
_DEFAULT_STROKE = "#1f2933"
_DEFAULT_JOINT_FILL = "#52606d"


@dataclass(frozen=True)
class SkeletonPoint2D:
    """A single 2D pose landmark for privacy-safe SVG rendering.

    Coordinates may be unavailable and should then remain ``None`` so the
    renderer can skip them honestly instead of inventing values.
    """

    landmark_index: int
    x: float | None
    y: float | None
    confidence: float = 1.0

    def __post_init__(self) -> None:
        validate_landmark_index(self.landmark_index)
        validate_coordinate(self.x, "x", optional=True)
        validate_coordinate(self.y, "y", optional=True)
        validate_confidence(self.confidence)

    def is_available(self, min_confidence: float) -> bool:
        return (
            self.x is not None
            and self.y is not None
            and self.confidence >= min_confidence
        )


def render_skeleton_svg(
    keypoints: Iterable[PoseKeypoint | SkeletonPoint2D],
    *,
    width: int = 240,
    height: int = 240,
    padding: int = 24,
    joint_radius: float = 4.0,
    stroke_width: float = 3.0,
    min_confidence: float = 0.5,
) -> str:
    """Render one pose as an abstract, frame-free SVG stick figure."""

    _validate_canvas(width=width, height=height, padding=padding)
    _validate_non_negative_number(joint_radius, "joint_radius")
    _validate_non_negative_number(stroke_width, "stroke_width")
    _validate_unit_interval(min_confidence, "min_confidence")

    pose = _normalize_pose(keypoints)
    bounds = _pose_bounds([pose], min_confidence)
    return _compose_svg(
        panel_count=1,
        panel_width=width,
        height=height,
        rendered_groups=[
            _render_pose_group(
                pose,
                bounds=bounds,
                panel_origin_x=0.0,
                panel_width=width,
                height=height,
                padding=padding,
                joint_radius=joint_radius,
                stroke_width=stroke_width,
                min_confidence=min_confidence,
            )
        ],
    )


def render_sequence_summary_svg(
    pose_sequence: Sequence[Iterable[PoseKeypoint | SkeletonPoint2D]],
    *,
    width_per_pose: int = 180,
    height: int = 240,
    padding: int = 24,
    joint_radius: float = 3.5,
    stroke_width: float = 2.5,
    min_confidence: float = 0.5,
    max_poses: int = 3,
) -> str:
    """Render a deterministic multi-pose summary from a short pose sequence."""

    if not pose_sequence:
        raise ValueError("at least one pose is required")
    _validate_canvas(width=width_per_pose, height=height, padding=padding)
    _validate_non_negative_number(joint_radius, "joint_radius")
    _validate_non_negative_number(stroke_width, "stroke_width")
    _validate_unit_interval(min_confidence, "min_confidence")
    if (
        isinstance(max_poses, bool)
        or not isinstance(max_poses, Integral)
        or max_poses <= 0
    ):
        raise ValueError("max_poses must be a positive integer")

    normalized_poses = [_normalize_pose(pose) for pose in pose_sequence]
    if len(normalized_poses) == 1:
        return render_skeleton_svg(
            normalized_poses[0],
            width=width_per_pose,
            height=height,
            padding=padding,
            joint_radius=joint_radius,
            stroke_width=stroke_width,
            min_confidence=min_confidence,
        )

    selected_indices = _sample_pose_indices(len(normalized_poses), max_poses)
    selected_poses = [normalized_poses[index] for index in selected_indices]
    bounds = _pose_bounds(selected_poses, min_confidence)
    rendered_groups = [
        _render_pose_group(
            pose,
            bounds=bounds,
            panel_origin_x=index * width_per_pose,
            panel_width=width_per_pose,
            height=height,
            padding=padding,
            joint_radius=joint_radius,
            stroke_width=stroke_width,
            min_confidence=min_confidence,
        )
        for index, pose in enumerate(selected_poses)
    ]
    return _compose_svg(
        panel_count=len(selected_poses),
        panel_width=width_per_pose,
        height=height,
        rendered_groups=rendered_groups,
    )


def _normalize_pose(
    keypoints: Iterable[PoseKeypoint | SkeletonPoint2D],
) -> list[SkeletonPoint2D]:
    normalized_points = [_coerce_point(point) for point in keypoints]
    if not normalized_points:
        raise ValueError("at least one keypoint is required")

    seen_indices: set[int] = set()
    for point in normalized_points:
        if point.landmark_index in seen_indices:
            raise ValueError(
                f"duplicate landmark_index in one pose: {point.landmark_index}"
            )
        seen_indices.add(point.landmark_index)
    return sorted(normalized_points, key=lambda point: point.landmark_index)


def _coerce_point(point: PoseKeypoint | SkeletonPoint2D) -> SkeletonPoint2D:
    if isinstance(point, SkeletonPoint2D):
        return point
    if isinstance(point, PoseKeypoint):
        return SkeletonPoint2D(
            landmark_index=point.landmark_index,
            x=point.x,
            y=point.y,
            confidence=point.confidence,
        )
    raise TypeError("keypoints must contain PoseKeypoint or SkeletonPoint2D values")


def _sample_pose_indices(count: int, max_poses: int) -> list[int]:
    if count <= max_poses:
        return list(range(count))
    if max_poses == 1:
        return [count // 2]

    sampled = {
        round(step * (count - 1) / (max_poses - 1)) for step in range(max_poses)
    }
    return sorted(sampled)


def _pose_bounds(
    poses: Sequence[Sequence[SkeletonPoint2D]],
    min_confidence: float,
) -> tuple[float, float, float, float]:
    available_points = [
        point
        for pose in poses
        for point in pose
        if point.is_available(min_confidence)
    ]
    if not available_points:
        return (0.0, 1.0, 0.0, 1.0)

    x_values = [float(point.x) for point in available_points if point.x is not None]
    y_values = [float(point.y) for point in available_points if point.y is not None]
    return (min(x_values), max(x_values), min(y_values), max(y_values))


def _render_pose_group(
    pose: Sequence[SkeletonPoint2D],
    *,
    bounds: tuple[float, float, float, float],
    panel_origin_x: float,
    panel_width: int,
    height: int,
    padding: int,
    joint_radius: float,
    stroke_width: float,
    min_confidence: float,
) -> str:
    x_min, x_max, y_min, y_max = bounds
    span_x = max(x_max - x_min, 1e-9)
    span_y = max(y_max - y_min, 1e-9)
    scale = min(
        (panel_width - 2 * padding) / span_x,
        (height - 2 * padding) / span_y,
    )
    used_width = span_x * scale
    used_height = span_y * scale
    offset_x = panel_origin_x + (panel_width - used_width) / 2 - x_min * scale
    offset_y = (height - used_height) / 2 - y_min * scale

    available_points = {
        point.landmark_index: (
            offset_x + float(point.x) * scale,
            offset_y + float(point.y) * scale,
        )
        for point in pose
        if point.is_available(min_confidence)
    }

    parts = ["  <g>"]
    for start_index, end_index in DEFAULT_SKELETON_EDGES:
        start = available_points.get(start_index)
        end = available_points.get(end_index)
        if start is None or end is None:
            continue
        parts.append(
            "    <line "
            f"x1=\"{_format_number(start[0])}\" "
            f"y1=\"{_format_number(start[1])}\" "
            f"x2=\"{_format_number(end[0])}\" "
            f"y2=\"{_format_number(end[1])}\" "
            f"stroke=\"{_DEFAULT_STROKE}\" "
            f"stroke-width=\"{_format_number(stroke_width)}\" "
            "stroke-linecap=\"round\" />"
        )

    for landmark_index in sorted(available_points):
        x_value, y_value = available_points[landmark_index]
        parts.append(
            "    <circle "
            f"cx=\"{_format_number(x_value)}\" "
            f"cy=\"{_format_number(y_value)}\" "
            f"r=\"{_format_number(joint_radius)}\" "
            f"fill=\"{_DEFAULT_JOINT_FILL}\" />"
        )

    parts.append("  </g>")
    return "\n".join(parts)


def _compose_svg(
    *,
    panel_count: int,
    panel_width: int,
    height: int,
    rendered_groups: Sequence[str],
) -> str:
    width = panel_count * panel_width
    parts = [
        f"<svg xmlns=\"{_SVG_NAMESPACE}\" "
        f"viewBox=\"0 0 {width} {height}\" "
        f"width=\"{width}\" "
        f"height=\"{height}\">"
    ]
    parts.extend(rendered_groups)
    parts.append("</svg>")
    return "\n".join(parts)


def _format_number(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _validate_canvas(*, width: int, height: int, padding: int) -> None:
    if (
        isinstance(width, bool)
        or not isinstance(width, Integral)
        or width <= 0
    ):
        raise ValueError("width must be a positive integer")
    if (
        isinstance(height, bool)
        or not isinstance(height, Integral)
        or height <= 0
    ):
        raise ValueError("height must be a positive integer")
    if (
        isinstance(padding, bool)
        or not isinstance(padding, Integral)
        or padding < 0
    ):
        raise ValueError("padding must be a non-negative integer")
    if width <= 2 * padding or height <= 2 * padding:
        raise ValueError("padding leaves no drawable area")


def _validate_non_negative_number(value: float, name: str) -> None:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ValueError(f"{name} must be a finite, non-negative number")


def _validate_unit_interval(value: float, name: str) -> None:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{name} must be a finite number in [0, 1]")
