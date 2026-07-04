from ssbd_behavior.acquisition import SSBDPlusSegment
from ssbd_behavior.features import WindowSpec, generate_windows_for_video


def _segment(video_id, start, end, category="armflapping"):
    return SSBDPlusSegment(
        video_id=video_id,
        url="https://example.invalid/synthetic",
        start_time=start,
        end_time=end,
        category=category,
    )


def test_window_generation_count():
    windows = generate_windows_for_video(
        "video-1", 5.0, [], WindowSpec(window_size_s=2.0, stride_s=1.0)
    )

    assert [(window.start_s, window.end_s) for window in windows] == [
        (0.0, 2.0),
        (1.0, 3.0),
        (2.0, 4.0),
        (3.0, 5.0),
    ]


def test_window_labels_use_minimum_annotation_overlap():
    windows = generate_windows_for_video(
        "video-1",
        5.0,
        [_segment("video-1", 1, 2)],
        WindowSpec(window_size_s=2.0, stride_s=1.0),
        minimum_overlap_fraction=0.5,
    )

    assert [window.label for window in windows] == [1, 1, 0, 0]
    assert [window.annotated_overlap_fraction for window in windows] == [
        0.5,
        0.5,
        0.0,
        0.0,
    ]


def test_no_class_annotations_are_negative():
    windows = generate_windows_for_video(
        "video-1",
        2.0,
        [_segment("video-1", 0, 2, category="no-class")],
        WindowSpec(window_size_s=2.0, stride_s=1.0),
    )

    assert [window.label for window in windows] == [0]


def test_windows_do_not_cross_video_boundaries():
    annotations = [
        _segment("video-1", 2, 4),
        _segment("video-2", 0, 2),
    ]
    spec = WindowSpec(window_size_s=2.0, stride_s=2.0)

    video_1_windows = generate_windows_for_video("video-1", 3.0, annotations, spec)
    video_2_windows = generate_windows_for_video("video-2", 2.0, annotations, spec)

    assert [(window.video_id, window.start_s, window.end_s) for window in video_1_windows] == [
        ("video-1", 0.0, 2.0)
    ]
    assert [(window.video_id, window.start_s, window.end_s) for window in video_2_windows] == [
        ("video-2", 0.0, 2.0)
    ]
    assert [window.label for window in video_1_windows] == [0]
    assert [window.label for window in video_2_windows] == [1]
