"""298 维特征工程测试（sop/features.py）。只需要 numpy。"""

from __future__ import annotations

import numpy as np
import pytest

from sop import features as F
from sop.features import FeatureExtractor, HandLandmarks, PartBox, SlidingWindow


def make_hand(shift: float = 0.0, spread: float = 1.0) -> HandLandmarks:
    """造一只合法的手：21 点沿对角线均匀排开，下标 0 是手腕。"""
    points = tuple(
        (0.5 + shift + i * 0.01 * spread, 0.5 + shift + i * 0.002 * spread, 0.0)
        for i in range(F.NUM_LANDMARKS)
    )
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return HandLandmarks(
        points=points, score=0.9, handedness="Right",
        xyxy=(min(xs), min(ys), max(xs), max(ys)),
    )


def part(label: str, cx: float, cy: float, size: float = 0.05, conf: float = 0.8):
    half = size / 2
    return PartBox(label=label, confidence=conf,
                   xyxy=(cx - half, cy - half, cx + half, cy + half))


# ------------------------------------------------------------------ 布局


def test_dimension_is_298_and_offsets_are_contiguous():
    assert F.FEATURE_DIM == 298
    assert FeatureExtractor.DIM == 298
    expected = [
        ("relative", 0, 63), ("raw", 63, 63), ("hand_part", 126, 12),
        ("grip", 138, 2), ("velocity", 140, 63), ("acceleration", 203, 63),
        ("part_embed", 266, 32),
    ]
    for name, offset, _ in expected:
        assert F.OFFSETS[name] == offset, name
    assert sum(size for _, _, size in expected) == 298


def test_window_constants_match_design_doc():
    assert F.WINDOW_SIZE == 16
    assert F.STRIDE == 2
    assert F.TOP_K_PARTS == 4
    assert F.GRIP_THRESHOLD == 0.05


def test_extract_shape_and_dtype():
    vector = FeatureExtractor().extract(make_hand(), [])
    assert vector.shape == (298,)
    assert vector.dtype == np.float32


def test_no_hand_returns_zeros():
    extractor = FeatureExtractor()
    extractor.extract(make_hand(), [])
    vector = extractor.extract(None, [part("barrel", 0.5, 0.5)])
    assert np.count_nonzero(vector) == 0


def test_rejects_wrong_landmark_shape():
    broken = HandLandmarks(points=((0.1, 0.2, 0.0),), score=0.9,
                           handedness="Right", xyxy=(0, 0, 1, 1))
    with pytest.raises(ValueError, match=r"\(21, 3\)"):
        FeatureExtractor().extract(broken, [])


# -------------------------------------------------------------- 手部特征


def test_relative_coordinates_zero_at_wrist():
    vector = FeatureExtractor().extract(make_hand(shift=0.3), [])
    start = F.OFFSETS["relative"]
    assert np.allclose(vector[start:start + 3], 0.0)
    # 第 1 个点相对手腕应是 (0.01, 0.002, 0)
    assert np.allclose(vector[start + 3:start + 6], [0.01, 0.002, 0.0], atol=1e-6)


def test_raw_coordinates_preserved():
    hand = make_hand(shift=0.2)
    vector = FeatureExtractor().extract(hand, [])
    start = F.OFFSETS["raw"]
    assert np.allclose(vector[start:start + 3], hand.points[0], atol=1e-6)


def test_velocity_zero_on_first_frame_then_tracks_motion():
    extractor = FeatureExtractor()
    start = F.OFFSETS["velocity"]

    first = extractor.extract(make_hand(shift=0.0), [])
    assert np.allclose(first[start:start + 63], 0.0)

    second = extractor.extract(make_hand(shift=0.1), [])
    assert np.allclose(second[start:start + 3], [0.1, 0.1, 0.0], atol=1e-5)


def test_acceleration_is_velocity_difference():
    extractor = FeatureExtractor()
    start = F.OFFSETS["acceleration"]

    extractor.extract(make_hand(shift=0.0), [])
    extractor.extract(make_hand(shift=0.1), [])          # v = 0.1
    third = extractor.extract(make_hand(shift=0.3), [])  # v = 0.2 → a = 0.1
    assert np.allclose(third[start:start + 3], [0.1, 0.1, 0.0], atol=1e-5)


def test_reset_clears_motion_history():
    extractor = FeatureExtractor()
    extractor.extract(make_hand(shift=0.0), [])
    extractor.reset()
    vector = extractor.extract(make_hand(shift=0.9), [])
    start = F.OFFSETS["velocity"]
    assert np.allclose(vector[start:start + 63], 0.0)


def test_grip_features():
    start = F.OFFSETS["grip"]

    closed = FeatureExtractor().extract(make_hand(spread=1.0), [])
    distance = closed[start]
    assert distance == pytest.approx(0.040792, abs=1e-4)
    assert closed[start + 1] == 1.0          # 小于 0.05 → 抓取

    opened = FeatureExtractor().extract(make_hand(spread=4.0), [])
    assert opened[start] > F.GRIP_THRESHOLD
    assert opened[start + 1] == 0.0


# -------------------------------------------------------------- 零件特征


def test_hand_part_features_zero_without_parts():
    vector = FeatureExtractor().extract(make_hand(), [])
    spatial = vector[F.OFFSETS["hand_part"]:F.OFFSETS["hand_part"] + 12]
    embed = vector[F.OFFSETS["part_embed"]:]
    assert np.count_nonzero(spatial) == 0
    assert np.count_nonzero(embed) == 0


def test_part_embedding_layout():
    """4 槽 × 6 类 one-hot（0..23）+ 4 个置信度（24..27）+ 4 个补零（28..31）。"""
    hand = make_hand()
    hx, hy = hand.palm
    parts = [
        part("barrel", hx, hy, conf=0.90),            # 最近
        part("spring", hx + 0.10, hy, conf=0.80),
        part("refill", hx + 0.20, hy, conf=0.70),
        part("tip", hx + 0.30, hy, conf=0.60),
        part("cap", hx + 0.40, hy, conf=0.50),        # 第 5 近，应被丢弃
    ]
    vector = FeatureExtractor().extract(hand, parts)
    embed = vector[F.OFFSETS["part_embed"]:]

    assert embed.shape == (32,)
    for slot, label in enumerate(["barrel", "spring", "refill", "tip"]):
        one_hot = embed[slot * 6:(slot + 1) * 6]
        assert one_hot.sum() == 1.0
        assert one_hot[F.PARTS.index(label)] == 1.0

    assert np.allclose(embed[24:28], [0.90, 0.80, 0.70, 0.60], atol=1e-6)
    assert np.count_nonzero(embed[28:32]) == 0        # 末尾补零


def test_hand_part_spatial_is_relative_offset_and_iou():
    hand = make_hand()
    hx, hy = hand.palm
    near = part("barrel", hx + 0.10, hy + 0.05, size=0.02)
    vector = FeatureExtractor().extract(hand, [near])

    start = F.OFFSETS["hand_part"]
    assert vector[start] == pytest.approx(0.10, abs=1e-5)
    assert vector[start + 1] == pytest.approx(0.05, abs=1e-5)
    assert 0.0 <= vector[start + 2] <= 1.0


def test_rank_parts_returns_nearest_four():
    hand = make_hand()
    hx, hy = hand.palm
    parts = [part(f"p{i}", hx + i * 0.05, hy) for i in range(6)]
    ranked = F.rank_parts(hand, parts)
    assert [p.label for p in ranked] == ["p0", "p1", "p2", "p3"]


def test_unknown_part_label_does_not_break_one_hot():
    hand = make_hand()
    hx, hy = hand.palm
    vector = FeatureExtractor().extract(hand, [part("finished_pen", hx, hy, conf=0.7)])
    embed = vector[F.OFFSETS["part_embed"]:]
    assert embed[0:6].sum() == 0.0        # 不在 6 类里，one-hot 全零
    assert embed[24] == pytest.approx(0.7, abs=1e-6)


# ------------------------------------------------------------------- IoU


def test_iou_identical_boxes():
    assert F.iou((0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0)) == pytest.approx(1.0)


def test_iou_disjoint_boxes():
    assert F.iou((0.0, 0.0, 0.1, 0.1), (0.5, 0.5, 0.6, 0.6)) == 0.0


def test_iou_touching_edges_is_zero():
    assert F.iou((0.0, 0.0, 0.5, 0.5), (0.5, 0.0, 1.0, 0.5)) == 0.0


def test_iou_quarter_overlap():
    # 两个边长 0.2 的框错开一半 → 交 0.1×0.2，并 2×0.04−0.02
    value = F.iou((0.0, 0.0, 0.2, 0.2), (0.1, 0.0, 0.3, 0.2))
    assert value == pytest.approx(0.02 / 0.06, abs=1e-6)


# --------------------------------------------------------- dominant_part


def test_dominant_part_prefers_overlapping_box():
    hand = make_hand()
    hx, hy = hand.palm
    overlapping = part("refill", hx + 0.02, hy, size=0.3)   # 与手部框相交
    far = part("cap", 0.95, 0.95, size=0.02)
    assert F.dominant_part(hand, [far, overlapping]) == "refill"


def test_dominant_part_falls_back_to_nearest_when_no_overlap():
    hand = make_hand()
    near = part("spring", 0.05, 0.05, size=0.01)
    farther = part("cap", 0.99, 0.99, size=0.01)
    assert F.dominant_part(hand, [farther, near]) == "spring"


def test_dominant_part_none_without_hand_or_parts():
    assert F.dominant_part(None, [part("barrel", 0.5, 0.5)]) is None
    assert F.dominant_part(make_hand(), []) is None


# ------------------------------------------------------------ 滑动窗口


def test_sliding_window_emits_every_stride_frames():
    window = SlidingWindow(window_size=4, stride=2)
    shapes = [window.push(np.full(298, i, dtype=np.float32)) for i in range(6)]
    assert [s is None for s in shapes] == [True, False, True, False, True, False]


def test_sliding_window_pads_with_first_frame_when_short():
    window = SlidingWindow(window_size=4, stride=1)
    window.push(np.full(298, 7.0, dtype=np.float32))
    batch = window.push(np.full(298, 9.0, dtype=np.float32))

    assert batch.shape == (4, 298)
    assert batch[0][0] == 7.0 and batch[1][0] == 7.0 and batch[2][0] == 7.0
    assert batch[3][0] == 9.0


def test_sliding_window_keeps_only_last_n_frames():
    window = SlidingWindow(window_size=3, stride=1)
    for value in range(5):
        batch = window.push(np.full(298, value, dtype=np.float32))
    assert [row[0] for row in batch] == [2.0, 3.0, 4.0]


def test_sliding_window_reset():
    window = SlidingWindow(window_size=4, stride=1)
    for _ in range(3):
        window.push(np.zeros(298, dtype=np.float32))
    window.reset()
    batch = window.push(np.full(298, 5.0, dtype=np.float32))
    assert all(row[0] == 5.0 for row in batch)


def test_iter_windows():
    sequence = np.arange(10 * 298, dtype=np.float32).reshape(10, 298)
    windows = list(F.iter_windows(sequence, window_size=4, stride=2))
    assert [start for start, _ in windows] == [0, 2, 4, 6]
    assert all(chunk.shape == (4, 298) for _, chunk in windows)


def test_iter_windows_shorter_than_window_yields_nothing():
    sequence = np.zeros((3, 298), dtype=np.float32)
    assert list(F.iter_windows(sequence, window_size=16, stride=2)) == []
