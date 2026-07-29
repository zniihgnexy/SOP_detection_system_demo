"""298 维特征工程（设计文档 §4.2.4）。

每帧特征向量的构成，顺序与文档 §4.2.4 末尾的汇总表**完全一致**：

    偏移      维度   内容
    ------   ----   --------------------------------------------------
      0..62    63   手部关键点，相对手腕（landmark[0]）→ 姿态不变性
     63..125   63   手部关键点，原始归一化坐标
    126..137   12   手-零件空间关系：最近 4 个零件 ×（dx, dy, IoU）
    138..139    2   手指开合度：拇指-食指距离 + 阈值二值化的抓取态
    140..202   63   帧间速度   v(t) = p(t) - p(t-1)
    203..265   63   帧间加速度 a(t) = v(t) - v(t-1)
    266..297   32   零件类别嵌入
    ------   ----
             298

对设计文档未定项的实现选择
--------------------------
1. **只取一只手**。文档 §4.2.4 的分解按单手 21×3=63 推算（63×4 + 12 + 2 + 32
   = 298），而 §4.2.3 又配置了 ``max_num_hands=2``。两者只能取一个：这里取
   **单手**，选 MediaPipe 置信度最高的那只作为主操作手。理由是 298 这个数字在
   §4.2.5 的模型输入、§4.2.7 的 ONNX 形状、§6.2 的延迟表里被反复引用，
   改成双手（~550 维）要连带推翻三处。双手仍然检测，只是特征只用主手。
2. **32 维零件类别嵌入的算法**文档没定。这里用最确定的做法：
   最近 4 个零件各一个 6 类 one-hot（4×6=24 维）+ 各自的检测置信度（4 维）
   = 28 维，末尾补 4 个零到 32。**不**从 YOLO 内部取 embedding —— 那要改检测器，
   而且会破坏「298 维是纯预处理输出」这个边界，§4.2.7 的 ONNX 输入形状依赖它。
3. **没检测到手时**返回全零向量并清空速度链。这样动作模型会看到一段静止信号，
   通常被判为 Idle，符合「手不在工作区就是闲置」的定义（§4.2.1）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fsm import PARTS

# --------------------------------------------------------------------- 常量

#: 滑动窗口长度与步长（§4.2.4）：16 帧 ≈ 0.64 秒 @25fps
WINDOW_SIZE = 16
STRIDE = 2

#: 每帧特征维度（§4.2.4 汇总表）
FEATURE_DIM = 298

#: 参与手-零件关系计算的最近零件个数（§4.2.4）
TOP_K_PARTS = 4

#: 抓取判定阈值：拇指-食指归一化距离小于此值视为抓取（§4.2.4）
GRIP_THRESHOLD = 0.05

#: 零件类别嵌入维度
PART_EMBED_DIM = 32

# MediaPipe 21 点里用到的下标（§4.2.3 拓扑图）
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
NUM_LANDMARKS = 21

# 各特征组在 298 维向量里的起始偏移，供调试和可视化使用
OFFSETS = {
    "relative": 0,
    "raw": 63,
    "hand_part": 126,
    "grip": 138,
    "velocity": 140,
    "acceleration": 203,
    "part_embed": 266,
}


# ----------------------------------------------------------------- 数据类型


@dataclass(frozen=True)
class PartBox:
    """一个零件检测框。坐标都是相对图像尺寸的 0~1 归一化值。"""

    label: str
    confidence: float
    xyxy: tuple[float, float, float, float]

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass(frozen=True)
class HandLandmarks:
    """一只手的 21 个关键点。points 是 21×3 的归一化 (x, y, z)。"""

    points: tuple[tuple[float, float, float], ...]
    score: float
    handedness: str
    xyxy: tuple[float, float, float, float]

    @property
    def palm(self) -> tuple[float, float]:
        return (self.points[WRIST][0], self.points[WRIST][1])


# --------------------------------------------------------------- 特征提取


class FeatureExtractor:
    """把每帧的（主手关键点, 零件框列表）转成 298 维特征。

    有状态：速度和加速度需要前两帧。换视频或换一次装配时记得 ``reset()``。
    """

    DIM = FEATURE_DIM

    def __init__(self) -> None:
        self._prev_points: np.ndarray | None = None
        self._prev_velocity: np.ndarray | None = None

    def reset(self) -> None:
        self._prev_points = None
        self._prev_velocity = None

    def extract(
        self, hand: HandLandmarks | None, parts: list[PartBox] | tuple[PartBox, ...] = ()
    ) -> np.ndarray:
        """返回形状 (298,) 的 float32 向量。"""
        if hand is None:
            self.reset()
            return np.zeros(FEATURE_DIM, dtype=np.float32)

        points = np.asarray(hand.points, dtype=np.float32)      # (21, 3)
        if points.shape != (NUM_LANDMARKS, 3):
            raise ValueError(
                f"手部关键点形状应为 (21, 3)，实际是 {points.shape}"
            )

        raw = points.reshape(-1)                                # 63
        relative = (points - points[WRIST]).reshape(-1)         # 63

        if self._prev_points is None:
            velocity = np.zeros(NUM_LANDMARKS * 3, dtype=np.float32)
        else:
            velocity = (points - self._prev_points).reshape(-1)

        if self._prev_velocity is None:
            acceleration = np.zeros(NUM_LANDMARKS * 3, dtype=np.float32)
        else:
            acceleration = velocity - self._prev_velocity

        hand_part, part_embed = _part_features(hand, parts)
        grip = _grip_features(points)

        self._prev_points = points
        self._prev_velocity = velocity

        feature = np.concatenate(
            [relative, raw, hand_part, grip, velocity, acceleration, part_embed]
        ).astype(np.float32)

        assert feature.shape == (FEATURE_DIM,), feature.shape
        return feature


def _grip_features(points: np.ndarray) -> np.ndarray:
    """手指开合度（§4.2.4 第 3 组）：拇指-食指距离 + 二值抓取态。"""
    distance = float(np.linalg.norm(points[THUMB_TIP] - points[INDEX_TIP]))
    return np.array(
        [distance, 1.0 if distance < GRIP_THRESHOLD else 0.0], dtype=np.float32
    )


def _part_features(
    hand: HandLandmarks, parts: list[PartBox] | tuple[PartBox, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """手-零件空间关系（12 维）与零件类别嵌入（32 维）。

    没有零件检测结果时两者都是全零，动作模型仍可依靠 254 维手部特征工作。
    """
    spatial = np.zeros(TOP_K_PARTS * 3, dtype=np.float32)
    embed = np.zeros(PART_EMBED_DIM, dtype=np.float32)
    if not parts:
        return spatial, embed

    for slot, part in enumerate(rank_parts(hand, parts)):
        px, py = part.center
        hx, hy = hand.palm
        spatial[slot * 3 + 0] = px - hx
        spatial[slot * 3 + 1] = py - hy
        spatial[slot * 3 + 2] = iou(hand.xyxy, part.xyxy)

        if part.label in PARTS:
            embed[slot * len(PARTS) + PARTS.index(part.label)] = 1.0
        embed[TOP_K_PARTS * len(PARTS) + slot] = part.confidence

    return spatial, embed


def rank_parts(
    hand: HandLandmarks, parts: list[PartBox] | tuple[PartBox, ...]
) -> list[PartBox]:
    """按手掌中心到零件中心的距离排序，取最近的 TOP_K_PARTS 个（§4.2.4）。"""
    hx, hy = hand.palm
    ordered = sorted(
        parts,
        key=lambda p: (p.center[0] - hx) ** 2 + (p.center[1] - hy) ** 2,
    )
    return ordered[:TOP_K_PARTS]


def dominant_part(
    hand: HandLandmarks | None, parts: list[PartBox] | tuple[PartBox, ...]
) -> str | None:
    """判断当前正在操作哪个零件，作为 Observation.target_part。

    优先取与手部框 IoU 最大的零件（说明手正握着它）；都不相交时退回距离最近的。
    动作模型只输出动作类别，目标零件必须由零件检测提供 —— 所以没有 YOLO 权重时
    这里返回 None，SOP 判定会退化成只按动作匹配。
    """
    if hand is None or not parts:
        return None

    overlapping = [(iou(hand.xyxy, p.xyxy), p) for p in parts]
    best_iou, best_part = max(overlapping, key=lambda pair: pair[0])
    if best_iou > 0.0:
        return best_part.label

    nearest = rank_parts(hand, parts)
    return nearest[0].label if nearest else None


def iou(
    box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]
) -> float:
    """两个 xyxy 框的交并比。"""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    if inter_w <= 0.0 or inter_h <= 0.0:
        return 0.0

    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


# --------------------------------------------------------------- 滑动窗口


class SlidingWindow:
    """在线推理用的特征缓冲（§4.2.7）。

    每帧 ``push`` 一个 298 维特征；每 ``stride`` 帧返回一个 (window_size, 298)
    的窗口，其余时候返回 None（表示这一帧不用推理，沿用上次结果）。
    缓冲不足时复制首帧补齐，与 §4.2.7 的 ``_pad_window`` 一致。
    """

    def __init__(self, window_size: int = WINDOW_SIZE, stride: int = STRIDE) -> None:
        self.window_size = window_size
        self.stride = stride
        self._buffer: list[np.ndarray] = []
        self._counter = 0

    def reset(self) -> None:
        self._buffer.clear()
        self._counter = 0

    def push(self, feature: np.ndarray) -> np.ndarray | None:
        self._buffer.append(np.asarray(feature, dtype=np.float32))
        if len(self._buffer) > self.window_size:
            self._buffer.pop(0)

        self._counter += 1
        if self._counter % self.stride != 0:
            return None

        if len(self._buffer) < self.window_size:
            pad = [self._buffer[0]] * (self.window_size - len(self._buffer))
            return np.stack(pad + self._buffer)
        return np.stack(self._buffer)


def iter_windows(
    features: np.ndarray, window_size: int = WINDOW_SIZE, stride: int = STRIDE
):
    """训练用：把一段特征序列切成 (起始下标, 窗口) 对。

    ``features`` 形状 (T, 298)。不足一个窗口的尾部丢弃。
    """
    total = len(features)
    for start in range(0, max(0, total - window_size + 1), stride):
        yield start, features[start:start + window_size]
