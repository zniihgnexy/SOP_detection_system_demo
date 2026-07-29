"""感知层：YOLOv8n 零件检测 + MediaPipe 手部关键点（设计文档 §4.1 / §4.2.3）。

本模块是唯一 import ultralytics / mediapipe / cv2 的地方，被 pipeline 延迟导入，
所以 ``run.py --replay`` 和 ``--selfcheck`` 不受这些重依赖影响。

主操作手的选择：MediaPipe 按 §4.2.3 配置为 ``max_num_hands=2``（双手都检测，
这样握持手和操作手同时在画面里也不会丢），但特征只用置信度最高的那只 ——
原因见 sop/features.py 模块注释第 1 条。

叠加层文字用英文：cv2.putText 不支持中文字形，会画成一串方框。中文步骤名在
Web 页面里显示（§8.3.2 的叠加元素表），画面上只叠加步骤号和英文动作名。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .features import HandLandmarks, PartBox, WRIST

# §8.3.2 叠加元素配色，OpenCV 用 BGR
COLOR_PART = (83, 200, 0)          # #00C853 绿：已就绪的零件
COLOR_ACTIVE = (255, 121, 41)      # #2979FF 蓝：正在被操作的零件
COLOR_MISSING = (68, 23, 255)      # #FF1744 红：应出现但没检测到
COLOR_SKELETON = (212, 188, 0)     # #00BCD4 青：手部骨架连线
COLOR_LANDMARK = (255, 255, 255)   # 白：关键点
COLOR_TEXT = (255, 255, 255)


class Perception:
    """一帧进，(主手关键点, 零件框列表) 出。

    ``yolo_weights`` 传 None 时不做零件检测，零件列表恒为空 —— 此时 298 维特征里
    的 44 维零件相关分量为零，SOP 判定退化成只按动作匹配。
    """

    def __init__(
        self,
        yolo_weights: str | Path | None = None,
        part_conf: float = 0.35,
        max_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        import mediapipe as mp

        self._mp_hands = mp.solutions.hands
        self._connections = self._mp_hands.HAND_CONNECTIONS
        # 参数照抄 §4.2.3 的提取配置
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.part_conf = part_conf
        self._yolo = None
        if yolo_weights is not None:
            path = Path(yolo_weights)
            if not path.is_file():
                raise FileNotFoundError(
                    f"找不到零件检测权重 {path}。需要先按 "
                    f"docs/user-manual-zh.md 第 5 章训练 YOLOv8n。"
                )
            from ultralytics import YOLO

            self._yolo = YOLO(str(path))

    # --- 推理 ---

    def __call__(self, frame_bgr: np.ndarray) -> tuple[HandLandmarks | None, list[PartBox]]:
        return self.detect_hand(frame_bgr), self.detect_parts(frame_bgr)

    def detect_hand(self, frame_bgr: np.ndarray) -> HandLandmarks | None:
        """返回主操作手（置信度最高的那只），没检测到手时返回 None。"""
        result = self._hands.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not result.multi_hand_landmarks:
            return None

        best: HandLandmarks | None = None
        for index, landmarks in enumerate(result.multi_hand_landmarks):
            score, label = 1.0, "Unknown"
            if result.multi_handedness and index < len(result.multi_handedness):
                classification = result.multi_handedness[index].classification[0]
                score, label = float(classification.score), classification.label

            points = tuple(
                (float(p.x), float(p.y), float(p.z)) for p in landmarks.landmark
            )
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            hand = HandLandmarks(
                points=points,
                score=score,
                handedness=label,
                xyxy=(min(xs), min(ys), max(xs), max(ys)),
            )
            if best is None or hand.score > best.score:
                best = hand

        return best

    def detect_parts(self, frame_bgr: np.ndarray) -> list[PartBox]:
        """返回归一化坐标的零件框。没有 YOLO 权重时返回空列表。"""
        if self._yolo is None:
            return []

        result = self._yolo.predict(
            frame_bgr, conf=self.part_conf, verbose=False
        )[0]
        names = result.names

        parts: list[PartBox] = []
        for box in result.boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxyn[0].tolist())
            parts.append(PartBox(
                label=str(names[int(box.cls)]),
                confidence=float(box.conf),
                xyxy=(x1, y1, x2, y2),
            ))
        return parts

    def close(self) -> None:
        self._hands.close()

    def __enter__(self) -> "Perception":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


# ------------------------------------------------------------- 图像预处理


def preprocess(
    frame_bgr: np.ndarray,
    roi: tuple[float, float, float, float] | None = None,
    equalize: bool = False,
) -> np.ndarray:
    """图像预处理层（§3.1 第 2 层），两项都默认关闭。

    ``roi``      归一化的 (x1, y1, x2, y2)，裁掉工作区以外的背景。
                 裁剪后所有归一化坐标都相对裁剪结果，叠加层也画在同一张图上。
    ``equalize`` 对 LAB 的 L 通道做 CLAHE，抵消环境光波动（§3.1「光照补偿」）。

    文档里还提到高斯/中值去噪，这里没做：YOLO 与 MediaPipe 内部都有自己的
    归一化，工业相机的传感器噪声通常靠调曝光和增益解决，先加滤波容易反而
    磨掉笔尖这类小目标的边缘。真需要时在这里加一行 cv2.GaussianBlur 即可。
    """
    out = frame_bgr

    if roi is not None:
        height, width = out.shape[:2]
        x1, y1, x2, y2 = roi
        out = out[
            int(y1 * height):int(y2 * height),
            int(x1 * width):int(x2 * width),
        ]
        if out.size == 0:
            raise ValueError(f"ROI {roi} 裁出了空图像，检查取值是否在 0~1 之间")

    if equalize:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        lightness, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        out = cv2.cvtColor(
            cv2.merge((clahe.apply(lightness), a_channel, b_channel)),
            cv2.COLOR_LAB2BGR,
        )

    return out


# --------------------------------------------------------------- 叠加绘制


def draw_overlay(
    frame_bgr: np.ndarray,
    hand: HandLandmarks | None,
    parts: list[PartBox] | tuple[PartBox, ...],
    *,
    active_part: str | None = None,
    missing_parts: tuple[str, ...] = (),
    action: str | None = None,
    action_confidence: float | None = None,
    step_label: str | None = None,
    connections=None,
) -> np.ndarray:
    """按 §8.3.2 的叠加元素表画检测结果。返回新图，不改原图。"""
    canvas = frame_bgr.copy()
    height, width = canvas.shape[:2]

    for part in parts:
        x1, y1, x2, y2 = part.xyxy
        pt1 = (int(x1 * width), int(y1 * height))
        pt2 = (int(x2 * width), int(y2 * height))
        is_active = active_part is not None and part.label == active_part
        color = COLOR_ACTIVE if is_active else COLOR_PART
        cv2.rectangle(canvas, pt1, pt2, color, 2 if is_active else 1)
        cv2.putText(
            canvas, f"{part.label} {part.confidence:.2f}",
            (pt1[0], max(12, pt1[1] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
        )

    if hand is not None:
        pixels = [(int(p[0] * width), int(p[1] * height)) for p in hand.points]
        for start, end in (connections or ()):
            if start < len(pixels) and end < len(pixels):
                cv2.line(canvas, pixels[start], pixels[end], COLOR_SKELETON, 2, cv2.LINE_AA)
        for index, point in enumerate(pixels):
            # 指尖画大一点（§8.3.2：较大圆为指尖）
            radius = 4 if index in (4, 8, 12, 16, 20) else 2
            cv2.circle(canvas, point, radius, COLOR_LANDMARK, -1, cv2.LINE_AA)
        cv2.circle(canvas, pixels[WRIST], 5, COLOR_SKELETON, -1, cv2.LINE_AA)

    if missing_parts:
        cv2.putText(
            canvas, f"MISSING: {','.join(missing_parts)}",
            (8, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            COLOR_MISSING, 1, cv2.LINE_AA,
        )

    if step_label:                                  # 左上角：SOP 步骤（§8.3.2）
        cv2.putText(canvas, step_label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, COLOR_TEXT, 2, cv2.LINE_AA)

    if action:                                      # 右上角：当前动作（§8.3.2）
        text = action
        if action_confidence is not None:
            text = f"{action} {action_confidence * 100:.0f}%"
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.putText(canvas, text, (width - size[0] - 8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2, cv2.LINE_AA)

    return canvas
