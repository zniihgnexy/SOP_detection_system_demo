"""实时检测编排（设计文档 §5.2 在线阶段流程）。

一帧走完 8 步：

    1 采集      → cv2.VideoCapture
    2 预处理    → perception.preprocess（ROI / 光照，默认关闭）
    3 目标检测  → YOLOv8n 零件框 + MediaPipe 手部关键点
    4 特征      → 298 维（含帧间速度、加速度，所以第 3、4 步天然含状态变化分析）
    5 动作分类  → 16 帧滑窗，每 stride=2 帧推理一次，其余帧复用上次结果
    6 状态机    → fsm.step 推进 SOP
    7 异常检查  → 由 fsm.step 内部完成（超时 / 零件 / 顺序）
    8 输出      → 叠加层 + 事件 + 装配完成时落库

一次装配结束（走到最后一步或视频结束）会自动落库并重置状态，接着检测下一支笔。

本模块 import torch / cv2 / mediapipe，因此只在真实检测时被加载 ——
``run.py --replay`` 和 ``--selfcheck`` 不会碰到它。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import fsm
from .features import FeatureExtractor, PartBox, SlidingWindow, dominant_part
from .fsm import MatchState, SOPTemplate
from .model import load_checkpoint
from .perception import Perception, draw_overlay, preprocess
from .records import Recorder


@dataclass(frozen=True)
class PipelineConfig:
    sop_path: str
    yolo_weights: str
    action_weights: str
    station_id: str = "ST01"
    db_path: str = "sop.db"
    device: str = "cpu"
    part_conf: float = 0.35
    roi: tuple[float, float, float, float] | None = None
    equalize: bool = False


@dataclass(frozen=True)
class FrameResult:
    """一帧的全部产出，Web 层和终端输出都从这里取数据。"""

    frame: np.ndarray
    timestamp_ms: int
    action: str
    confidence: float
    target_part: str | None
    parts: tuple[PartBox, ...]
    state: MatchState
    events: tuple[dict, ...] = field(default_factory=tuple)


class DetectionPipeline:
    """把感知、特征、动作识别、SOP 判定串起来。"""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.template: SOPTemplate = fsm.load_template(config.sop_path)

        # 两个权重都必须就位，缺任何一个都在这里报 FileNotFoundError，
        # 由 run.py 捕获后提示去手册第 5、6 章
        self.model, self.model_meta = load_checkpoint(
            config.action_weights, device=config.device
        )
        self.perception = Perception(
            yolo_weights=config.yolo_weights, part_conf=config.part_conf
        )

        self.extractor = FeatureExtractor()
        self.window = SlidingWindow()
        self.recorder = Recorder(config.db_path)
        self.template_id = self.recorder.upsert_template(self.template)

        self.state = fsm.new_state()
        self._last_action = "Idle"
        self._last_confidence = 0.0
        self._assembly_index = 0

    # ------------------------------------------------------------ 单帧

    def process_frame(self, frame_bgr: np.ndarray, timestamp_ms: int) -> FrameResult:
        processed = preprocess(
            frame_bgr, roi=self.config.roi, equalize=self.config.equalize
        )

        hand, parts = self.perception(processed)
        feature = self.extractor.extract(hand, parts)

        batch = self.window.push(feature)
        if batch is not None:
            index, confidence = self.model.predict(batch)
            self._last_action = fsm.ACTIONS[index]
            self._last_confidence = confidence

        target_part = dominant_part(hand, parts)
        detected = tuple(p.label for p in parts)

        observation = fsm.Observation(
            action=self._last_action,
            confidence=self._last_confidence,
            timestamp_ms=timestamp_ms,
            target_part=target_part,
            # 没有零件检测结果时传 None，让 fsm 跳过零件校验而不是误报
            parts_present=detected if parts else None,
        )
        self.state, events = fsm.step(self.template, self.state, observation)

        # 缺失零件只在真的接了零件检测时才提示，否则画面上会一直挂着红字
        missing: tuple[str, ...] = ()
        if parts:
            current = self.template.step_at(self.state.step_index)
            required = current.required_parts_present if current else ()
            missing = tuple(p for p in required if p not in detected)

        overlay = draw_overlay(
            processed, hand, parts,
            active_part=target_part,
            missing_parts=missing,
            action=self._last_action,
            action_confidence=self._last_confidence,
            step_label=self._step_label(),
            connections=self.perception.connections,
        )

        if any(e["type"] == "assembly_complete" for e in events):
            self._store_and_reset(timestamp_ms)

        return FrameResult(
            frame=overlay,
            timestamp_ms=timestamp_ms,
            action=self._last_action,
            confidence=self._last_confidence,
            target_part=target_part,
            parts=tuple(parts),
            state=self.state,
            events=tuple(events),
        )

    def _step_label(self) -> str:
        """左上角叠加文字。cv2 画不了中文，所以只放步骤号和英文动作名。"""
        total = len(self.template.steps)
        current = self.template.step_at(self.state.step_index)
        if current is None:
            return f"DONE {total}/{total}"
        return f"{current.id} {self.state.step_index + 1}/{total} {current.expected_action}"

    # ------------------------------------------------- 一次装配的生命周期

    def finish_assembly(self, timestamp_ms: int) -> tuple[MatchState, list[dict]]:
        """视频结束或手动收尾时补一次完整性检查并落库。

        一支笔刚好在最后一帧装完时，状态已经被 _store_and_reset 重置成全新的了，
        这时候不能再 finalize —— 否则会凭空记一条「六步全漏」的 FAIL。
        """
        if self.state.finished:
            return self.state, []
        if self.state.step_index == 0 and not self.state.completed:
            return self.state, []          # 这一轮还没开始，没有东西可收尾

        self.state, events = fsm.finalize(self.template, self.state, timestamp_ms)
        self._store_and_reset(timestamp_ms)
        return self.state, events

    def _store_and_reset(self, timestamp_ms: int) -> int:
        record_id = self.recorder.save_record(
            station_id=self.config.station_id,
            template_id=self.template_id,
            result="PASS" if self.state.passed else "FAIL",
            anomalies=self.state.anomalies,
            steps_completed=self.state.completed,
            duration_ms=timestamp_ms - self.state.started_ms,
        )
        self._assembly_index += 1
        # 重置，准备检测下一支笔。特征提取器也要清，否则速度会跨装配串味
        self.state = fsm.new_state(timestamp_ms)
        self.extractor.reset()
        self.window.reset()
        return record_id

    def summary(self) -> dict:
        return self.recorder.summary(self.config.station_id)

    def close(self) -> None:
        self.perception.close()
        self.recorder.close()

    def __enter__(self) -> "DetectionPipeline":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ------------------------------------------------------- 视频循环

    def run_headless(self, source: str | int, verbose: bool = True) -> int:
        """不开界面，跑完一个视频或摄像头流，结果打印到终端。"""
        capture, timestamp_of = open_source(source)
        if not capture.isOpened():
            print(f"错误：打不开视频源 {source!r}")
            return 2

        print(f"工位 {self.config.station_id}  模板 {self.template.model_name} "
              f"{self.template.version}  {len(self.template.steps)} 个步骤")
        print(f"视频源 {source!r}   记录库 {self.config.db_path}\n")

        frame_index = 0
        timestamp_ms = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp_ms = timestamp_of(capture, frame_index)
                result = self.process_frame(frame, timestamp_ms)
                frame_index += 1

                if verbose:
                    for event in result.events:
                        line = fsm.format_event(event, timestamp_ms)
                        if line:
                            print(line)
                        if event["type"] == "assembly_complete":
                            print(f"  ── 第 {self._assembly_index} 支："
                                  f"{_summarize(event)}\n")
        finally:
            capture.release()
            _, events = self.finish_assembly(timestamp_ms)
            if verbose:
                for event in events:
                    if event["type"] == "assembly_complete":
                        print(f"  ── 视频结束时这支笔尚未装完："
                              f"{_summarize(event)}")

        stats = self.summary()
        rate = ""
        if stats["pass_rate"] is not None:
            rate = f"（{stats['pass_rate'] * 100:.1f}%）"
        print(f"\n共 {frame_index} 帧。工位累计 {stats['total']} 次装配，"
              f"合格 {stats['passed']} 次{rate}")
        print(f"查看记录：sqlite3 {self.config.db_path} "
              f"\"SELECT id,timestamp,result,anomaly_type FROM detection_records;\"")
        return 0


def _summarize(event: dict) -> str:
    """assembly_complete 事件的一行摘要。"""
    parts = [event["result"], f"总耗时 {event['total_duration_ms'] / 1000:.1f}s"]
    if event["anomaly_types"]:
        parts.append("异常：" + "、".join(event["anomaly_types"]))
    return "  ".join(parts)


def open_source(source: str | int):
    """打开视频源，返回 (capture, 时间戳函数)。

    视频文件按帧号乘帧间隔算时间戳（可复现）；摄像头按墙上时钟算（实时）。
    """
    if isinstance(source, str) and not source.isdigit():
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"找不到视频文件 {path}")
        capture = cv2.VideoCapture(str(path))
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        interval = 1000.0 / fps

        def timestamp_of(_capture, frame_index: int) -> int:
            return int(frame_index * interval)
    else:
        capture = cv2.VideoCapture(int(source))
        start = time.monotonic()

        def timestamp_of(_capture, _frame_index: int) -> int:
            return int((time.monotonic() - start) * 1000)

    return capture, timestamp_of
