"""编排层与界面状态测试（sop/pipeline.py、web/server.py）。

需要 cv2 和 torch。没装的机器上整个文件跳过。
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("cv2", reason="未安装 opencv-python，跳过编排层测试")
pytest.importorskip("torch", reason="未安装 torch，跳过编排层测试")

from sop import fsm                                              # noqa: E402
from sop.features import PartBox                                 # noqa: E402
from sop.pipeline import PipelineConfig, _summarize, open_source  # noqa: E402
from web.server import LiveState, build_snapshot                 # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = fsm.load_template(ROOT / "configs" / "gelpen_0.5.json")


# --------------------------------------------------------------- 配置


def test_pipeline_config_defaults():
    config = PipelineConfig(
        sop_path="configs/gelpen_0.5.json",
        yolo_weights="models/pen_parts.pt",
        action_weights="models/action_model.pt",
    )
    assert config.station_id == "ST01"
    assert config.db_path == "sop.db"
    assert config.roi is None and config.equalize is False


def test_open_source_missing_file():
    with pytest.raises(FileNotFoundError, match="找不到视频文件"):
        open_source("no_such_video.mp4")


def test_summarize_pass_and_fail():
    clean = _summarize({
        "result": "PASS", "total_duration_ms": 7600, "anomaly_types": [],
    })
    assert "PASS" in clean and "7.6s" in clean and "异常" not in clean

    dirty = _summarize({
        "result": "FAIL", "total_duration_ms": 6400,
        "anomaly_types": ["MISSING_STEP", "TIMEOUT"],
    })
    assert "MISSING_STEP" in dirty and "TIMEOUT" in dirty


# ------------------------------------------------------- 界面状态快照


class StubPipeline:
    """build_snapshot 只用到这几个属性，不必真起模型。"""

    def __init__(self, template, station_id="ST09"):
        self.template = template
        self.config = PipelineConfig(
            sop_path="", yolo_weights="", action_weights="", station_id=station_id
        )
        self._assembly_index = 3


def make_result(state, *, action="Insert", confidence=0.94,
                target_part="refill", parts=(), timestamp_ms=5000):
    from sop.pipeline import FrameResult
    import numpy as np

    return FrameResult(
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        timestamp_ms=timestamp_ms,
        action=action,
        confidence=confidence,
        target_part=target_part,
        parts=tuple(parts),
        state=state,
        events=(),
    )


def advance(events):
    """把动作序列喂进状态机，返回最终状态。"""
    state = fsm.new_state()
    for timestamp, action, part in events:
        state, _ = fsm.step(TEMPLATE, state, fsm.Observation(
            action=action, confidence=0.95, timestamp_ms=timestamp, target_part=part,
        ))
    return state


def test_snapshot_fields_follow_design_doc():
    state = advance([(1000, "Pick", "barrel"), (2000, "Insert", "spring")])
    snapshot = build_snapshot(StubPipeline(TEMPLATE), make_result(state))

    assert snapshot["type"] == "station_update"
    assert snapshot["station_id"] == "ST09"
    assert snapshot["current_step"] == "S3"
    assert snapshot["current_step_name"] == "装笔芯"
    assert snapshot["expected_action"] == "Insert"
    assert snapshot["current_action"] == "Insert"
    assert snapshot["action_confidence"] == 0.94
    assert snapshot["assembly_index"] == 4
    assert snapshot["timeout_ms"] == 5000
    assert snapshot["elapsed_ms"] == 3000          # 5000 - 进入 S3 的 2000
    assert len(snapshot["steps"]) == 6


def test_snapshot_step_statuses():
    state = advance([(1000, "Pick", "barrel"), (2000, "Insert", "spring")])
    statuses = {
        step["id"]: step["status"]
        for step in build_snapshot(StubPipeline(TEMPLATE), make_result(state))["steps"]
    }
    assert statuses["S1"] == "done"
    assert statuses["S2"] == "done"
    assert statuses["S3"] == "active"
    assert statuses["S4"] == "pending"


def test_snapshot_marks_skipped_step():
    state = advance([
        (1000, "Pick", "barrel"),
        (2000, "Insert", "refill"),        # 跳过 S2
    ])
    statuses = {
        step["id"]: step["status"]
        for step in build_snapshot(StubPipeline(TEMPLATE), make_result(state))["steps"]
    }
    assert statuses["S2"] == "skipped"
    assert statuses["S3"] == "done"


def test_parts_status_values():
    """§8.5.2 规定的四种取值：in_hand / assembled / operating / waiting。"""
    state = advance([(1000, "Pick", "barrel"), (2000, "Insert", "spring")])
    snapshot = build_snapshot(
        StubPipeline(TEMPLATE), make_result(state, target_part="refill")
    )
    parts = snapshot["parts_status"]

    assert parts["barrel"] == "in_hand"      # S1 是 Pick，完成后在手上
    assert parts["spring"] == "assembled"
    assert parts["refill"] == "operating"    # 正被操作
    assert parts["tip"] == "waiting"
    assert "finished_pen" not in parts       # 成品不是零件


def test_snapshot_status_idle_when_action_idle():
    state = fsm.new_state()
    snapshot = build_snapshot(StubPipeline(TEMPLATE), make_result(state, action="Idle"))
    assert snapshot["status"] == "idle"


def test_snapshot_exposes_detected_parts_and_anomalies():
    state = advance([(1000, "Pick", "barrel"), (2000, "Insert", "refill")])
    boxes = [PartBox("refill", 0.91, (0.1, 0.1, 0.2, 0.2))]
    snapshot = build_snapshot(StubPipeline(TEMPLATE), make_result(state, parts=boxes))

    assert snapshot["detected_parts"] == [{"label": "refill", "confidence": 0.91}]
    assert [a["type"] for a in snapshot["anomalies"]] == ["MISSING_STEP"]


# ------------------------------------------------------------ LiveState


def test_live_state_event_sequencing():
    state = LiveState()
    assert state.latest_seq() == 0

    state.update(b"jpeg-1", {"type": "station_update"}, [{"type": "step_completed"}])
    seq, fresh = state.events_since(0)
    assert seq == 1 and len(fresh) == 1

    _, again = state.events_since(seq)
    assert again == []                       # 同一个事件不会重复推送

    state.update(b"jpeg-2", {}, [{"type": "anomaly_detected"}])
    _, more = state.events_since(seq)
    assert len(more) == 1
    assert state.jpeg() == b"jpeg-2"


def test_live_state_caps_event_backlog():
    state = LiveState()
    for index in range(300):
        state.update(b"x", {}, [{"type": "step_completed", "n": index}])
    _, everything = state.events_since(0)
    assert len(everything) <= 200            # MAX_EVENTS
