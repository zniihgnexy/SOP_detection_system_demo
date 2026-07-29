"""SOP 判定引擎测试（sop/fsm.py）。

这些测试不需要模型权重、不需要数据、不需要第三方包（pytest 除外），
因为判定层只依赖标准库。改动 fsm.py 后必须全绿。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sop import fsm

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = fsm.load_template(ROOT / "configs" / "gelpen_0.5.json")

#: 完全符合 SOP 的一次装配
GOOD = [
    (1000, "Pick", "barrel"),
    (2000, "Insert", "spring"),
    (3000, "Insert", "refill"),
    (4000, "Screw", "tip"),
    (5000, "Press", "cap"),
    (6000, "Place", "finished_pen"),
]


def feed(template, events, state=None):
    """把 (t, action, part[, conf[, parts_present]]) 序列喂给状态机。"""
    state = state or fsm.new_state()
    collected = []
    for event in events:
        obs = fsm.Observation(
            action=event[1],
            confidence=event[3] if len(event) > 3 else 0.95,
            timestamp_ms=event[0],
            target_part=event[2],
            parts_present=event[4] if len(event) > 4 else None,
        )
        state, produced = fsm.step(template, state, obs)
        collected += produced
    return state, collected


def kinds(state) -> list[str]:
    return sorted({a.type for a in state.anomalies})


# --------------------------------------------------------------- 正常路径


def test_normal_sequence_passes():
    state, events = feed(TEMPLATE, GOOD)
    assert state.finished
    assert state.passed
    assert kinds(state) == []
    assert state.completed == ("S1", "S2", "S3", "S4", "S5", "S6")
    assert [e["type"] for e in events].count("step_completed") == 6
    assert events[-1]["type"] == "assembly_complete"
    assert events[-1]["result"] == "PASS"


def test_step_durations_are_not_zero():
    _, events = feed(TEMPLATE, GOOD)
    durations = [e["duration_ms"] for e in events if e["type"] == "step_completed"]
    assert all(d == 1000 for d in durations), durations


def test_transient_actions_neither_advance_nor_alarm():
    state, _ = feed(TEMPLATE, [
        (1000, "Pick", "barrel"),
        (1500, "Pick", "spring"),      # 拿零件，不是装配步骤
        (1800, "Align", "spring"),     # 对位，不是装配步骤
        (2000, "Idle", None),
    ])
    assert state.step_index == 1
    assert kinds(state) == []


def test_leading_idle_does_not_time_out_first_step():
    """S1 限时 3000ms，但开工前的等待不该计时（§4.3.1 的 S0_IDLE）。"""
    state, _ = feed(TEMPLATE, [
        (0, "Idle", None),
        (5000, "Idle", None),
        (5500, "Pick", "barrel"),
    ])
    assert kinds(state) == []
    assert state.completed == ("S1",)


def test_low_confidence_is_discarded():
    state, events = feed(TEMPLATE, [(1000, "Pick", "barrel", 0.3)])
    assert state.step_index == 0
    assert state.completed == ()
    assert events == []


# ----------------------------------------------------------------- 异常


def test_missing_step():
    state, _ = feed(TEMPLATE, [
        (1000, "Pick", "barrel"),
        (2000, "Insert", "refill"),        # 跳过 S2 装弹簧
        (3000, "Screw", "tip"),
        (4000, "Press", "cap"),
        (5000, "Place", "finished_pen"),
    ])
    assert kinds(state) == ["MISSING_STEP"]
    assert not state.passed
    assert [a.step_id for a in state.anomalies] == ["S2"]


def test_wrong_order_when_skipped_step_done_late():
    """先装笔尖后装笔芯：跳步瞬间判漏装，补做时再判错序。"""
    state, _ = feed(TEMPLATE, [
        (1000, "Pick", "barrel"),
        (2000, "Insert", "spring"),
        (3000, "Screw", "tip"),            # S4 提前
        (4000, "Insert", "refill"),        # S3 补做
        (5000, "Press", "cap"),
        (6000, "Place", "finished_pen"),
    ])
    assert kinds(state) == ["MISSING_STEP", "WRONG_ORDER"]
    assert state.finished and not state.passed


def test_extra_step_on_repeat():
    state, _ = feed(TEMPLATE, [
        (1000, "Pick", "barrel"),
        (2000, "Insert", "spring"),
        (2500, "Insert", "spring"),        # 重复已完成的 S2
    ])
    assert "EXTRA_STEP" in kinds(state)


def test_wrong_part_when_action_exists_with_other_part():
    state, _ = feed(TEMPLATE, [
        (1000, "Pick", "barrel"),
        (2000, "Insert", "grip"),          # SOP 里有 Insert，但没有 Insert(grip)
    ])
    assert kinds(state) == ["WRONG_PART"]


def test_wrong_order_when_action_not_in_sop():
    state, _ = feed(TEMPLATE, [
        (1000, "Pick", "barrel"),
        (2000, "Screw", "spring"),         # SOP 里没有 Screw(spring)，但有 Screw
    ])
    # Screw 存在于 SOP（S4），只是零件不对 → 归为零件错误
    assert kinds(state) == ["WRONG_PART"]


def test_required_parts_missing_triggers_wrong_part():
    absent_spring = ("barrel", "refill", "tip", "cap")
    state, _ = feed(TEMPLATE, [
        (1000, "Pick", "barrel", 0.95, absent_spring),
        (2000, "Insert", "spring", 0.95, absent_spring),
    ])
    assert "WRONG_PART" in kinds(state)


def test_parts_check_skipped_when_detection_absent():
    """parts_present 为 None（没有 YOLO 权重）时不该误报零件缺失。"""
    state, _ = feed(TEMPLATE, GOOD)
    assert "WRONG_PART" not in kinds(state)


def test_timeout():
    state, _ = feed(TEMPLATE, [
        (1000, "Pick", "barrel"),
        (9000, "Insert", "spring"),        # S2 限时 5000ms
    ])
    assert "TIMEOUT" in kinds(state)
    timeout = next(a for a in state.anomalies if a.type == "TIMEOUT")
    assert timeout.step_id == "S2"


def test_same_anomaly_reported_only_once():
    state, _ = feed(TEMPLATE, [
        (1000, "Pick", "barrel"),
        (9000, "Idle", None),
        (10000, "Idle", None),
        (11000, "Idle", None),
    ])
    assert len([a for a in state.anomalies if a.type == "TIMEOUT"]) == 1


def test_finalize_reports_all_unfinished_steps():
    state, _ = feed(TEMPLATE, [(1000, "Pick", "barrel"), (2000, "Insert", "spring")])
    assert not state.finished

    state, events = fsm.finalize(TEMPLATE, state, 3000)
    assert state.finished
    missing = {a.step_id for a in state.anomalies if a.type == "MISSING_STEP"}
    assert missing == {"S3", "S4", "S5", "S6"}
    assert events[-1]["type"] == "assembly_complete"
    assert events[-1]["result"] == "FAIL"


def test_no_events_after_finished():
    state, _ = feed(TEMPLATE, GOOD)
    state2, events = fsm.step(
        TEMPLATE, state,
        fsm.Observation("Insert", 0.99, 9000, "spring"),
    )
    assert state2 is state
    assert events == []


# ------------------------------------------------------------- optional


def _template_with_optional() -> fsm.SOPTemplate:
    return fsm.SOPTemplate(
        model_name="T", version="v1",
        steps=(
            fsm.SOPStep("A", "甲", "Pick", "barrel", timeout_ms=10**6),
            fsm.SOPStep("B", "乙", "Insert", "grip", timeout_ms=10**6, optional=True),
            fsm.SOPStep("C", "丙", "Insert", "spring", timeout_ms=10**6),
        ),
    )


def test_optional_step_can_be_skipped_silently():
    template = _template_with_optional()
    state, _ = feed(template, [(1000, "Pick", "barrel"), (2000, "Insert", "spring")])
    assert state.finished
    assert state.passed, [a.message for a in state.anomalies]
    assert state.completed == ("A", "C")


# ------------------------------------------------------------ 不可变性


def test_step_does_not_mutate_input_state():
    state = fsm.new_state()
    new, _ = fsm.step(TEMPLATE, state, fsm.Observation("Pick", 0.95, 1000, "barrel"))
    assert state.step_index == 0
    assert state.completed == ()
    assert new is not state
    assert new.step_index == 1


def test_dataclasses_are_frozen():
    state = fsm.new_state()
    with pytest.raises(Exception):
        state.step_index = 5        # type: ignore[misc]


# --------------------------------------------------------- 模板载入校验


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "sop.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _step(**over) -> dict:
    base = {
        "id": "S1", "name": "取笔杆", "expected_action": "Pick",
        "target_part": "barrel", "timeout_ms": 3000,
    }
    base.update(over)
    return base


def test_load_template_reads_all_fields():
    assert TEMPLATE.model_name == "GelPen_0.5mm"
    assert len(TEMPLATE.steps) == 6
    assert TEMPLATE.steps[1].target_part == "spring"
    assert TEMPLATE.steps[1].parent_part == "barrel"
    assert TEMPLATE.initial_parts == ("barrel", "spring", "refill", "tip", "cap")


def test_load_template_rejects_unknown_action(tmp_path):
    path = _write(tmp_path, {"model_name": "T", "steps": [_step(expected_action="Twist")]})
    with pytest.raises(ValueError, match="Twist"):
        fsm.load_template(path)


def test_load_template_rejects_duplicate_ids(tmp_path):
    path = _write(tmp_path, {"model_name": "T", "steps": [_step(), _step()]})
    with pytest.raises(ValueError, match="重复"):
        fsm.load_template(path)


def test_load_template_rejects_empty_steps(tmp_path):
    path = _write(tmp_path, {"model_name": "T", "steps": []})
    with pytest.raises(ValueError, match="不能为空"):
        fsm.load_template(path)


# ------------------------------------------------------------- 查询方法


def test_template_query_helpers():
    assert TEMPLATE.expected_action(0) == "Pick"
    assert TEMPLATE.required_parts(1) == ("spring", "barrel")
    assert TEMPLATE.timeout(1) == 5000
    assert TEMPLATE.step_at(99) is None
    assert TEMPLATE.expected_action(99) is None
    assert TEMPLATE.find_step("Insert", "refill") == 2
    assert TEMPLATE.find_step("Insert", "nonexistent") is None
    assert TEMPLATE.find_step("Insert") == 1          # 无零件信息时取第一个 Insert
    assert TEMPLATE.has_action("Screw") is True
    assert TEMPLATE.has_action("Align") is False
    assert TEMPLATE.valid_skips(0) == ()              # 本模板没有可选步骤
    assert TEMPLATE.all_steps_completed(
        ("S1", "S2", "S3", "S4", "S5", "S6")) is True
    assert [s.id for s in TEMPLATE.missing_steps(("S1",))] == \
        ["S2", "S3", "S4", "S5", "S6"]


def test_constants_match_design_doc():
    assert len(fsm.ACTIONS) == 7
    assert fsm.ACTIONS[0] == "Pick" and fsm.ACTIONS[-1] == "Idle"
    assert len(fsm.PARTS) == 6
    assert fsm.MIN_CONFIDENCE == 0.6
