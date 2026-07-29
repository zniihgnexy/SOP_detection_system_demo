"""SOP 时序匹配引擎（设计文档 §4.3）。

设计要点
--------
* **纯函数式**：``step()`` 接收当前状态和一次观测，返回**新**状态与事件列表，
  从不修改入参。所有数据类都是 frozen，状态更新一律走 ``dataclasses.replace``。
* **只依赖标准库**。因此 ``run.py --replay`` 在没装 torch / mediapipe / opencv、
  没有任何模型权重和训练数据的情况下，也能完整验证判定逻辑。

对设计文档未定项的实现选择
--------------------------
1. **步骤粒度 = 一个装配动作**，共 6 步（S1..S6）。依据是 §4.3.4 的
   ``sop_step_details`` 示例（step 1 = Pick barrel、step 2 = Insert spring）、
   §8.3.2 的六格进度条、§8.3.3 编辑器节点图 —— 三处一致。§4.3.1 状态图里
   ``Pick(spring) → Insert(spring, barrel)`` 这类复合边因此被拆开看待：
   中间那个 Pick 属于过渡动作，不推进状态机。
   （§4.3.1 的 ``S2_SPRING`` 这套「后置条件」命名不再单独维护，
   状态由步骤下标推导。）
2. **过渡动作**：Pick / Align / Idle 在不匹配当前期望动作时**不算异常** ——
   拿起零件、微调对位本身不构成装配步骤。Insert / Screw / Press / Place
   属于提交动作，不匹配即判异常。
3. **在线判定的固有特性**：跳步的瞬间只能判为漏装（MISSING_STEP）；
   若被跳过的步骤稍后补做，会再报一次错序（WRONG_ORDER）。同一根因两条异常
   是正确行为，不是重复报警。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

# --------------------------------------------------------------------- 常量

#: 7 类装配动作，顺序即模型输出通道顺序（§4.2.1）
ACTIONS: tuple[str, ...] = (
    "Pick", "Align", "Insert", "Screw", "Press", "Place", "Idle",
)

#: 6 类零件，顺序即 YOLO 类别序号（§7 Phase 1）。
#: 改动这里必须同步 configs/pen_parts_dataset.yaml 的 names。
PARTS: tuple[str, ...] = ("barrel", "cap", "spring", "refill", "tip", "grip")

#: 不匹配当前步骤时也不判异常的过渡动作
TRANSIENT_ACTIONS = frozenset({"Pick", "Align", "Idle"})

#: 动作有效性下限（§4.3.2 Step A）：低于此值视为噪声直接丢弃
MIN_CONFIDENCE = 0.6


class AnomalyType:
    """6 类异常（§4.4）。"""

    MISSING_STEP = "MISSING_STEP"   # 漏装
    WRONG_ORDER = "WRONG_ORDER"     # 错序
    EXTRA_STEP = "EXTRA_STEP"       # 多装
    WRONG_PART = "WRONG_PART"       # 零件错误
    TIMEOUT = "TIMEOUT"             # 装配超时
    DROPPED = "DROPPED"             # 零件掉落


#: 告警级别映射（§8.3.6）。零件错误直接算严重，其余为一般。
_SEVERITY = {
    AnomalyType.WRONG_PART: "critical",
    AnomalyType.DROPPED: "critical",
}


# ----------------------------------------------------------------- 数据模型


@dataclass(frozen=True)
class SOPStep:
    """一个装配步骤（§4.3.1）。"""

    id: str
    name: str
    expected_action: str
    target_part: str
    parent_part: str | None = None
    timeout_ms: int = 5000
    required_parts_present: tuple[str, ...] = ()
    optional: bool = False


@dataclass(frozen=True)
class SOPTemplate:
    """一个笔型的标准装配顺序（§4.3.1）。"""

    model_name: str
    version: str
    steps: tuple[SOPStep, ...]
    initial_parts: tuple[str, ...] = ()
    final_parts: tuple[str, ...] = ()

    # --- §4.3.2 / §5.3 伪代码所调用的查询方法 ---

    def step_at(self, index: int) -> SOPStep | None:
        """下标越界返回 None（装配完成后 step_index 会等于步骤总数）。"""
        if 0 <= index < len(self.steps):
            return self.steps[index]
        return None

    def expected_action(self, index: int) -> str | None:
        step_ = self.step_at(index)
        return step_.expected_action if step_ else None

    def required_parts(self, index: int) -> tuple[str, ...]:
        step_ = self.step_at(index)
        return step_.required_parts_present if step_ else ()

    def timeout(self, index: int) -> int | None:
        step_ = self.step_at(index)
        return step_.timeout_ms if step_ else None

    def valid_skips(self, index: int) -> tuple[int, ...]:
        """从 index 起可以合法跳过的步骤下标，即所有 optional 步骤。"""
        return tuple(i for i in range(index, len(self.steps)) if self.steps[i].optional)

    def find_step(self, action: str, target_part: str | None = None) -> int | None:
        """按（动作, 目标零件）定位步骤下标，找不到返回 None。

        S2 和 S3 的 expected_action 都是 Insert，靠 target_part 区分。
        动作模型只给出动作类别时 target_part 为 None，退化为按动作匹配第一个。
        """
        for i, step_ in enumerate(self.steps):
            if step_.expected_action == action and step_.target_part == target_part:
                return i
        if target_part is None:
            for i, step_ in enumerate(self.steps):
                if step_.expected_action == action:
                    return i
        return None

    def has_action(self, action: str) -> bool:
        """SOP 里是否存在这个动作（用于区分「零件用错」和「动作不该出现」）。"""
        return any(step_.expected_action == action for step_ in self.steps)

    def all_steps_completed(self, completed: tuple[str, ...]) -> bool:
        return not self.missing_steps(completed)

    def missing_steps(self, completed: tuple[str, ...]) -> tuple[SOPStep, ...]:
        return tuple(
            step_ for step_ in self.steps
            if not step_.optional and step_.id not in completed
        )


@dataclass(frozen=True)
class Observation:
    """一次动作预测的观测结果。"""

    action: str
    confidence: float
    timestamp_ms: int
    target_part: str | None = None
    #: 当前帧检测到的零件。None 表示未接入零件检测（回放模式，或没有 YOLO 权重），
    #: 此时跳过零件校验而不是误报 WRONG_PART。
    parts_present: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Anomaly:
    type: str
    message: str
    step_id: str | None
    timestamp_ms: int


@dataclass(frozen=True)
class MatchState:
    """状态机当前状态。step_index 指向**下一个期望完成**的步骤。"""

    step_index: int = 0
    completed: tuple[str, ...] = ()
    step_entered_ms: int = 0
    started_ms: int = 0
    anomalies: tuple[Anomaly, ...] = ()
    finished: bool = False

    @property
    def passed(self) -> bool:
        return self.finished and not self.anomalies


# ------------------------------------------------------------------- 载入


def load_template(path: str | Path) -> SOPTemplate:
    """从 JSON 文件读取 SOP 模板。字段说明见手册第 7 章。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    steps = tuple(
        SOPStep(
            id=s["id"],
            name=s["name"],
            expected_action=s["expected_action"],
            target_part=s["target_part"],
            parent_part=s.get("parent_part"),
            timeout_ms=int(s.get("timeout_ms", 5000)),
            required_parts_present=tuple(s.get("required_parts_present", ())),
            optional=bool(s.get("optional", False)),
        )
        for s in raw["steps"]
    )

    if not steps:
        raise ValueError(f"{path}: steps 不能为空")

    for step_ in steps:
        if step_.expected_action not in ACTIONS:
            raise ValueError(
                f"{path}: 步骤 {step_.id} 的 expected_action='{step_.expected_action}' "
                f"不是 7 类动作之一，可选值：{'、'.join(ACTIONS)}"
            )

    ids = [s.id for s in steps]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: 步骤 id 有重复：{ids}")

    return SOPTemplate(
        model_name=raw["model_name"],
        version=raw.get("version", "v1.0"),
        steps=steps,
        initial_parts=tuple(raw.get("initial_parts", ())),
        final_parts=tuple(raw.get("final_parts", ())),
    )


def new_state(started_ms: int = 0) -> MatchState:
    """新建一次装配的初始状态。"""
    return MatchState(step_entered_ms=started_ms, started_ms=started_ms)


# ------------------------------------------------------------------- 引擎


def step(
    template: SOPTemplate, state: MatchState, obs: Observation
) -> tuple[MatchState, list[dict]]:
    """推进一次状态机，返回 (新状态, 事件列表)。

    事件的 type 字段与 §8.5.2 的 WebSocket 事件名一致，可直接推给前端。
    检查顺序照搬 §4.3.2：A 有效性过滤 → D 超时 → C 零件校验 → B SOP 匹配。
    """
    if state.finished:
        return state, []

    # Step A：置信度过滤 —— 低于阈值视为噪声
    if obs.confidence < MIN_CONFIDENCE:
        return state, []

    # 装配开始前的空闲不计入第一步限时（§4.3.1 的 S0_IDLE 是等待零件就位的状态）
    if obs.action == "Idle" and state.step_index == 0 and not state.completed:
        return replace(
            state, step_entered_ms=obs.timestamp_ms, started_ms=obs.timestamp_ms
        ), []

    events: list[dict] = []

    # Step D：超时检查
    state, ev = _check_timeout(template, state, obs)
    events += ev

    # Step C：零件校验
    state, ev = _check_parts(template, state, obs)
    events += ev

    if obs.action == "Idle":
        return state, events

    # Step B：SOP 匹配
    state, ev = _match_action(template, state, obs)
    return state, events + ev


def format_event(event: dict, timestamp_ms: int | None = None) -> str | None:
    """把事件格式化成一行终端输出。返回 None 表示这个事件不单独打印。

    放在 fsm 里是因为回放（run.py，只用标准库）和实时检测（pipeline.py，需要
    torch/cv2）都要打印同样的东西，抽到这里可以避免两份格式代码走样。
    """
    stamp = "" if timestamp_ms is None else f"{timestamp_ms / 1000:>6.1f}s  "
    kind = event["type"]

    if kind == "step_completed":
        return (f"    ✓ {stamp}{event['step_id']} {event['step_name']:<6}"
                f"  耗时 {event['duration_ms']:>5}ms  置信度 {event['confidence']}")
    if kind == "anomaly_detected":
        return f"    ✗ {stamp}{event['anomaly_type']:<13} {event['message']}"
    return None


def finalize(
    template: SOPTemplate, state: MatchState, timestamp_ms: int
) -> tuple[MatchState, list[dict]]:
    """装配结束时的完整性检查（§5.3 第 6 步）并产出 assembly_complete 事件。

    正常流程下由最后一个步骤自动触发；视频提前结束时由 pipeline 主动调用。
    """
    if state.finished:
        return state, []

    events: list[dict] = []
    for missing in template.missing_steps(state.completed):
        if _already_reported(state, AnomalyType.MISSING_STEP, missing.id):
            continue
        state, ev = _add_anomaly(state, Anomaly(
            AnomalyType.MISSING_STEP,
            f"漏装：步骤 {missing.id} {missing.name} 始终未执行",
            missing.id, timestamp_ms,
        ))
        events += ev

    state = replace(state, finished=True)
    events.append({
        "type": "assembly_complete",
        "result": "PASS" if not state.anomalies else "FAIL",
        "total_duration_ms": timestamp_ms - state.started_ms,
        "steps_completed": list(state.completed),
        "anomaly_types": sorted({a.type for a in state.anomalies}),
    })
    return state, events


# --------------------------------------------------------------- 内部实现


def _already_reported(state: MatchState, kind: str, step_id: str | None) -> bool:
    """同一步骤的同类异常只报一次，避免流式输入下刷屏。"""
    return any(a.type == kind and a.step_id == step_id for a in state.anomalies)


def _add_anomaly(state: MatchState, anomaly: Anomaly) -> tuple[MatchState, list[dict]]:
    new = replace(state, anomalies=state.anomalies + (anomaly,))
    return new, [{
        "type": "anomaly_detected",
        "anomaly_type": anomaly.type,
        "message": anomaly.message,
        "step_id": anomaly.step_id,
        "timestamp_ms": anomaly.timestamp_ms,
        "severity": _SEVERITY.get(anomaly.type, "warning"),
    }]


def _check_timeout(
    template: SOPTemplate, state: MatchState, obs: Observation
) -> tuple[MatchState, list[dict]]:
    current = template.step_at(state.step_index)
    if current is None:
        return state, []

    elapsed = obs.timestamp_ms - state.step_entered_ms
    if elapsed <= current.timeout_ms:
        return state, []
    if _already_reported(state, AnomalyType.TIMEOUT, current.id):
        return state, []

    return _add_anomaly(state, Anomaly(
        AnomalyType.TIMEOUT,
        f"超时：步骤 {current.id} {current.name} 已耗时 {elapsed}ms，"
        f"限时 {current.timeout_ms}ms",
        current.id, obs.timestamp_ms,
    ))


def _check_parts(
    template: SOPTemplate, state: MatchState, obs: Observation
) -> tuple[MatchState, list[dict]]:
    if obs.parts_present is None:      # 未接入零件检测，跳过而不是误报
        return state, []

    current = template.step_at(state.step_index)
    if current is None:
        return state, []

    missing = tuple(
        p for p in current.required_parts_present if p not in obs.parts_present
    )
    if not missing or _already_reported(state, AnomalyType.WRONG_PART, current.id):
        return state, []

    return _add_anomaly(state, Anomaly(
        AnomalyType.WRONG_PART,
        f"零件缺失：步骤 {current.id} {current.name} 需要 "
        f"{'、'.join(missing)}，当前画面中未检测到",
        current.id, obs.timestamp_ms,
    ))


def _matches(step_: SOPStep, obs: Observation) -> bool:
    if obs.action != step_.expected_action:
        return False
    return obs.target_part is None or obs.target_part == step_.target_part


def _describe(obs: Observation) -> str:
    return f"{obs.action}({obs.target_part})" if obs.target_part else obs.action


def _match_action(
    template: SOPTemplate, state: MatchState, obs: Observation
) -> tuple[MatchState, list[dict]]:
    current = template.step_at(state.step_index)
    if current is None:
        return state, []

    # 命中当前期望步骤 → 推进
    if _matches(current, obs):
        return _complete_step(template, state, obs, state.step_index)

    # 过渡动作不匹配也不算异常（拿零件、对位）
    if obs.action in TRANSIENT_ACTIONS:
        return state, []

    hit = template.find_step(obs.action, obs.target_part)

    if hit is None:
        # 动作本身在 SOP 里存在，只是零件对不上 → 用错了零件
        if template.has_action(obs.action):
            return _add_anomaly(state, Anomaly(
                AnomalyType.WRONG_PART,
                f"零件错误：检测到 {_describe(obs)}，"
                f"但 {current.id} {current.name} 期望的是 {current.target_part}",
                current.id, obs.timestamp_ms,
            ))
        return _add_anomaly(state, Anomaly(
            AnomalyType.WRONG_ORDER,
            f"多余动作：检测到 {_describe(obs)}，SOP 中没有这个装配动作"
            f"（当前期望 {current.id} {current.name}）",
            current.id, obs.timestamp_ms,
        ))

    # 命中后续步骤 → 中间的必做步骤被跨过
    if hit > state.step_index:
        state, events = _report_skipped(template, state, obs, upto=hit)
        state, ev = _complete_step(template, state, obs, hit)
        return state, events + ev

    # 命中更早的步骤
    earlier = template.steps[hit]
    if earlier.id in state.completed:
        return _add_anomaly(state, Anomaly(
            AnomalyType.EXTRA_STEP,
            f"多装：重复执行已完成的步骤 {earlier.id} {earlier.name}",
            earlier.id, obs.timestamp_ms,
        ))

    # 先前被跳过、现在补做 → 顺序错误
    state, events = _add_anomaly(state, Anomaly(
        AnomalyType.WRONG_ORDER,
        f"错序：步骤 {earlier.id} {earlier.name} 在 "
        f"{template.steps[state.step_index - 1].name} 之后才补做",
        earlier.id, obs.timestamp_ms,
    ))
    return replace(state, completed=state.completed + (earlier.id,)), events


def _report_skipped(
    template: SOPTemplate, state: MatchState, obs: Observation, upto: int
) -> tuple[MatchState, list[dict]]:
    """报告 [state.step_index, upto) 之间被跨过的必做步骤。"""
    skippable = set(template.valid_skips(state.step_index))
    events: list[dict] = []

    for i in range(state.step_index, upto):
        skipped = template.steps[i]
        if i in skippable or skipped.id in state.completed:
            continue
        if _already_reported(state, AnomalyType.MISSING_STEP, skipped.id):
            continue
        state, ev = _add_anomaly(state, Anomaly(
            AnomalyType.MISSING_STEP,
            f"漏装：步骤 {skipped.id} {skipped.name} 被跳过",
            skipped.id, obs.timestamp_ms,
        ))
        events += ev

    return state, events


def _complete_step(
    template: SOPTemplate, state: MatchState, obs: Observation, index: int
) -> tuple[MatchState, list[dict]]:
    done = template.steps[index]
    completed = (
        state.completed if done.id in state.completed
        else state.completed + (done.id,)
    )
    next_index = index + 1

    # 本步耗时 = 从上一步完成（进入本步）到现在，必须在覆盖 step_entered_ms 之前算
    duration_ms = obs.timestamp_ms - state.step_entered_ms

    state = replace(
        state,
        step_index=next_index,
        completed=completed,
        step_entered_ms=obs.timestamp_ms,
    )
    events = [{
        "type": "step_completed",
        "step_id": done.id,
        "step_name": done.name,
        "duration_ms": duration_ms,
        "confidence": round(obs.confidence, 3),
    }]

    if next_index >= len(template.steps):
        state, ev = finalize(template, state, obs.timestamp_ms)
        events += ev

    return state, events
