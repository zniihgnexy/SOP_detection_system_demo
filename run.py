#!/usr/bin/env python3
"""装笔顺序 SOP 智能检测系统 —— 命令行入口。

三种模式：

    python run.py --selfcheck
        环境自检。不需要模型权重、不需要数据，用来确认装机是否正确。

    python run.py --replay configs/demo_sequence.json
        回放手写动作序列，验证 SOP 判定层。只用标准库，即使 torch /
        mediapipe / opencv 都没装也能跑。全部用例符合预期时退出码为 0。

    python run.py --video demo.mp4          （或 --camera 0）
        真实检测。需要已训练好的 YOLO 与 Bi-LSTM 权重，见手册第 5、6 章。
        加 --web 则同时启动浏览器界面。

完整步骤见 docs/user-manual-zh.md。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sop import fsm                                    # noqa: E402  只依赖标准库
from sop.records import Recorder                       # noqa: E402


# ------------------------------------------------------------------ 小工具

def _s(ms: int) -> str:
    """毫秒转易读秒数。"""
    return f"{ms / 1000:.1f}s"


def _rule(title: str = "", width: int = 72) -> None:
    if title:
        print(f"\n{'═' * width}\n {title}\n{'═' * width}")
    else:
        print("─" * width)


def _parse_expectation(text: str) -> tuple[str, set[str]]:
    """把 "FAIL / MISSING_STEP + WRONG_ORDER" 解析成 ("FAIL", {两个类型})。"""
    head, _, tail = text.partition("/")
    result = head.strip().upper()
    types = {t.strip().upper() for t in tail.split("+") if t.strip()}
    return result, types


# --------------------------------------------------------------- 模式：回放

def cmd_replay(args: argparse.Namespace) -> int:
    spec = json.loads(Path(args.replay).read_text(encoding="utf-8"))

    template_path = args.sop if args.sop_given else spec.get("template", args.sop)
    template = fsm.load_template(ROOT / template_path
                                 if not Path(template_path).is_absolute()
                                 else template_path)
    station_id = spec.get("station_id", args.station)
    sequences = spec.get("sequences", [])
    if not sequences:
        print(f"错误：{args.replay} 里没有 sequences。", file=sys.stderr)
        return 2

    _rule(f"SOP 判定层回放 · 模板 {template.model_name} {template.version} "
          f"· {len(template.steps)} 个步骤")
    print(f" 记录数据库：{args.db}")

    outcomes: list[tuple[str, bool, str]] = []

    with Recorder(args.db) as recorder:
        template_id = recorder.upsert_template(template)

        for index, seq in enumerate(sequences, start=1):
            name = seq.get("name", f"序列 {index}")
            expect = seq.get("expect", "")
            print(f"\n【{index}/{len(sequences)}】{name}")
            if expect:
                print(f"  期望结果：{expect}")
            if seq.get("description"):
                print(f"  {seq['description']}")
            print()

            state = fsm.new_state()
            for raw in seq.get("events", []):
                obs = fsm.Observation(
                    action=raw["action"],
                    confidence=float(raw.get("confidence", 1.0)),
                    timestamp_ms=int(raw["t"]),
                    target_part=raw.get("target_part"),
                    parts_present=(tuple(raw["parts_present"])
                                   if "parts_present" in raw else None),
                )
                state, events = fsm.step(template, state, obs)
                for event in events:
                    _print_event(event, obs.timestamp_ms)

            # 序列跑完但没走到最后一步 → 补一次完整性检查
            if not state.finished:
                last_ms = int(seq["events"][-1]["t"]) if seq.get("events") else 0
                state, events = fsm.finalize(template, state, last_ms)
                for event in events:
                    _print_event(event, last_ms)

            result = "PASS" if state.passed else "FAIL"
            types = sorted({a.type for a in state.anomalies})
            duration = (state.step_entered_ms - state.started_ms)
            record_id = recorder.save_record(
                station_id=station_id,
                template_id=template_id,
                result=result,
                anomalies=state.anomalies,
                steps_completed=state.completed,
                duration_ms=duration,
            )

            ok, detail = True, ""
            if expect:
                want_result, want_types = _parse_expectation(expect)
                ok = result == want_result and (not want_types or set(types) == want_types)
                if not ok:
                    detail = (f"实际 {result}"
                              f"{' / ' + ' + '.join(types) if types else ''}")
            outcomes.append((name, ok, detail))

            mark = "✓" if ok else "✗"
            print(f"  {'─' * 66}")
            print(f"  {mark} 结果：{result}"
                  f"{'  异常：' + '、'.join(types) if types else ''}"
                  f"  总耗时 {_s(duration)}  已入库 #{record_id}")

        stats = recorder.summary(station_id)

    _rule("回放汇总")
    for name, ok, detail in outcomes:
        print(f"  {'✓ 符合预期' if ok else '✗ 不符预期'}  {name}"
              f"{'  →  ' + detail if detail else ''}")
    print(f"\n 数据库累计：{stats['total']} 条，合格 {stats['passed']} 条")
    print(f" 查看记录：sqlite3 {args.db} "
          f"\"SELECT id,result,anomaly_type FROM detection_records;\"")

    failed = [n for n, ok, _ in outcomes if not ok]
    if failed:
        print(f"\n 有 {len(failed)} 个用例不符合预期，判定引擎可能被改坏了。")
        return 1
    print("\n 全部用例符合预期。判定层工作正常。")
    return 0


def _print_event(event: dict, timestamp_ms: int) -> None:
    kind = event["type"]
    if kind == "step_completed":
        print(f"    ✓ {_s(timestamp_ms):>6}  {event['step_id']} {event['step_name']:<6}"
              f"  耗时 {event['duration_ms']:>5}ms  置信度 {event['confidence']}")
    elif kind == "anomaly_detected":
        print(f"    ✗ {_s(timestamp_ms):>6}  {event['anomaly_type']:<13} {event['message']}")
    elif kind == "assembly_complete":
        pass          # 汇总行统一打印


# ------------------------------------------------------------- 模式：自检

def cmd_selfcheck(args: argparse.Namespace) -> int:
    _rule("环境自检")
    problems: list[str] = []

    # 1) Python 版本
    major, minor = sys.version_info[:2]
    version_ok = (major, minor) in {(3, 10), (3, 11), (3, 12)}
    print(f"  {'✓' if version_ok else '✗'} Python {major}.{minor} "
          f"（要求 3.10 / 3.11 / 3.12）")
    if not version_ok:
        problems.append(
            f"Python {major}.{minor} 不在支持范围。MediaPipe 没有 3.13 / 3.14 的"
            f"安装包，请改用 3.10~3.12，见手册 1.2 节。"
        )

    # 2) 第三方依赖
    print("\n  依赖包：")
    deps = [
        ("numpy", "特征计算"),
        ("cv2", "视频读写（包名 opencv-python）"),
        ("torch", "Bi-LSTM 训练与推理"),
        ("mediapipe", "手部关键点"),
        ("ultralytics", "YOLOv8n 零件检测"),
        ("fastapi", "Web 界面"),
    ]
    missing: list[str] = []
    for module, purpose in deps:
        try:
            __import__(module)
            print(f"    ✓ {module:<12} {purpose}")
        except ImportError:
            print(f"    ✗ {module:<12} {purpose}  —— 未安装")
            missing.append(module)
    if missing:
        problems.append("缺少依赖：pip install -r requirements.txt")

    # 3) 配置文件
    print("\n  配置文件：")
    for rel in ("configs/gelpen_0.5.json", "configs/demo_sequence.json",
                "configs/pen_parts_dataset.yaml"):
        exists = (ROOT / rel).is_file()
        print(f"    {'✓' if exists else '✗'} {rel}")
        if not exists:
            problems.append(f"缺少配置文件 {rel}")

    # 4) 判定层实跑一遍
    print("\n  判定层：")
    try:
        template = fsm.load_template(ROOT / "configs/gelpen_0.5.json")
        state = fsm.new_state()
        for i, (action, part) in enumerate([
            ("Pick", "barrel"), ("Insert", "spring"), ("Insert", "refill"),
            ("Screw", "tip"), ("Press", "cap"), ("Place", "finished_pen"),
        ]):
            state, _ = fsm.step(template, state, fsm.Observation(
                action=action, confidence=0.95,
                timestamp_ms=(i + 1) * 1000, target_part=part))
        if state.passed:
            print(f"    ✓ SOP 模板 {template.model_name} 载入正常，"
                  f"{len(template.steps)} 个步骤，标准序列判定为 PASS")
        else:
            print("    ✗ 标准序列没有判定为 PASS")
            problems.append("判定层异常，请检查 sop/fsm.py 是否被改动")
    except Exception as exc:                      # noqa: BLE001  自检要报出全部原因
        print(f"    ✗ 判定层报错：{exc}")
        problems.append(f"判定层报错：{exc}")

    # 5) 模型权重（缺失是正常的，尚未训练）
    print("\n  模型权重：")
    for path, what, chapter in (
        (args.yolo, "零件检测模型", "第 5 章"),
        (args.action_model, "动作识别模型", "第 6 章"),
    ):
        full = ROOT / path if not Path(path).is_absolute() else Path(path)
        if full.is_file():
            print(f"    ✓ {path}  {what}")
        else:
            print(f"    · {path}  {what} —— 尚未训练，需按手册{chapter}准备")

    # 6) 摄像头（可选）
    print("\n  摄像头：")
    if "cv2" in missing:
        print("    · 跳过（opencv 未安装）")
    else:
        import cv2
        cap = cv2.VideoCapture(args.camera if args.camera is not None else 0)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok:
                h, w = frame.shape[:2]
                print(f"    ✓ 摄像头可用，分辨率 {w}×{h}")
            else:
                print("    · 摄像头打开了但读不到画面")
            cap.release()
        else:
            print("    · 没有可用摄像头（只用视频文件的话可以忽略）")

    _rule()
    if problems:
        print(f"\n 发现 {len(problems)} 个需要处理的问题：\n")
        for i, p in enumerate(problems, start=1):
            print(f"   {i}. {p}")
        print("\n 排错办法见 docs/user-manual-zh.md 第 11 章。")
        return 1

    print("\n 环境正常。下一步：python run.py --replay configs/demo_sequence.json")
    return 0


# ------------------------------------------------------------- 模式：检测

def cmd_detect(args: argparse.Namespace) -> int:
    """真实视频/摄像头检测。这里才会 import 重依赖。"""
    try:
        from sop.pipeline import DetectionPipeline, PipelineConfig
    except ImportError as exc:
        print(f"错误：缺少依赖 —— {exc}\n"
              f"先跑 pip install -r requirements.txt，"
              f"再用 python run.py --selfcheck 确认。", file=sys.stderr)
        return 2

    source: str | int = args.video if args.video else (args.camera or 0)
    config = PipelineConfig(
        sop_path=str(ROOT / args.sop if not Path(args.sop).is_absolute() else args.sop),
        yolo_weights=args.yolo,
        action_weights=args.action_model,
        station_id=args.station,
        db_path=args.db,
    )

    try:
        pipeline = DetectionPipeline(config)
    except FileNotFoundError as exc:
        print(f"\n错误：{exc}\n", file=sys.stderr)
        print("模型权重还没准备好。两个办法：\n"
              "  1. 按 docs/user-manual-zh.md 第 5、6 章采集数据并训练；\n"
              "  2. 想先确认判定逻辑是否正常，跑："
              "     python run.py --replay configs/demo_sequence.json",
              file=sys.stderr)
        return 2

    if args.web:
        from web.server import serve
        return serve(pipeline, source, host=args.host, port=args.port)

    return pipeline.run_headless(source, verbose=True)


# ---------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="装笔顺序 SOP 智能检测系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--selfcheck", action="store_true", help="环境自检，不需要权重")
    mode.add_argument("--replay", metavar="JSON", help="回放动作序列，验证判定层")
    mode.add_argument("--video", metavar="FILE", help="检测视频文件")
    mode.add_argument("--camera", metavar="N", nargs="?", const=0, type=int,
                      help="检测摄像头，默认 0 号")

    parser.add_argument("--sop", default="configs/gelpen_0.5.json", help="SOP 模板路径")
    parser.add_argument("--station", default="ST01", help="工位编号")
    parser.add_argument("--db", default="sop.db", help="记录数据库路径")
    parser.add_argument("--yolo", default="models/pen_parts.pt", help="零件检测权重")
    parser.add_argument("--action-model", default="models/action_model.pt",
                        dest="action_model", help="动作识别权重")
    parser.add_argument("--web", action="store_true", help="同时启动浏览器界面")
    parser.add_argument("--host", default="127.0.0.1", help="界面监听地址")
    parser.add_argument("--port", type=int, default=8000, help="界面端口")

    args = parser.parse_args(argv)
    # 回放时优先用序列文件里指定的模板，除非用户显式传了 --sop
    args.sop_given = any(a.startswith("--sop") for a in (argv or sys.argv[1:]))

    if args.selfcheck:
        return cmd_selfcheck(args)
    if args.replay:
        return cmd_replay(args)
    return cmd_detect(args)


if __name__ == "__main__":
    raise SystemExit(main())
