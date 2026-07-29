"""单页监控界面服务（设计文档 §8.5）。

实现的接口：

    GET  /                                    单页界面
    GET  /api/v1/stations                     工位列表与状态
    GET  /api/v1/stations/{id}/video          MJPEG 视频流（带检测叠加层）
    GET  /api/v1/records?limit=               检测记录
    GET  /api/v1/stats/overview               合格率汇总
    WS   /ws/live                             实时推送

推送事件沿用 §8.5.2 的 type 命名：``station_update`` / ``step_completed``
/ ``anomaly_detected`` / ``assembly_complete``。

采集与推理跑在一个后台线程里，Web 端只读最近一帧的结果 —— 这样浏览器卡了
或者没人看，检测也不会停。
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

from sop.pipeline import DetectionPipeline, FrameResult, open_source

HERE = Path(__file__).resolve().parent
MAX_EVENTS = 200


class LiveState:
    """后台线程写、Web 请求读的共享槽位。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._snapshot: dict = {}
        self._events: list[tuple[int, dict]] = []
        self._seq = 0
        self.running = True
        self.error: str | None = None

    def update(self, jpeg: bytes, snapshot: dict, events) -> None:
        with self._lock:
            self._jpeg = jpeg
            self._snapshot = snapshot
            for event in events:
                self._seq += 1
                self._events.append((self._seq, event))
            if len(self._events) > MAX_EVENTS:
                self._events = self._events[-MAX_EVENTS:]

    def jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._snapshot)

    def latest_seq(self) -> int:
        with self._lock:
            return self._seq

    def events_since(self, seq: int) -> tuple[int, list[dict]]:
        with self._lock:
            fresh = [event for number, event in self._events if number > seq]
            return self._seq, fresh


# --------------------------------------------------------------- 状态快照


#: §8.5.2 的 parts_status 取值
STATUS_WAITING = "waiting"
STATUS_OPERATING = "operating"
STATUS_ASSEMBLED = "assembled"
STATUS_IN_HAND = "in_hand"


def build_snapshot(pipeline: DetectionPipeline, result: FrameResult) -> dict:
    """组装 station_update 负载（§8.5.2）。"""
    template = pipeline.template
    state = result.state
    current = template.step_at(state.step_index)

    steps = []
    parts_status: dict[str, str] = {}
    for index, step_ in enumerate(template.steps):
        if step_.id in state.completed:
            status = "done"
        elif index == state.step_index:
            status = "active"
        elif index < state.step_index:
            status = "skipped"
        else:
            status = "pending"
        steps.append({
            "id": step_.id, "name": step_.name,
            "action": step_.expected_action, "target_part": step_.target_part,
            "status": status,
        })

        if step_.target_part in (None, "finished_pen"):
            continue
        if step_.target_part == result.target_part:
            parts_status[step_.target_part] = STATUS_OPERATING
        elif step_.id in state.completed:
            # 第一步是「取笔杆」，完成后笔杆是在手上而不是装好了
            parts_status[step_.target_part] = (
                STATUS_IN_HAND if step_.expected_action == "Pick" else STATUS_ASSEMBLED
            )
        else:
            parts_status[step_.target_part] = STATUS_WAITING

    return {
        "type": "station_update",
        "station_id": pipeline.config.station_id,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "idle" if result.action == "Idle" else "running",
        "model_name": template.model_name,
        "model_version": template.version,
        "current_step": current.id if current else None,
        "current_step_name": current.name if current else "已完成",
        "expected_action": current.expected_action if current else None,
        "current_action": result.action,
        "action_confidence": round(result.confidence, 3),
        "target_part": result.target_part,
        "elapsed_ms": max(0, result.timestamp_ms - state.step_entered_ms),
        "timeout_ms": current.timeout_ms if current else None,
        "assembly_index": pipeline._assembly_index + 1,
        "steps": steps,
        "parts_status": parts_status,
        "detected_parts": [
            {"label": p.label, "confidence": round(p.confidence, 3)} for p in result.parts
        ],
        "anomalies": [
            {"type": a.type, "message": a.message, "step_id": a.step_id}
            for a in state.anomalies
        ],
    }


# ------------------------------------------------------------- 采集线程


def capture_loop(
    pipeline: DetectionPipeline, source: str | int, state: LiveState, loop_video: bool
) -> None:
    try:
        capture, timestamp_of = open_source(source)
        if not capture.isOpened():
            state.error = f"打不开视频源 {source!r}"
            return

        frame_index = 0
        while state.running:
            ok, frame = capture.read()
            if not ok:
                if loop_video:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_index = 0
                    continue
                break

            result = pipeline.process_frame(frame, timestamp_of(capture, frame_index))
            frame_index += 1

            ok, buffer = cv2.imencode(
                ".jpg", result.frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            )
            if ok:
                state.update(
                    buffer.tobytes(), build_snapshot(pipeline, result), result.events
                )
        capture.release()
    except Exception as exc:                        # noqa: BLE001  线程里必须兜住
        state.error = f"{type(exc).__name__}: {exc}"
    finally:
        state.running = False


# ---------------------------------------------------------------- 应用


def create_app(pipeline: DetectionPipeline, state: LiveState):
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

    app = FastAPI(title="装笔顺序 SOP 智能检测系统", docs_url="/api/docs")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (HERE / "index.html").read_text(encoding="utf-8")

    @app.get("/api/v1/stations")
    def stations() -> JSONResponse:
        snapshot = state.snapshot()
        return JSONResponse([snapshot] if snapshot else [])

    @app.get("/api/v1/stations/{station_id}/video")
    def video(station_id: str) -> StreamingResponse:
        boundary = "frame"

        def frames():
            while state.running:
                jpeg = state.jpeg()
                if jpeg is None:
                    time.sleep(0.05)
                    continue
                yield (
                    b"--" + boundary.encode() + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                    + jpeg + b"\r\n"
                )
                time.sleep(1 / 15)          # 界面 15fps 够用，检测仍是全帧率

        return StreamingResponse(
            frames(), media_type=f"multipart/x-mixed-replace; boundary={boundary}"
        )

    @app.get("/api/v1/records")
    def records(limit: int = 20) -> JSONResponse:
        return JSONResponse(
            pipeline.recorder.recent(limit=min(limit, 200),
                                     station_id=pipeline.config.station_id)
        )

    @app.get("/api/v1/stats/overview")
    def overview() -> JSONResponse:
        return JSONResponse(pipeline.summary())

    @app.websocket("/ws/live")
    async def live(socket: WebSocket) -> None:
        await socket.accept()
        seen = state.latest_seq()
        try:
            while True:
                snapshot = state.snapshot()
                if snapshot:
                    await socket.send_text(json.dumps(snapshot, ensure_ascii=False))

                seen, fresh = state.events_since(seen)
                for event in fresh:
                    await socket.send_text(json.dumps(event, ensure_ascii=False))

                if state.error:
                    await socket.send_text(json.dumps(
                        {"type": "error", "message": state.error}, ensure_ascii=False
                    ))
                await asyncio.sleep(0.4)
        except WebSocketDisconnect:
            return
        except RuntimeError:
            return          # 客户端半路关连接

    return app


def serve(
    pipeline: DetectionPipeline,
    source: str | int,
    host: str = "127.0.0.1",
    port: int = 8000,
    loop_video: bool = True,
) -> int:
    """启动界面。视频文件默认循环播放，方便演示。"""
    import uvicorn

    state = LiveState()
    thread = threading.Thread(
        target=capture_loop, args=(pipeline, source, state, loop_video), daemon=True
    )
    thread.start()

    print(f"\n界面地址： http://{host}:{port}")
    print(f"视频源：   {source!r}")
    print("按 Ctrl+C 停止\n")

    try:
        uvicorn.run(create_app(pipeline, state), host=host, port=port, log_level="warning")
    except KeyboardInterrupt:
        pass
    finally:
        state.running = False
        thread.join(timeout=2.0)

    if state.error:
        print(f"采集线程报错：{state.error}")
        return 1
    return 0
