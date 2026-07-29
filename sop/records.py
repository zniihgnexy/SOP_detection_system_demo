"""检测记录持久化（设计文档 §4.3.4）。

设计文档指定 PostgreSQL + Redis；demo 用 **SQLite 单文件**替代：表结构和字段名
保持一致，但零配置、免安装、免起服务。三张表就是文档里的那三张：

    sop_templates       笔型模板（一个笔型一个版本一行）
    sop_step_details    模板展开后的步骤明细，便于用 SQL 直接查
    detection_records   每次装配的检测结果

只依赖标准库 sqlite3，因此回放模式无需安装任何第三方包。
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .fsm import Anomaly, SOPTemplate

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sop_templates (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name  TEXT NOT NULL,
    version     TEXT NOT NULL,
    steps_json  TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    UNIQUE (model_name, version)
);

CREATE TABLE IF NOT EXISTS sop_step_details (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    template_id INTEGER NOT NULL REFERENCES sop_templates (id),
    step_order  INTEGER NOT NULL,
    step_id     TEXT    NOT NULL,
    step_name   TEXT    NOT NULL,
    action      TEXT    NOT NULL,
    target_part TEXT,
    timeout_ms  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS detection_records (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp            TEXT    NOT NULL,
    station_id           TEXT    NOT NULL,
    template_id          INTEGER REFERENCES sop_templates (id),
    result               TEXT    NOT NULL,   -- PASS / FAIL
    anomaly_type         TEXT,               -- 首个异常类型，与 §4.3.4 字段一致
    anomalies_json       TEXT    NOT NULL,   -- 全部异常明细
    steps_completed_json TEXT    NOT NULL,
    duration_ms          INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_records_station ON detection_records (station_id);
CREATE INDEX IF NOT EXISTS idx_records_time    ON detection_records (timestamp);
"""


class Recorder:
    """检测记录读写。用作上下文管理器：``with Recorder("sop.db") as rec:``"""

    def __init__(self, db_path: str | Path = "sop.db") -> None:
        self.path = Path(db_path)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # --- 模板 ---

    def upsert_template(self, template: SOPTemplate) -> int:
        """写入模板并展开步骤明细，返回 template_id。同名同版本则复用已有行。"""
        cur = self._conn.execute(
            "SELECT id FROM sop_templates WHERE model_name = ? AND version = ?",
            (template.model_name, template.version),
        )
        row = cur.fetchone()
        if row:
            return int(row["id"])

        steps_json = json.dumps(
            [
                {
                    "id": s.id,
                    "name": s.name,
                    "expected_action": s.expected_action,
                    "target_part": s.target_part,
                    "parent_part": s.parent_part,
                    "timeout_ms": s.timeout_ms,
                    "required_parts_present": list(s.required_parts_present),
                    "optional": s.optional,
                }
                for s in template.steps
            ],
            ensure_ascii=False,
        )
        cur = self._conn.execute(
            "INSERT INTO sop_templates (model_name, version, steps_json, created_at) "
            "VALUES (?, ?, ?, ?)",
            (template.model_name, template.version, steps_json, _now()),
        )
        template_id = int(cur.lastrowid)

        self._conn.executemany(
            "INSERT INTO sop_step_details "
            "(template_id, step_order, step_id, step_name, action, target_part, timeout_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (template_id, i + 1, s.id, s.name, s.expected_action,
                 s.target_part, s.timeout_ms)
                for i, s in enumerate(template.steps)
            ],
        )
        self._conn.commit()
        return template_id

    # --- 检测记录 ---

    def save_record(
        self,
        *,
        station_id: str,
        template_id: int | None,
        result: str,
        anomalies: tuple[Anomaly, ...],
        steps_completed: tuple[str, ...],
        duration_ms: int,
    ) -> int:
        payload = [
            {
                "type": a.type,
                "message": a.message,
                "step_id": a.step_id,
                "timestamp_ms": a.timestamp_ms,
            }
            for a in anomalies
        ]
        cur = self._conn.execute(
            "INSERT INTO detection_records "
            "(timestamp, station_id, template_id, result, anomaly_type, "
            " anomalies_json, steps_completed_json, duration_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                _now(),
                station_id,
                template_id,
                result,
                anomalies[0].type if anomalies else None,
                json.dumps(payload, ensure_ascii=False),
                json.dumps(list(steps_completed), ensure_ascii=False),
                duration_ms,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def recent(self, limit: int = 20, station_id: str | None = None) -> list[dict]:
        """最近的检测记录，新的在前。供 Web 界面的记录列表使用。"""
        if station_id:
            cur = self._conn.execute(
                "SELECT * FROM detection_records WHERE station_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (station_id, limit),
            )
        else:
            cur = self._conn.execute(
                "SELECT * FROM detection_records ORDER BY id DESC LIMIT ?", (limit,)
            )
        return [_row_to_dict(r) for r in cur.fetchall()]

    def summary(self, station_id: str | None = None) -> dict:
        """合格率汇总（§4.5 统计分析的最小版本）。"""
        where, args = ("WHERE station_id = ?", (station_id,)) if station_id else ("", ())
        cur = self._conn.execute(
            f"SELECT COUNT(*) AS total, "
            f"       SUM(CASE WHEN result = 'PASS' THEN 1 ELSE 0 END) AS passed "
            f"FROM detection_records {where}",
            args,
        )
        row = cur.fetchone()
        total = int(row["total"] or 0)
        passed = int(row["passed"] or 0)
        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(passed / total, 4) if total else None,
        }

    # --- 生命周期 ---

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Recorder":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _row_to_dict(row: sqlite3.Row) -> dict:
    out = dict(row)
    out["anomalies"] = json.loads(out.pop("anomalies_json"))
    out["steps_completed"] = json.loads(out.pop("steps_completed_json"))
    return out
