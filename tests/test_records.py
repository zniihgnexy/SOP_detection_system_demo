"""检测记录持久化测试（sop/records.py）。"""

from __future__ import annotations

from pathlib import Path

from sop import fsm
from sop.records import Recorder

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = fsm.load_template(ROOT / "configs" / "gelpen_0.5.json")

ANOMALIES = (
    fsm.Anomaly(fsm.AnomalyType.MISSING_STEP, "漏装：步骤 S2 装弹簧 被跳过", "S2", 2600),
    fsm.Anomaly(fsm.AnomalyType.TIMEOUT, "超时：步骤 S4 装笔尖", "S4", 9000),
)


def test_schema_is_created(tmp_path):
    with Recorder(tmp_path / "a.db") as rec:
        tables = {
            row[0] for row in
            rec._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert {"sop_templates", "sop_step_details", "detection_records"} <= tables


def test_upsert_template_is_idempotent(tmp_path):
    with Recorder(tmp_path / "b.db") as rec:
        first = rec.upsert_template(TEMPLATE)
        second = rec.upsert_template(TEMPLATE)
        assert first == second
        count = rec._conn.execute("SELECT COUNT(*) FROM sop_templates").fetchone()[0]
        assert count == 1


def test_step_details_expanded_in_order(tmp_path):
    with Recorder(tmp_path / "c.db") as rec:
        template_id = rec.upsert_template(TEMPLATE)
        rows = rec._conn.execute(
            "SELECT step_order, step_id, action, target_part, timeout_ms "
            "FROM sop_step_details WHERE template_id = ? ORDER BY step_order",
            (template_id,),
        ).fetchall()

    assert len(rows) == 6
    assert [r["step_id"] for r in rows] == ["S1", "S2", "S3", "S4", "S5", "S6"]
    assert rows[0]["action"] == "Pick" and rows[0]["target_part"] == "barrel"
    assert rows[1]["timeout_ms"] == 5000


def test_save_and_read_back(tmp_path):
    with Recorder(tmp_path / "d.db") as rec:
        template_id = rec.upsert_template(TEMPLATE)
        record_id = rec.save_record(
            station_id="ST07",
            template_id=template_id,
            result="FAIL",
            anomalies=ANOMALIES,
            steps_completed=("S1", "S3"),
            duration_ms=6400,
        )
        assert record_id > 0

        records = rec.recent(limit=10)

    assert len(records) == 1
    row = records[0]
    assert row["station_id"] == "ST07"
    assert row["result"] == "FAIL"
    assert row["duration_ms"] == 6400
    # anomaly_type 存首个异常，与设计文档 4.3.4 的字段保持一致
    assert row["anomaly_type"] == "MISSING_STEP"
    assert len(row["anomalies"]) == 2
    assert row["anomalies"][1]["step_id"] == "S4"
    assert row["steps_completed"] == ["S1", "S3"]


def test_recent_is_newest_first_and_filters_by_station(tmp_path):
    with Recorder(tmp_path / "e.db") as rec:
        for station in ("ST01", "ST02", "ST01"):
            rec.save_record(
                station_id=station, template_id=None, result="PASS",
                anomalies=(), steps_completed=("S1",), duration_ms=100,
            )

        everything = rec.recent()
        only_st01 = rec.recent(station_id="ST01")

    assert [r["id"] for r in everything] == [3, 2, 1]
    assert len(only_st01) == 2
    assert all(r["station_id"] == "ST01" for r in only_st01)


def test_summary_pass_rate(tmp_path):
    with Recorder(tmp_path / "f.db") as rec:
        assert rec.summary() == {
            "total": 0, "passed": 0, "failed": 0, "pass_rate": None
        }

        for result in ("PASS", "PASS", "PASS", "FAIL"):
            rec.save_record(
                station_id="ST01", template_id=None, result=result,
                anomalies=(), steps_completed=(), duration_ms=1,
            )

        assert rec.summary() == {
            "total": 4, "passed": 3, "failed": 1, "pass_rate": 0.75
        }


def test_passing_record_has_null_anomaly_type(tmp_path):
    with Recorder(tmp_path / "g.db") as rec:
        rec.save_record(
            station_id="ST01", template_id=None, result="PASS",
            anomalies=(), steps_completed=("S1",), duration_ms=10,
        )
        row = rec.recent()[0]

    assert row["anomaly_type"] is None
    assert row["anomalies"] == []
