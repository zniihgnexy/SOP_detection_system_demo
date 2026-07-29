"""回放模式端到端测试（run.py --replay）。

configs/demo_sequence.json 里每个序列都写了 expect 字段，run.py 会逐个核对，
全部符合预期时退出码为 0。这个测试因此同时守住了判定引擎和示例数据。
"""

from __future__ import annotations

from pathlib import Path

import run

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "configs" / "demo_sequence.json"


def test_all_demo_sequences_match_expectations(tmp_path):
    code = run.main(["--replay", str(DEMO), "--db", str(tmp_path / "replay.db")])
    assert code == 0


def test_replay_writes_one_record_per_sequence(tmp_path):
    import json

    from sop.records import Recorder

    db = tmp_path / "replay.db"
    run.main(["--replay", str(DEMO), "--db", str(db)])

    expected = len(json.loads(DEMO.read_text(encoding="utf-8"))["sequences"])
    with Recorder(db) as rec:
        records = rec.recent(limit=100)
        stats = rec.summary()

    assert len(records) == expected
    assert stats["total"] == expected
    assert stats["passed"] == 1          # 4 个示例里只有「正常装配」应该合格


def test_selfcheck_runs_without_crashing(tmp_path):
    """自检在缺依赖的机器上应返回 1 而不是抛异常。"""
    code = run.main(["--selfcheck", "--db", str(tmp_path / "x.db")])
    assert code in (0, 1)


def test_missing_sequences_key_is_reported(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"station_id": "ST01"}', encoding="utf-8")
    code = run.main(["--replay", str(bad), "--db", str(tmp_path / "y.db")])
    assert code == 2
