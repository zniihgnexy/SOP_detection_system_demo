"""仓库一致性检查。不需要任何第三方依赖（pytest 除外）。

这些用例守的是「两处写了同一件事，改一处忘了另一处」类型的问题。
"""

from __future__ import annotations

import json
import py_compile
import re
from pathlib import Path

import pytest

from sop.fsm import ACTIONS, PARTS

ROOT = Path(__file__).resolve().parent.parent

SOURCES = sorted(
    path for path in ROOT.rglob("*.py")
    if not any(part in {".venv", "venv", "__pycache__", "build"} for part in path.parts)
)


def test_found_the_sources():
    assert len(SOURCES) >= 10


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(ROOT)))
def test_python_files_compile(path, tmp_path):
    py_compile.compile(
        str(path), cfile=str(tmp_path / "out.pyc"), doraise=True
    )


def test_configs_are_valid_json():
    for name in ("gelpen_0.5.json", "demo_sequence.json"):
        payload = json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))
        assert payload, name


def test_yolo_dataset_classes_match_fsm_parts():
    """configs/pen_parts_dataset.yaml 的 names 必须与 fsm.PARTS 同名同序。

    两边不一致会导致 YOLO 输出的标签对不上 SOP 模板里的 target_part，
    而且这种错不会报异常，只会让检测悄悄一直失败。
    """
    text = (ROOT / "configs" / "pen_parts_dataset.yaml").read_text(encoding="utf-8")
    pairs = re.findall(r"^\s+(\d+):\s*([A-Za-z_]+)", text, flags=re.MULTILINE)
    assert pairs, "没在 yaml 里找到 names 列表"

    indexed = [name for _, name in sorted(pairs, key=lambda p: int(p[0]))]
    assert tuple(indexed) == PARTS


def test_sop_template_only_uses_known_actions():
    payload = json.loads((ROOT / "configs" / "gelpen_0.5.json").read_text(encoding="utf-8"))
    for step in payload["steps"]:
        assert step["expected_action"] in ACTIONS, step


def test_demo_sequence_only_uses_known_actions():
    payload = json.loads(
        (ROOT / "configs" / "demo_sequence.json").read_text(encoding="utf-8")
    )
    for sequence in payload["sequences"]:
        for event in sequence["events"]:
            assert event["action"] in ACTIONS, event


def test_demo_sequence_references_existing_template():
    payload = json.loads(
        (ROOT / "configs" / "demo_sequence.json").read_text(encoding="utf-8")
    )
    assert (ROOT / payload["template"]).is_file()


def test_requirements_states_python_version_limit():
    """MediaPipe 装不上 3.13/3.14，这个限制必须写在依赖清单最显眼的地方。"""
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "3.10" in text and "3.12" in text
    assert "mediapipe" in text.lower()


def test_gitignore_excludes_local_claude_settings():
    """.claude/settings.local.json 含本机权限配置，不该进公开仓库。"""
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".claude/" in text
    assert "*.db" in text
    assert "models/" in text
