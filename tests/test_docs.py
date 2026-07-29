"""文档一致性检查。

手册里写的命令行参数必须真的存在，链接必须真的能点开。
文档和代码脱节是这类交付物最常见的问题，而且用户踩到时会以为是自己搞错了。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
MANUAL = ROOT / "docs" / "user-manual-zh.md"
DESIGN = ROOT / "docs" / "design.md"

#: 第三方工具的参数（python / pip / pytest / sqlite3），不拿去和本项目的 argparse 比对
FOREIGN_FLAGS = {"--help", "--version"}


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def slugify(heading: str) -> str:
    """粗略复刻 GitHub 的标题锚点生成规则。"""
    text = heading.strip().lower()
    text = re.sub(r"[^\w一-鿿\s-]", "", text)
    return re.sub(r"\s+", "-", text.strip())


# ------------------------------------------------------------- 文件存在


def test_three_documents_exist():
    for path in (README, MANUAL, DESIGN):
        assert path.is_file(), path
        assert len(read(path)) > 500, f"{path} 内容太短"


def test_readme_points_at_manual_first():
    """README 第一个链接应该是手册 —— clone 下来第一眼该看怎么用。"""
    head = read(README)[:1200]
    assert "docs/user-manual-zh.md" in head


def test_design_doc_is_the_original_spec():
    text = read(DESIGN)
    assert "装笔顺序SOP智能检测系统" in text
    assert "需求分析与方案设计文档" in text


# --------------------------------------------------------- 命令行参数


def collect_documented_flags() -> set[str]:
    flags: set[str] = set()
    for path in (README, MANUAL):
        flags |= set(re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]+)", read(path)))
    return flags - FOREIGN_FLAGS


def collect_real_flags() -> set[str]:
    flags: set[str] = set()
    for rel in ("run.py", "scripts/train_action.py"):
        source = read(ROOT / rel)
        flags |= set(re.findall(r'"(--[a-z][a-z0-9-]+)"', source))
    return flags


def test_every_documented_flag_exists():
    documented = collect_documented_flags()
    real = collect_real_flags()
    unknown = documented - real
    assert not unknown, f"手册里写了不存在的参数：{sorted(unknown)}"


def test_core_flags_are_documented():
    """这几个参数是主流程必经的，手册里必须提到。"""
    documented = collect_documented_flags()
    for flag in ("--selfcheck", "--replay", "--video", "--camera", "--web", "--yolo"):
        assert flag in documented, flag


# ---------------------------------------------------------------- 链接


def collect_links(path: Path) -> list[str]:
    return re.findall(r"\[[^\]]*\]\(([^)]+)\)", read(path))


@pytest.mark.parametrize("source", [README, MANUAL], ids=["README", "manual"])
def test_relative_links_resolve(source):
    broken = []
    for link in collect_links(source):
        if link.startswith(("http://", "https://", "#", "mailto:")):
            continue
        target, _, _anchor = link.partition("#")
        if not target:
            continue
        if not (source.parent / target).resolve().exists():
            broken.append(link)
    assert not broken, f"{source.name} 里有失效链接：{broken}"


def test_readme_anchor_into_manual_resolves():
    anchors = {slugify(m) for m in re.findall(r"^#{1,6}\s+(.*)$", read(MANUAL), re.M)}
    for link in collect_links(README):
        target, _, anchor = link.partition("#")
        if target.endswith("user-manual-zh.md") and anchor:
            assert anchor in anchors, (
                f"README 指向手册的锚点 #{anchor} 不存在。现有锚点：{sorted(anchors)}"
            )


# ------------------------------------------------ 手册与代码事实一致


def test_manual_lists_all_seven_actions():
    from sop.fsm import ACTIONS

    text = read(MANUAL)
    for action in ACTIONS:
        assert f"`{action}`" in text, action


def test_manual_lists_all_six_parts():
    from sop.fsm import PARTS

    text = read(MANUAL)
    for part in PARTS:
        assert f"`{part}`" in text, part


def test_manual_lists_all_six_anomaly_types():
    from sop.fsm import AnomalyType

    text = read(MANUAL)
    kinds = [v for k, v in vars(AnomalyType).items() if not k.startswith("_")]
    assert len(kinds) == 6
    for kind in kinds:
        assert f"`{kind}`" in text, kind


def test_manual_states_the_python_version_limit():
    text = read(MANUAL)
    assert "3.10" in text and "3.12" in text
    assert "3.13" in text and "3.14" in text      # 必须点明哪两个版本不行


def test_manual_warns_about_binding_to_all_interfaces():
    """--host 0.0.0.0 没有认证，手册必须警告不要暴露公网。"""
    text = read(MANUAL)
    assert "0.0.0.0" in text
    assert "公网" in text


def test_manual_says_repo_cannot_detect_out_of_the_box():
    """最重要的前置条件，README 和手册都必须在开头讲清楚。"""
    for path in (README, MANUAL):
        head = read(path)[:1500]
        assert "权重" in head, path


def test_step_count_in_manual_matches_template():
    from sop.fsm import load_template

    template = load_template(ROOT / "configs" / "gelpen_0.5.json")
    text = read(MANUAL)
    for step in template.steps:
        assert step.id in text, step.id
        assert step.name in text, step.name
