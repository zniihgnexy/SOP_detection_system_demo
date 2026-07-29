# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 仓库定位

装笔顺序 SOP 智能检测系统的**可运行实现**。三份文档分工不重叠：

| 文件 | 内容 |
|---|---|
| `README.md` | 短门面：是什么、开箱不能直接检测、三分钟上手 |
| `docs/user-manual-zh.md` | 零基础十三章操作手册，交付给使用者的主文档 |
| `docs/design.md` | 原始需求分析与方案设计文档（1677 行，V1.0，2026-07-27）。**规格的事实来源**，本文引用的 §x.y 都指它 |

**仓库不含模型权重**，必须用使用者自己产线的数据训练。因此 `--video` / `--camera`
开箱跑不动；`--selfcheck` 和 `--replay` 可以。

## 常用命令

```bash
pytest -q                                          # 108 passed, 2 skipped
python run.py --selfcheck                          # 环境自检，不需要权重
python run.py --replay configs/demo_sequence.json  # 判定层验收测试，退出码 0 = 全部符合预期
python run.py --video x.mp4 --web                  # 真实检测 + 界面（需权重）

yolo detect train model=yolov8n.pt data=configs/pen_parts_dataset.yaml \
  epochs=100 imgsz=544 batch=16 lr0=0.001 lrf=0.01 optimizer=AdamW
python scripts/train_action.py --yolo models/pen_parts.pt   # 抽特征+训练+评估+导出 ONNX

pytest -q tests/test_fsm.py -k timeout             # 跑单个用例
```

## 架构与一条硬约束

```
run.py ──┬── --selfcheck / --replay ──▶ sop/fsm.py + sop/records.py   ← 仅标准库
         └── --video / --camera ──────▶ sop/pipeline.py
                                          ├── sop/perception.py   cv2 + mediapipe + ultralytics
                                          ├── sop/features.py     numpy
                                          ├── sop/model.py        torch
                                          ├── sop/fsm.py
                                          └── sop/records.py
web/server.py ──▶ sop/pipeline.py（后台线程跑采集与推理，Web 只读最近一帧）
```

**`sop/fsm.py`、`sop/records.py`、`sop/__init__.py` 必须保持只依赖标准库。**
这不是风格偏好：`--replay` 和 `--selfcheck` 是使用者在没装齐依赖、没有任何权重时
唯一能验证系统的手段。往这三个文件里加一个 `import numpy` 就会让这条路径失效。
`sop/__init__.py` 因此刻意不 import 任何子模块。

数据流（§5.2）：两条感知分支 **汇入同一个 298 维特征向量**，不是各自独立判定。
动作模型只吃特征序列，判定引擎只吃动作标签 —— 三段之间的契约就是下面的常量。

## 不能漂移的常量

改动任何一处都要同步改另一处，否则是静默失败（不报错，只是一直检测不出来）。

| 常量 | 定义处 | 必须同步 | 有测试守着 |
|---|---|---|---|
| `ACTIONS` 7 类动作及顺序 | `sop/fsm.py` | 模型输出通道、标注 JSON、权重里存的类别表 | ✓ |
| `PARTS` 6 类零件及顺序 | `sop/fsm.py` | `configs/pen_parts_dataset.yaml` 的 `names` | ✓ |
| `FEATURE_DIM = 298` 及七组布局 | `sop/features.py` `OFFSETS` | 模型 `input_dim`、ONNX 输入形状 | ✓ |
| `WINDOW_SIZE=16` / `STRIDE=2` | `sop/features.py` | 位置编码长度、ONNX dummy 形状 | ✓ |
| `MIN_CONFIDENCE = 0.6` | `sop/fsm.py` | §4.3.2 Step A | ✓ |
| 6 类异常代号 | `sop/fsm.py` `AnomalyType` | 数据库 `anomaly_type`、WS 事件、手册 | ✓ |

`load_checkpoint` 会比对权重里存的 `actions` 和当前 `ACTIONS`，不一致直接报错要求重训 ——
比让它悄悄错着跑好。

## 设计文档 6 处矛盾的裁决

这些是**已经拍板的决定**，不要再重新推导。每处的理由写在对应文件的模块注释里，
汇总见手册第 12 章。

1. **298 维取单手**（置信度最高那只为主操作手）。§4.2.4 按单手推算，§4.2.3 却配了双手；
   298 这个数字被 §4.2.5 / §4.2.7 / §6.2 反复引用，改双手要连带推翻三处。
2. **步骤 = 一个装配动作，共 6 步**。依据 §4.3.4 的表 + §8.3.2 进度条 + §8.3.3 编辑器
   三处一致。§4.3.1 的 `S2_SPRING` 这套「后置条件」命名不再维护，状态由步骤下标推导。
3. **中间的 `Pick` / `Align` / `Idle` 是过渡动作**，不匹配当前步骤时不推进也不报异常；
   `Insert` / `Screw` / `Press` / `Place` 是提交动作，不匹配即判异常。
4. **§4.3.2 / §5.3 伪代码调用的 7 个方法**都实现在 `SOPTemplate` 上。
5. **32 维零件嵌入** = Top-4 零件各一个 6 类 one-hot（24）+ 各自置信度（4）+ 补 4 个零。
   不从 YOLO 内部取 embedding —— 那会破坏「298 维是纯预处理输出」这个边界。
6. **不做认证**，手册明确写只在内网用。

另外两处训练取舍：只实现 Focal Loss（跳过 §4.2.6 的 center / smooth 损失，原因见
`sop/model.py` 模块注释）；数据增强只做抖动 / 时间缩放 / Mixup。

## 环境与协作上的坑

- **目标 Python 是 3.10–3.12**（MediaPipe 无 3.13/3.14 轮子），但**本机只有 3.13 和 3.14**。
  这意味着 3.12+ 才支持的 f-string 写法（表达式内含反斜杠、同类型引号嵌套）在本机测不出来，
  却会让使用者直接 `SyntaxError`。写完新代码用 `tests/test_repo_health.py` 的编译检查加
  人工扫一遍。
- **pre-push 钩子会跑 `pytest -q`**，且「零个用例」也算失败。它用的是
  `/opt/anaconda3/bin/pytest`（Python 3.13，只有 numpy），所以 torch / cv2 相关的用例
  在本机会 skip，在装齐依赖的机器上才真跑。
- **remote 已从 HTTPS 切成 SSH**（`git@github.com:zniihgnexy/...`），因为没有 https 凭据助手。
- **提交不带 Co-Authored-By**，遵循用户 `rules/common/git-workflow.md` 里「归属已全局禁用」。
- **Ultralytics 的 `path:` 必须是绝对路径**：它相对自己的 `datasets_dir`（默认 `~/datasets`）
  解析，不是相对 yaml 所在目录。这个坑已经写在 yaml 注释和手册 5.1 节里。

## 刻意没做的

单工位（多工位就是多起进程，各用一个 `--station` 和 `--db`）｜SQLite 代替
PostgreSQL+Redis｜单进程代替 7 个 Docker 服务｜原生 HTML 单页代替 Vue SPA 与拖拽式
SOP 编辑器｜PyTorch 代替 TensorRT/OpenVINO/NCNN 多后端｜无告警推送、无 MES/PLC 联动、
无权限系统、无 DTW 自动生成模板、无 Transformer 版动作模型｜`DROPPED` 异常类型定义了
但没有触发逻辑（需要跨帧零件追踪）。

要加功能前先确认它是否在这份清单里 —— 在的话说明是有意省略，先和用户确认再动手。

## 设计文档导航（`docs/design.md` 行号）

| 节 | 行 | 节 | 行 |
|---|---|---|---|
| 3.1 五层架构 | 61 | 4.3.2 实时匹配算法 | 568 |
| 4.2.1 动作类别定义 | 145 | 4.3.4 多型号 SOP 管理 | 631 |
| 4.2.4 特征工程 | 254 | 4.4 异常与告警 | 655 |
| 4.2.5 时序模型架构 | 311 | 5.3 判定逻辑伪代码 | 729 |
| 4.2.6 训练策略 | 389 | 6.4 环境依赖表 | 831 |
| 4.2.7 推理部署优化 | 432 | 七 实施路线图 | 877 |
| 4.3.1 SOP 形式化建模 | 502 | 8.5 前后端接口 | 1554 |
