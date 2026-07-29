# 装笔顺序 SOP 智能检测系统

> Assembly-order SOP compliance detection for pen assembly lines.
> A fixed overhead camera watches the workbench; YOLOv8n locates the parts and
> MediaPipe tracks the hand; a 298-dim per-frame feature vector feeds a Bi-LSTM
> that classifies 7 assembly actions; a finite state machine matches the action
> sequence against a per-model SOP template and flags missing, out-of-order,
> duplicated, wrong-part, timed-out and dropped steps.

**📖 先看这个 → [中文使用手册](docs/user-manual-zh.md)** ｜ 算法方案见 [设计文档](docs/design.md)

---

## 这是什么

装配线上一台俯视相机盯着工作台，系统实时判断工人有没有**按标准顺序**装笔，
漏装、错序、多装、用错零件、超时都会当场报出来，并留下可追溯的记录。

```
相机 ──┬─▶ YOLOv8n 零件检测 ──┐
       └─▶ MediaPipe 手部关键点 ─┤
                                ▼
                    298 维/帧特征 ──▶ 16 帧滑窗 ──▶ Bi-LSTM ──▶ 7 类动作
                                                                   │
                                        SOP 模板 ──▶ 状态机 ◀───────┘
                                                       │
                                              合格 / 6 类异常 ──▶ 记录 + 界面
```

## ⚠ 开箱不能直接检测

这个仓库提供的是**完整可运行的工程**，但**不含模型权重**——权重必须用你自己产线的
数据训练，因为模型得认识你的零件、你的光照、你工人的手速，这部分没人能代劳。

| 能立刻跑 | 需要先准备数据 |
|---|---|
| 环境自检 `--selfcheck` | 零件检测（要标注图片训 YOLO） |
| **SOP 判定层回放 `--replay`** | 动作识别（要标注动作训 Bi-LSTM） |
| 手部关键点、298 维特征 | 真实视频/摄像头检测 |

判定层（状态机、6 类异常、超时、跳步）**逻辑完整且可立刻验证**，不依赖任何权重。

## 三分钟上手

需要 **Python 3.10 / 3.11 / 3.12**（MediaPipe 没有 3.13、3.14 的安装包）。

```bash
git clone git@github.com:zniihgnexy/SOP_detection_system_demo.git
cd SOP_detection_system_demo

python3.11 -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run.py --selfcheck                                # 检查环境
python run.py --replay configs/demo_sequence.json        # 看判定层跑起来
```

最后一条会跑 4 个示例：正常装配、漏装弹簧、错序、超时，并逐条核对判定结果。
全部符合预期时退出码为 0。

接下来怎么采数据、标注、训练、上线，**一步一步都写在 [中文使用手册](docs/user-manual-zh.md) 里**。

## 目录结构

```
run.py                     入口：--selfcheck / --replay / --video / --camera
sop/
  fsm.py                   SOP 状态机与 6 类异常判定（纯标准库）
  records.py               检测记录 SQLite（纯标准库）
  features.py              298 维特征工程
  model.py                 Bi-LSTM + 时序注意力
  perception.py            YOLOv8n + MediaPipe 封装、叠加层绘制
  pipeline.py              串联感知 → 特征 → 动作 → 判定
scripts/train_action.py    训练动作识别模型（含特征缓存与 ONNX 导出）
web/                       单页监控界面（FastAPI + 原生 HTML）
configs/
  gelpen_0.5.json          SOP 模板，换笔型改这个
  pen_parts_dataset.yaml   YOLO 数据集定义
  demo_sequence.json       回放用的示例动作序列
docs/
  user-manual-zh.md        中文使用手册 ← 从这里开始
  design.md                原始需求分析与方案设计文档
tests/                     92 个用例，pytest -q
```

## 与设计文档的差异

`docs/design.md` 描述的是完整产线系统。本仓库按「刚好够用」实现，明确简化了：

| 设计文档 | 本仓库 |
|---|---|
| PostgreSQL + Redis | SQLite 单文件，表结构照搬，零配置 |
| 7 个 Docker 服务 | 单进程 |
| Vue 3 SPA + 拖拽式 SOP 编辑器 | 单页原生 HTML；SOP 模板直接改 JSON |
| TensorRT / OpenVINO / NCNN 多后端 | PyTorch，附 ONNX 导出 |
| 多工位编排、MES / PLC 联动、告警推送 | 单工位，告警落终端与记录 |
| DTW 自动生成 SOP 模板、权限系统 | 未实现 |

设计文档里有 6 处规格自相矛盾（298 维单手还是双手、S 编号语义、复合状态转移等），
每一处的取法都写在对应模块的注释里，没有含糊带过。详见
[手册第 12 章](docs/user-manual-zh.md#12-这个-demo-不做什么)。

## 验收指标（来自设计文档 §7 Phase 2）

| 指标 | 目标 |
|---|---|
| 零件检测 mAP@0.5 | ≥ 0.90 |
| 动作分类准确率（7 类） | ≥ 92%，各类召回率 > 88% |
| 异常检出召回率 | ≥ 98% |
| 误报率 | ≤ 2% |
| 端到端时延 | ≤ 50ms |

`scripts/train_action.py` 训练结束会打印准确率、各类召回率和混淆矩阵，并按 92%
这条线决定退出码。
