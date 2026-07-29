# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 仓库现状

- 仓库当前**只有 `readme.md` 一个文件**（1677 行，中文）：《装笔顺序SOP智能检测系统 — 需求分析与方案设计文档》V1.0 初稿，日期 2026-07-27。
- **没有任何代码、依赖清单、构建脚本、配置文件或测试。** 收到"运行/构建/测试"类请求时，先确认是要从零搭建工程，而不是去找不存在的入口。
- `readme.md` 是唯一事实来源：改设计 → 改 `readme.md`；写代码 → 把它当规格书，并保持下文"规格常量"一致。
- 文档正文用中文，技术标识符（动作名、零件类名、状态名、API 路径、事件类型）用英文。新增内容沿用这一约定。
- 文档分析所依据的 8 帧素材（`frame1_000034` ~ `frame1_000041`，960×544 / 25fps / 俯视固定机位）**不在仓库内**。

## 系统架构

五层分层架构（§3.1），下表把每层映射到 §7 Phase 3 规划的容器服务：

| 层 | 职责 | 规划服务 |
|---|---|---|
| 视频采集层 | 工业相机多工位并行采集（USB/GigE/RTSP），≥25fps、≥960×544 | `camera-service` |
| 图像预处理层 | 去噪、ROI 裁切、直方图均衡光照补偿、自适应帧采样 | `camera-service` |
| 视觉分析引擎层 | 目标检测 / 动作识别 / 时序建模 / 异常检测 | `inference-service`（GPU） |
| 业务逻辑层 | SOP 模板管理、实时判定引擎、告警推送、数据记录 | `sop-engine` + `web-backend` |
| 展示与交互层 | 监控大屏、管理后台、移动端卡片推送 | `web-frontend` |

## 端到端数据流（理解本系统的关键）

这条链路横跨 §4.2、§4.3、§5.2、§6.2，是最需要一次性掌握的结构：

```
Camera(25fps) → 预处理
                  ├─▶ YOLOv8n        (零件框 + 类别)  ──┐
                  └─▶ MediaPipe Hands (21 关键点/手)  ──┤
                                                        ▼
                                       特征工程 → 298 维/帧
                                                        ▼
                                   16 帧滑动窗口 (stride=2)
                                                        ▼
                                   Bi-LSTM + Temporal Attention
                                                        ▼
                                     7 类动作 + confidence
                                                        ▼
                              FSM 状态机 ⟵ SOP 模板 (per 笔型)
                                                        ▼
                                PASS / 异常 → 告警 + 记录 + 关键帧
```

两条感知分支（YOLO 与 MediaPipe）**汇入同一个特征向量**，不是各自独立判定——这是最容易实现错的地方。动作识别只吃特征序列，不吃原图；SOP 判定只吃动作标签序列，不吃特征。三段之间的契约就是下面的常量。

时延预算（§6.2）：单帧感知 ~15ms + 动作识别 ~3ms + SOP 判定 <0.1ms ≈ **18ms**，对 40ms 帧间隔留有余量。

离线侧另有一条链路（§4.3.3）：3–5 段标准演示视频 → 动作序列 → DTW 对齐 → 多数投票定标准动作 → 时间分布 均值±2σ 作为超时阈值 → 自动生成 `SOPTemplate`。

## 规格常量（跨模块契约，实现时必须严格一致）

### 7 类动作（§4.2.1，模型输出顺序即此顺序）

`Pick` 拿取 ｜ `Align` 对准 ｜ `Insert` 插入 ｜ `Screw` 旋拧 ｜ `Press` 按压 ｜ `Place` 放置 ｜ `Idle` 闲置

`Idle` 样本量可达其他类 10 倍以上 → 用 Focal Loss（γ=2.0）或 weighted RandomSampler。

### 6 类零件（§7 Phase 1 标注类别，YOLO 类名）

`barrel` 笔杆 ｜ `cap` 笔帽 ｜ `spring` 弹簧 ｜ `refill` 笔芯 ｜ `tip` 笔尖锥套 ｜ `grip` 握胶

### FSM 状态（§4.3.1，状态名表示"已完成的后置条件"）

```
S0_IDLE ──Pick(barrel)──▶ S1_HOLD ──Pick+Insert(spring)──▶ S2_SPRING
        ──Pick+Insert(refill)──▶ S3_REFILL ──Pick+Screw(tip)──▶ S4_TIP
        ──Pick+Press(cap)──▶ S5_CAP ──Place(finished_pen)──▶ S6_DONE
```

### 298 维特征向量分解（§4.2.4，六处引用必须对齐）

| 特征组 | 维度 | 说明 |
|---|---|---|
| 关键点原始坐标 | 63 | MediaPipe (x,y,z) 归一化至 [0,1] |
| 关键点相对手掌中心 | 63 | 各点减 `landmark[0]`（手腕），获得姿态不变性 |
| 手-零件空间关系 | 12 | 最近 **Top-4** 零件 ×（距离 2 维 + 手/零件框 IoU 1 维），不足补零 |
| 手指开合度 | 2 | `‖landmark[4]−landmark[8]‖` + 阈值 0.05 的二值抓取态 |
| 帧间速度 | 63 | `v(t)=p(t)−p(t−1)` |
| 帧间加速度 | 63 | `a(t)=v(t)−v(t−1)` |
| 零件类别嵌入 | 32 | YOLO cls embedding |
| **合计** | **298** | 模型输入 `[batch, 16, 298]` |

### 关键阈值

| 参数 | 值 | 出处 |
|---|---|---|
| 滑动窗口 / 步长 | `window=16` (0.64s @25fps) / `stride=2` | §4.2.4 |
| MediaPipe | `max_num_hands=2`, `model_complexity=1`, det=0.7, track=0.5 | §4.2.3 |
| 抓取判定距离阈值 | 0.05 | §4.2.4 |
| 动作有效性下限 | `MIN_CONFIDENCE = 0.6`（低于则丢弃为噪声） | §4.3.2 |
| 可疑动作提示级 | confidence < 0.70 | §8.3.6 |
| 步骤超时示例 | Pick(barrel) 3000ms、Insert(spring) 5000ms | §4.3.4 |

### 异常类型（§4.4；DB 与 WebSocket 用 SCREAMING_SNAKE）

`MISSING_STEP` 漏装 ｜ `WRONG_ORDER` 错序 ｜ `EXTRA_STEP` 多装 ｜ `WRONG_PART` 零件错误 ｜ `TIMEOUT` 装配超时 ｜ `DROPPED` 零件掉落

§5.3 伪代码对应异常类：`SequenceError` / `PartError` / `TimeoutError` / `MissingStepError`。

告警分级（§8.3.6）：🔴 严重（连续 3 次异常 / 零件错误）→ 声光+企微+短信，5 分钟响应；🟡 一般（漏装/错序/超时）→ 企微卡片，15 分钟确认；🔵 提示（confidence<0.7）→ 仅记录。

### 验收 KPI（§7 Phase 2 / Phase 4）

零件检测 mAP@0.5 ≥ 0.90（mAP@0.5:0.95 > 0.70）｜动作分类 Accuracy ≥ 92%、per-class Recall > 88%｜异常检出 Recall ≥ 98%｜误报率 FPR ≤ 2%｜端到端时延 ≤ 50ms｜端到端准确率 > 95%（Phase 2）→ ≥ 97%（Phase 4 验收）

## 模型与训练规格

- **动作模型主选 Bi-LSTM**（§4.2.5）：`input=298, hidden=128, num_layers=2, bidirectional, dropout=0.2` → Temporal Attention 加权池化 → FC(256→128→64→7)，FC 段 dropout=0.3。约 0.5M 参数。
- **备选 Transformer Encoder**：`d_model=256, nhead=4, dim_ff=512, 4 layers, dropout=0.1, GELU`，约 1.2M 参数。两者都训，**通过配置文件切换**，不要硬编码选型。
- **损失**：`L_CE + 0.003·L_center + 0.01·L_smooth`（时序平滑损失约束相邻帧标签一致性）。
- **超参**：AdamW，lr 1e-3 + cosine annealing + 前 5 epoch warmup，wd 1e-4，bs 64，epochs 120，早停 patience=15，AMP FP16。增量微调用 lr 1e-5。
- **时序数据增强**：时间裁剪 [0.8,1.0]、时间缩放 [0.8,1.25]、关键点抖动 σ=0.005、手部平移 ±0.02、Mixup α=0.2；水平翻转仅在相机镜像安装时启用。
- **部署路径**：PyTorch → ONNX (`opset_version=14`, dummy `(1,16,298)`, batch 维动态) → 运行时按 TensorRT > CUDA > CPU 自动选择后端。

## 接口约定（§8.5）

REST base `http://{host}:8000/api/v1`，资源族：`/stations`、`/records`、`/stats/{overview,anomalies,stations}`、`/sop/templates`、`/alerts`。视频流 `GET /stations/{id}/video`（MJPEG/H.264）。

WebSocket `ws://{host}:8000/ws/live`，下行事件 `type`：

- `station_update`（每秒；含 `current_step`/`current_action`/`action_confidence`/`elapsed_ms`/`parts_status`）
- `step_completed`、`anomaly_detected`、`assembly_complete`

`parts_status` 取值：`in_hand` / `assembled` / `operating` / `waiting`。上行仅 `ack_alert`、`request_snapshot`。

数据库（PostgreSQL，§4.3.4）三张核心表：`sop_templates`（含 `steps_json`）、`sop_step_details`、`detection_records`。Redis 存工位实时状态与滑动窗口缓冲。

前端为 Vue 3 + Vite + Element Plus + Pinia + ECharts + Vue Flow（SOP 编辑器画布），组件树见 §8.4；Nginx 分流 `/api/*`、`/ws/*`、静态 `dist/`。视频叠加层配色（§8.3.2）：零件框 `#00C853`、当前操作 `#2979FF`、缺失零件红色虚线 `#FF1744`、手部骨架 `#00BCD4`。

## 命令

**目前无可执行命令**——以下是设计文档规定的形态，实现对应模块后再落地：

```bash
# 训练零件检测（§7 Phase 2.1，Ultralytics）
yolo detect train model=yolov8n.pt data=pen_parts_dataset.yaml \
     epochs=100 imgsz=544 batch=16 lr0=0.001 lrf=0.01 optimizer=AdamW

# 训练动作模型（§7 Phase 2.2 脚本骨架 train_action_model.py）
python train_action_model.py

# 整栈启动（§7 Phase 3：camera / inference / sop-engine / backend / frontend / postgres / redis）
docker compose up -d
```

服务端口：`web-backend` 8000（REST + WebSocket 同一 FastAPI 实例）、`web-frontend` 3000、PostgreSQL 15、Redis 7-alpine。`inference` 需 `runtime: nvidia`，模型挂载在 `/models`。

技术栈版本下限见 §6.4：Python 3.10+、PyTorch 2.0+、Ultralytics 8.x、MediaPipe 0.10+、OpenCV 4.8+、Albumentations 1.3+、ONNX 1.14+ / ONNX Runtime 1.16+、TensorRT 8.6+、FastAPI 0.100+、PostgreSQL 15+、Redis 7+、Vue 3.4+、Vite 5、Docker 24+。

## 文档导航（`readme.md` 行号）

| 节 | 行 | 节 | 行 |
|---|---|---|---|
| 二 素材帧 PSNR 分析 | 21 | 4.3.2 实时匹配算法 | 568 |
| 3.1 五层架构 | 61 | 4.3.3 DTW 模板生成 | 608 |
| 4.1 零件检测 | 122 | 4.3.4 多型号 SOP 管理 | 631 |
| 4.2.1 动作类别定义 | 145 | 4.4 异常与告警 | 655 |
| 4.2.4 特征工程 | 254 | 5.3 判定逻辑伪代码 | 729 |
| 4.2.5 时序模型架构 | 311 | 6.4 环境依赖表 | 831 |
| 4.2.6 训练策略 | 389 | 七 实施路线图 Phase 1–5 | 877 |
| 4.2.7 推理部署优化 | 432 | 8.3 核心页面设计 | 1218 |
| 4.3.1 SOP 形式化建模 | 502 | 8.5 前后端接口 | 1554 |

## 设计文档尚未闭合的问题

动手实现前需要先定夺，不要默默选一个：

1. **单手 vs 双手特征维度冲突。** 298 维中有 252 维（原始+相对+速度+加速度）按**单手** 63 维推导，但 §4.2.3 配置 `max_num_hands=2`（双手 126 维）。需明确：只取主操作手，还是扩到 ~550 维。§4.2.3 与 §4.2.4 都用了"126 维"指代不同含义，注意别混。
2. **S 编号语义不一致。** §4.3.1 的状态名是**后置条件**（`S2_SPRING` = 弹簧已装入），§8.3.2/§8.3.3 界面的 `S2` 是**进行中的动作**（"装弹簧"），且界面进度条不含 `S0`。落库前统一一套编号。
3. **`SOPStep.expected_action` 是单个动作，但 FSM 状态转移消耗两个动作**（如 `Pick(spring)` → `Insert(spring, barrel)`）。需要支持动作序列，或把 Pick 拆成独立步骤（§4.3.4 的 `sop_step_details` 示例正是拆开的）。
4. **`SOPTemplate` 数据类缺少判定引擎实际调用的方法**：`expected_action()`、`valid_skips()`、`required_parts()`、`timeout()`、`transition()`、`all_steps_completed()`、`missing_steps()` 只出现在 §4.3.2/§5.3 伪代码里，尚未定义在 §4.3.1 的 dataclass 上。
5. **32 维零件类别嵌入的生成方式未指定**（YOLO 哪一层、如何池化多个零件）。
6. **权限与认证接口缺失**：§8.2 有"用户 & 权限管理"页面，§8.5.1 的 REST 表里没有对应端点，SOP 模板也无 DELETE。
