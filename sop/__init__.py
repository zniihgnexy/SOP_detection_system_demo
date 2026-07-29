"""装笔顺序 SOP 智能检测系统 —— 核心库。

模块划分（括号内为设计文档 docs/design.md 的对应章节）：

    fsm.py          SOP 时序匹配引擎，纯标准库          (§4.3)
    records.py      检测记录持久化 SQLite，纯标准库      (§4.3.4)
    features.py     298 维特征工程                      (§4.2.4)
    model.py        Bi-LSTM 动作识别模型                (§4.2.5)
    perception.py   YOLOv8n + MediaPipe Hands 封装      (§4.1 / §4.2.3)
    pipeline.py     串联感知 → 特征 → 动作 → 判定        (§5.2)

本文件刻意不 import 任何子模块：fsm 与 records 只依赖标准库，
这样 ``run.py --replay`` 在没有安装 torch / mediapipe / opencv 时依然可用。
"""

__version__ = "1.0.0"
