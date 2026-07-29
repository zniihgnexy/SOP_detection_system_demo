"""动作识别模型（设计文档 §4.2.5）。

主模型 Bi-LSTM + 时序注意力，结构照搬 §4.2.5：

    输入 (B, 16, 298)
      → 可学习位置编码 (1, 16, 298)
      → Bi-LSTM(input=298, hidden=128, layers=2, bidirectional, dropout=0.2)
                                                          → (B, 16, 256)
      → 时序注意力 Scaled Dot-Product，加权后按时间平均池化 → (B, 256)
      → Linear(256→128) + BatchNorm + ReLU + Dropout(0.3)
      → Linear(128→64)  + BatchNorm + ReLU + Dropout(0.3)
      → Linear(64→7)                                       → logits (B, 7)

参数量约 0.5M，与 §4.2.5 的估算一致。Softmax 不放在模型里：训练时由损失函数
处理，推理时显式调用，这样导出的 ONNX 输出就是 logits，便于换损失或换温度。

对设计文档的取舍
----------------
§4.2.6 的总损失是 ``L_CE + 0.003·L_center + 0.01·L_smooth``。这里只实现
**Focal Loss**（§4.2.6 自己给出的、用于解决 Idle 类样本占比过高的 CE 替代方案），
另外两项跳过并说明原因：

* ``L_center`` 需要额外维护每类的可学习中心向量，属于精度调优手段，
  在还没有真实数据、连基线都没跑出来的阶段加它没有意义；
* ``L_smooth`` 约束的是「相邻帧预测标签一致」，但本模型一个窗口只输出一个标签，
  而训练时窗口是打乱的，批内相邻样本在时间上并不相邻，这一项无法正确表达。
  在线推理侧的时序平滑由 §4.2.7 的滑动窗口 + stride 复用上次预测天然承担。

备选的 Transformer Encoder（§4.2.5 后半）没有实现：文档把它定位为
「装配动作依赖 >32 帧长距离时序」时的切换项，属于 Bi-LSTM 精度不达标后的
优化路径，不是第一版必需件。真要切换时按 §4.2.5 的超参加一个类即可。
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import FEATURE_DIM, WINDOW_SIZE
from .fsm import ACTIONS

NUM_CLASSES = len(ACTIONS)          # 7


class TemporalAttention(nn.Module):
    """Scaled Dot-Product 自注意力 + 时间维平均池化（§4.2.5）。"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.w_q = nn.Linear(dim, dim)
        self.w_k = nn.Linear(dim, dim)
        self.w_v = nn.Linear(dim, dim)
        self.scale = math.sqrt(dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """(B, T, D) → (B, D)"""
        query = self.w_q(hidden)
        key = self.w_k(hidden)
        value = self.w_v(hidden)

        scores = torch.matmul(query, key.transpose(1, 2)) / self.scale
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, value)        # (B, T, D)
        return context.mean(dim=1)                    # 加权池化 → (B, D)


class BiLSTMActionClassifier(nn.Module):
    """7 类装配动作分类器。"""

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = NUM_CLASSES,
        window_size: int = WINDOW_SIZE,
        dropout: float = 0.2,
        head_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.config = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_classes": num_classes,
            "window_size": window_size,
            "dropout": dropout,
            "head_dropout": head_dropout,
        }

        # 可学习位置编码，帮助 LSTM 感知帧的绝对位置（§4.2.5）
        self.positional = nn.Parameter(torch.zeros(1, window_size, input_dim))
        nn.init.trunc_normal_(self.positional, std=0.02)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        feature_dim = hidden_dim * 2                  # 双向拼接 → 256
        self.attention = TemporalAttention(feature_dim)

        self.head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """(B, 16, 298) → logits (B, 7)"""
        if windows.dim() != 3:
            raise ValueError(
                f"输入应为 (batch, {WINDOW_SIZE}, {FEATURE_DIM})，实际是 "
                f"{tuple(windows.shape)}"
            )
        hidden, _ = self.lstm(windows + self.positional)
        pooled = self.attention(hidden)
        return self.head(pooled)

    @torch.inference_mode()
    def predict(self, window) -> tuple[int, float]:
        """单个窗口 (16, 298) → (类别下标, 置信度)。numpy 或 tensor 都接受。"""
        self.eval()
        tensor = torch.as_tensor(window, dtype=torch.float32)
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.to(next(self.parameters()).device)

        probs = torch.softmax(self.forward(tensor), dim=-1)[0]
        index = int(torch.argmax(probs))
        return index, float(probs[index])


class FocalLoss(nn.Module):
    """Focal Loss，γ 默认 2.0（§4.2.6）。

    Idle 类样本量可达其他类的 10 倍以上，标准交叉熵会被它主导。
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else torch.tensor([]))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = self.weight if self.weight.numel() else None
        log_probs = F.log_softmax(logits, dim=-1)
        ce = F.nll_loss(log_probs, target, weight=weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


# ------------------------------------------------------------ 存取与导出


def save_checkpoint(
    model: BiLSTMActionClassifier, path: str | Path, meta: dict | None = None
) -> None:
    """保存权重 + 结构超参 + 动作类别表，推理端无需再写死超参。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": model.config,
            "actions": list(ACTIONS),
            "meta": meta or {},
        },
        path,
    )


def load_checkpoint(
    path: str | Path, device: str = "cpu"
) -> tuple[BiLSTMActionClassifier, dict]:
    """读回模型。返回 (模型, meta)。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"找不到动作识别模型 {path}。需要先按 docs/user-manual-zh.md "
            f"第 6 章训练：python scripts/train_action.py"
        )

    payload = torch.load(path, map_location=device, weights_only=False)
    model = BiLSTMActionClassifier(**payload["config"])
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()

    saved_actions = payload.get("actions")
    if saved_actions and list(saved_actions) != list(ACTIONS):
        raise ValueError(
            f"权重里的动作类别表与当前代码不一致。\n"
            f"  权重：{saved_actions}\n  代码：{list(ACTIONS)}\n"
            f"说明 sop/fsm.py 的 ACTIONS 在训练之后被改过，需要重新训练。"
        )
    return model, payload.get("meta", {})


def export_onnx(
    model: BiLSTMActionClassifier, path: str | Path, opset: int = 14
) -> Path:
    """导出 ONNX（§4.2.7）。batch 维动态，输入形状 (batch, 16, 298)。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    dummy = torch.randn(1, model.config["window_size"], model.config["input_dim"])
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=opset,
    )
    return path
