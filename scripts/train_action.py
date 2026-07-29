#!/usr/bin/env python3
"""训练动作识别模型（设计文档 §4.2.6 / §7 Phase 2.2）。

一条命令跑完「特征提取 → 训练 → 评估 → 导出」：

    python scripts/train_action.py --yolo models/pen_parts.pt

它会做四件事：

1. 扫描 ``data/annotations/*.json``（格式见 §7 Phase 1 标注规范），对每段视频跑
   YOLO + MediaPipe 抽 298 维特征，结果缓存到 ``data/cache/``。
   第二次运行时直接读缓存，所以调超参不用重复抽特征（抽一遍很慢）。
2. 按**视频**切 70/15/15 训练/验证/测试集。注意是按视频切不是按窗口切 ——
   相邻窗口共享 14 帧，按窗口切会让验证集泄漏训练集内容，指标虚高。
3. 用 §4.2.6 的超参训练：AdamW、lr 1e-3 + cosine、5 epoch warmup、
   Focal Loss(γ=2)、AMP、早停 patience=15。
4. 评估后存 ``models/action_model.pt`` 并导出 ONNX。

数据增强只实现了 §4.2.6 的三项：关键点抖动、时间缩放、Mixup。
时间裁剪与时间缩放作用重叠；手部平移的效果已被「相对手腕坐标」这组特征抵消；
水平翻转只在相机镜像安装时才对，需要额外开关 —— 这三项因此省略。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sop.features import FEATURE_DIM, STRIDE, WINDOW_SIZE, FeatureExtractor  # noqa: E402
from sop.fsm import ACTIONS                                                  # noqa: E402
from sop.model import (                                                      # noqa: E402
    BiLSTMActionClassifier, FocalLoss, export_onnx, save_checkpoint,
)

ACTION_TO_INDEX = {name: i for i, name in enumerate(ACTIONS)}


# ------------------------------------------------------------- 特征缓存


def build_features(
    annotation_path: Path, video_root: Path, cache_dir: Path, yolo_weights: str
) -> tuple[np.ndarray, np.ndarray] | None:
    """一段视频 → (特征 (T,298), 标签 (T,))。命中缓存则直接读。"""
    spec = json.loads(annotation_path.read_text(encoding="utf-8"))
    video_path = video_root / spec["video"]
    cache_path = cache_dir / f"{annotation_path.stem}.npz"

    if cache_path.is_file() and cache_path.stat().st_mtime > annotation_path.stat().st_mtime:
        cached = np.load(cache_path)
        return cached["features"], cached["labels"]

    if not video_path.is_file():
        print(f"  跳过 {annotation_path.name}：找不到视频 {video_path}")
        return None

    import cv2

    from sop.perception import Perception, preprocess

    labels = _labels_from_annotations(spec)
    capture = cv2.VideoCapture(str(video_path))
    extractor = FeatureExtractor()
    rows: list[np.ndarray] = []

    with Perception(yolo_weights=yolo_weights) as perception:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            hand, parts = perception(preprocess(frame))
            rows.append(extractor.extract(hand, parts))
    capture.release()

    features = np.stack(rows).astype(np.float32) if rows else np.zeros((0, FEATURE_DIM), np.float32)
    length = min(len(features), len(labels))
    features, labels = features[:length], labels[:length]

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, features=features, labels=labels)
    print(f"  {annotation_path.name}: {length} 帧 → {cache_path.name}")
    return features, labels


def _labels_from_annotations(spec: dict) -> np.ndarray:
    """把 frame_range 区间标注展开成逐帧标签。未覆盖的帧记为 Idle。"""
    total = int(spec.get("total_frames", 0))
    segments = spec.get("annotations", [])
    if not total and segments:
        total = max(int(s["frame_range"][1]) for s in segments) + 1

    labels = np.full(total, ACTION_TO_INDEX["Idle"], dtype=np.int64)
    for segment in segments:
        action = segment["action"]
        if action not in ACTION_TO_INDEX:
            raise ValueError(
                f"标注里的动作 '{action}' 不是 7 类之一，可选值：{'、'.join(ACTIONS)}"
            )
        start, end = segment["frame_range"]
        labels[int(start):int(end) + 1] = ACTION_TO_INDEX[action]
    return labels


# ----------------------------------------------------------------- 数据集


class ActionWindowDataset(Dataset):
    """按 16 帧滑窗切样本。窗口标签取窗内出现最多的动作。

    只存整段特征序列，窗口在取样时切出来 —— 相邻窗口重叠 14 帧，
    预先展开会把内存放大 8 倍。
    """

    def __init__(
        self,
        sequences: list[np.ndarray],
        labels: list[np.ndarray],
        window_size: int = WINDOW_SIZE,
        stride: int = STRIDE,
        augment: bool = False,
        jitter_std: float = 0.005,
    ) -> None:
        self.sequences = sequences
        self.labels = labels
        self.window_size = window_size
        self.augment = augment
        self.jitter_std = jitter_std

        self.index: list[tuple[int, int]] = []
        for seq_id, sequence in enumerate(sequences):
            limit = len(sequence) - window_size + 1
            for start in range(0, max(0, limit), stride):
                self.index.append((seq_id, start))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int):
        seq_id, start = self.index[item]
        window = self.sequences[seq_id][start:start + self.window_size].copy()
        span = self.labels[seq_id][start:start + self.window_size]
        label = int(Counter(span.tolist()).most_common(1)[0][0])

        if self.augment:
            window = _augment(window, self.jitter_std)

        return torch.from_numpy(window), label

    def class_counts(self) -> Counter:
        counter: Counter = Counter()
        for seq_id, start in self.index:
            span = self.labels[seq_id][start:start + self.window_size]
            counter[int(Counter(span.tolist()).most_common(1)[0][0])] += 1
        return counter


def _augment(window: np.ndarray, jitter_std: float) -> np.ndarray:
    """时间缩放 [0.8, 1.25] + 关键点抖动（§4.2.6）。"""
    length = len(window)

    speed = np.random.uniform(0.8, 1.25)
    source = np.clip(np.arange(length) * speed, 0, length - 1)
    lower = np.floor(source).astype(int)
    upper = np.minimum(lower + 1, length - 1)
    weight = (source - lower)[:, None].astype(np.float32)
    window = window[lower] * (1.0 - weight) + window[upper] * weight

    window = window + np.random.normal(0.0, jitter_std, window.shape).astype(np.float32)
    return window.astype(np.float32)


def load_dataset(args: argparse.Namespace):
    """扫描标注目录，返回按视频划分好的三个数据集。"""
    annotation_dir = ROOT / args.annotations
    files = sorted(annotation_dir.glob("*.json"))
    if not files:
        raise SystemExit(
            f"\n{annotation_dir} 里没有标注文件。\n"
            f"需要先按 docs/user-manual-zh.md 第 3、4 章采集视频并标注动作。\n"
            f"每段视频一个 JSON，格式见手册 4.2 节。\n"
        )

    print(f"提取特征（{len(files)} 段视频，缓存目录 {args.cache}）：")
    sequences: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for path in files:
        result = build_features(
            path, ROOT / args.videos, ROOT / args.cache, args.yolo
        )
        if result is None or len(result[0]) < WINDOW_SIZE:
            continue
        sequences.append(result[0])
        labels.append(result[1])

    if not sequences:
        raise SystemExit("没有一段视频成功提取到特征，检查视频路径与标注是否匹配。")

    # 按视频划分，避免相邻窗口跨集合泄漏
    order = np.random.permutation(len(sequences))
    n_train = max(1, int(len(order) * 0.70))
    n_val = max(1, int(len(order) * 0.15)) if len(order) > 2 else 0
    groups = {
        "train": order[:n_train],
        "val": order[n_train:n_train + n_val],
        "test": order[n_train + n_val:],
    }

    datasets = {}
    for name, ids in groups.items():
        datasets[name] = ActionWindowDataset(
            [sequences[i] for i in ids], [labels[i] for i in ids],
            augment=(name == "train"),
        )
        print(f"  {name:5} {len(ids):3} 段视频，{len(datasets[name]):6} 个窗口")

    return datasets


# ------------------------------------------------------------------- 训练


def train(args: argparse.Namespace) -> int:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    datasets = load_dataset(args)
    if not len(datasets["train"]):
        raise SystemExit("训练集为空，视频太短（每段至少要 16 帧）。")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n设备：{device}")

    counts = datasets["train"].class_counts()
    print("训练集类别分布：")
    for index, name in enumerate(ACTIONS):
        print(f"  {name:<7} {counts.get(index, 0):6}")

    loaders = {
        name: DataLoader(
            dataset, batch_size=args.batch_size,
            shuffle=(name == "train"), num_workers=args.workers,
            drop_last=(name == "train" and len(dataset) > args.batch_size),
        )
        for name, dataset in datasets.items() if len(dataset)
    }

    model = BiLSTMActionClassifier().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量：{total_params / 1e6:.2f}M")

    criterion = FocalLoss(gamma=args.gamma).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_loss = float("inf")
    best_state = None
    patience_left = args.patience

    print(f"\n开始训练：{args.epochs} epoch，batch {args.batch_size}\n")
    for epoch in range(1, args.epochs + 1):
        # 前 5 个 epoch 线性 warmup（§4.2.6）
        if epoch <= args.warmup:
            for group in optimizer.param_groups:
                group["lr"] = args.lr * epoch / args.warmup

        model.train()
        running = 0.0
        for windows, targets in loaders["train"]:
            windows, targets = windows.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)

            if args.mixup > 0.0 and np.random.rand() < 0.5:
                loss = _mixup_step(model, criterion, windows, targets, args.mixup, use_amp)
            else:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss = criterion(model(windows), targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss) * len(windows)

        if epoch > args.warmup:
            scheduler.step()

        train_loss = running / max(1, len(datasets["train"]))
        line = f"epoch {epoch:3}/{args.epochs}  train_loss {train_loss:.4f}"

        if "val" in loaders:
            val_loss, val_acc = evaluate(model, loaders["val"], criterion, device)
            line += f"  val_loss {val_loss:.4f}  val_acc {val_acc * 100:.2f}%"
            improved = val_loss < best_loss - 1e-4
        else:
            val_loss, improved = train_loss, train_loss < best_loss - 1e-4

        if improved:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
            line += "  ← best"
        else:
            patience_left -= 1

        print(line)
        if patience_left <= 0:
            print(f"\n验证损失连续 {args.patience} 个 epoch 没改善，早停。")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # --- 评估 ---
    report_split = "test" if "test" in loaders else ("val" if "val" in loaders else "train")
    print(f"\n在 {report_split} 集上评估：")
    accuracy, recalls, matrix = detailed_report(model, loaders[report_split], device)

    print(f"\n  总体准确率 {accuracy * 100:.2f}%"
          f"   （§7 Phase 2 验收线 ≥ 92%）")
    print("\n  各类召回率（验收线 > 88%）：")
    for index, name in enumerate(ACTIONS):
        value = recalls[index]
        flag = " " if value is None else ("✓" if value > 0.88 else "✗")
        shown = "  无样本" if value is None else f"{value * 100:6.2f}%"
        print(f"    {flag} {name:<7} {shown}")

    print("\n  混淆矩阵（行=真实，列=预测）：")
    header = "         " + "".join(f"{n[:4]:>6}" for n in ACTIONS)
    print(header)
    for index, name in enumerate(ACTIONS):
        print(f"    {name[:5]:<6}" + "".join(f"{int(v):>6}" for v in matrix[index]))

    # --- 保存 ---
    weights_path = ROOT / args.out
    save_checkpoint(model, weights_path, meta={
        "accuracy": round(accuracy, 4),
        "split": report_split,
        "epochs_run": epoch,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    print(f"\n权重已存：{weights_path}")

    if not args.no_onnx:
        try:
            onnx_path = export_onnx(model.cpu(), weights_path.with_suffix(".onnx"))
            print(f"ONNX 已存：{onnx_path}")
        except Exception as exc:                     # noqa: BLE001
            print(f"ONNX 导出失败（不影响使用）：{exc}")

    print(f"\n下一步：python run.py --video 你的视频.mp4 --web")
    return 0 if accuracy >= 0.92 else 1


def _mixup_step(model, criterion, windows, targets, alpha, use_amp):
    """Mixup α=0.2（§4.2.6）。两个样本线性插值，损失按同比例混合。"""
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(len(windows), device=windows.device)
    mixed = lam * windows + (1.0 - lam) * windows[perm]
    with torch.amp.autocast("cuda", enabled=use_amp):
        logits = model(mixed)
        return lam * criterion(logits, targets) + (1.0 - lam) * criterion(logits, targets[perm])


@torch.inference_mode()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for windows, targets in loader:
        windows, targets = windows.to(device), targets.to(device)
        logits = model(windows)
        total_loss += float(criterion(logits, targets)) * len(windows)
        correct += int((logits.argmax(dim=-1) == targets).sum())
        seen += len(windows)
    return total_loss / max(1, seen), correct / max(1, seen)


@torch.inference_mode()
def detailed_report(model, loader, device):
    """返回 (准确率, 每类召回率, 混淆矩阵)。"""
    model.eval()
    size = len(ACTIONS)
    matrix = np.zeros((size, size), dtype=np.int64)

    for windows, targets in loader:
        predictions = model(windows.to(device)).argmax(dim=-1).cpu().numpy()
        for truth, guess in zip(targets.numpy(), predictions):
            matrix[int(truth), int(guess)] += 1

    accuracy = float(np.trace(matrix) / max(1, matrix.sum()))
    recalls: list[float | None] = []
    for index in range(size):
        support = matrix[index].sum()
        recalls.append(float(matrix[index, index] / support) if support else None)
    return accuracy, recalls, matrix


# -------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="训练动作识别模型（Bi-LSTM）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--yolo", default="models/pen_parts.pt",
                        help="零件检测权重，抽特征要用（先完成手册第 5 章）")
    parser.add_argument("--annotations", default="data/annotations", help="动作标注 JSON 目录")
    parser.add_argument("--videos", default="data/videos", help="视频目录")
    parser.add_argument("--cache", default="data/cache", help="特征缓存目录")
    parser.add_argument("--out", default="models/action_model.pt", help="权重输出路径")

    # 超参默认值全部来自 §4.2.6
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4, dest="weight_decay")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal Loss 的 γ")
    parser.add_argument("--mixup", type=float, default=0.2, help="Mixup 的 α，0 表示关闭")

    parser.add_argument("--device", default=None, help="cuda / cpu，默认自动")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-onnx", action="store_true", dest="no_onnx")

    return train(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
