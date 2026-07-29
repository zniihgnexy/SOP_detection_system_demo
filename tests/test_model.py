"""动作识别模型测试（sop/model.py）。

需要 torch。没装 torch 的机器上整个文件会被跳过 —— 判定层的测试不受影响。
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="未安装 torch，跳过模型测试")

from sop.features import FEATURE_DIM, WINDOW_SIZE      # noqa: E402
from sop.fsm import ACTIONS                            # noqa: E402
from sop.model import (                                # noqa: E402
    BiLSTMActionClassifier, FocalLoss, export_onnx, load_checkpoint, save_checkpoint,
)


@pytest.fixture()
def model():
    torch.manual_seed(0)
    return BiLSTMActionClassifier()


def test_forward_shape(model):
    logits = model(torch.randn(5, WINDOW_SIZE, FEATURE_DIM))
    assert logits.shape == (5, len(ACTIONS))


def test_num_classes_follows_action_table(model):
    assert model.config["num_classes"] == 7 == len(ACTIONS)


def test_parameter_count_near_design_estimate(model):
    """§4.2.5 估算约 0.5M 参数，允许一个量级内的偏差。"""
    total = sum(p.numel() for p in model.parameters())
    assert 300_000 < total < 900_000, total


def test_rejects_wrong_input_rank(model):
    with pytest.raises(ValueError, match="batch"):
        model(torch.randn(WINDOW_SIZE, FEATURE_DIM))


def test_predict_works_with_single_window(model):
    """batch=1 时 BatchNorm 只有在 eval 模式下才能跑，predict 必须自己切换。"""
    index, confidence = model.predict(torch.randn(WINDOW_SIZE, FEATURE_DIM))
    assert 0 <= index < len(ACTIONS)
    assert 0.0 <= confidence <= 1.0


def test_predict_accepts_numpy(model):
    numpy = pytest.importorskip("numpy")
    window = numpy.zeros((WINDOW_SIZE, FEATURE_DIM), dtype="float32")
    index, confidence = model.predict(window)
    assert isinstance(index, int) and isinstance(confidence, float)


def test_probabilities_sum_to_one(model):
    model.eval()
    with torch.inference_mode():
        probs = torch.softmax(model(torch.randn(3, WINDOW_SIZE, FEATURE_DIM)), dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(3), atol=1e-5)


# --------------------------------------------------------------- 存取


def test_checkpoint_roundtrip(tmp_path, model):
    path = tmp_path / "models" / "action.pt"
    save_checkpoint(model, path, meta={"accuracy": 0.93})
    assert path.is_file()

    restored, meta = load_checkpoint(path)
    assert meta["accuracy"] == 0.93
    assert restored.config == model.config

    window = torch.randn(1, WINDOW_SIZE, FEATURE_DIM)
    model.eval()
    with torch.inference_mode():
        assert torch.allclose(model(window), restored(window), atol=1e-5)


def test_load_checkpoint_missing_file_points_at_manual(tmp_path):
    with pytest.raises(FileNotFoundError, match="第 6 章"):
        load_checkpoint(tmp_path / "nope.pt")


def test_load_checkpoint_detects_changed_action_table(tmp_path, model):
    path = tmp_path / "stale.pt"
    save_checkpoint(model, path)

    payload = torch.load(path, weights_only=False)
    payload["actions"] = ["Pick", "Insert"]          # 模拟训练后改了 ACTIONS
    torch.save(payload, path)

    with pytest.raises(ValueError, match="重新训练"):
        load_checkpoint(path)


# ----------------------------------------------------------- Focal Loss


def test_focal_loss_with_gamma_zero_equals_cross_entropy():
    logits = torch.randn(8, len(ACTIONS))
    target = torch.randint(0, len(ACTIONS), (8,))

    focal = FocalLoss(gamma=0.0)(logits, target)
    cross = torch.nn.functional.cross_entropy(logits, target)
    assert focal == pytest.approx(float(cross), abs=1e-5)


def test_focal_loss_downweights_easy_samples():
    """γ=2 时，已经预测得很准的样本贡献应远小于标准交叉熵。"""
    confident = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    target = torch.tensor([0])

    focal = float(FocalLoss(gamma=2.0)(confident, target))
    cross = float(torch.nn.functional.cross_entropy(confident, target))
    assert focal < cross


def test_focal_loss_is_positive_and_finite():
    loss = FocalLoss()(torch.randn(16, len(ACTIONS)),
                       torch.randint(0, len(ACTIONS), (16,)))
    assert torch.isfinite(loss) and loss > 0


# --------------------------------------------------------------- ONNX


def test_export_onnx(tmp_path, model):
    pytest.importorskip("onnx", reason="未安装 onnx，跳过导出测试")
    path = export_onnx(model, tmp_path / "action_model.onnx")
    assert path.is_file() and path.stat().st_size > 0
