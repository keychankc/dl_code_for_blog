"""模型导出脚本：将训练好的模型导出为 ONNX / TorchScript。

可以单独运行这个脚本进行导出；
也可以在训练结束后由 run.py 自动调用 `export_torchscript` 完成导出。
"""

import os
from typing import Dict

import torch

from src.model import SimpleCNN


def load_state_dict_from_lightning_ckpt(ckpt_path: str) -> Dict[str, torch.Tensor]:
    """从 Lightning checkpoint 中提取纯模型的 state_dict。

    Lightning 的 checkpoint 里通常有一个 `state_dict` 字段，
    key 形式类似 `model.conv.0.weight`，其中前缀 `model.` 是
    LightningModule 中的属性名，我们需要去掉这个前缀，
    才能加载到原始的 nn.Module 中。
    """

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]

    new_state_dict = {}
    for k, v in state_dict.items():
        # 去掉开头的 "model." 前缀（对应 LitClassifier.model）
        if k.startswith("model."):
            new_key = k[len("model.") :]
        else:
            new_key = k
        new_state_dict[new_key] = v

    return new_state_dict


def export_onnx(
    checkpoint_path: str,
    out: str = "model.onnx",
    num_classes: int = 10,
):
    """导出 ONNX 模型。

    Args:
        checkpoint_path: 训练生成的 Lightning checkpoint 路径。
        out: 导出 ONNX 的文件名。
        num_classes: 分类类别数（需与训练时一致）。
    """

    model = SimpleCNN(num_classes=num_classes)
    state_dict = load_state_dict_from_lightning_ckpt(checkpoint_path)
    model.load_state_dict(state_dict)
    model.eval()

    # 创建一个假的输入（dummy）用于跟踪模型的计算图
    dummy = torch.randn(1, 3, 32, 32)

    torch.onnx.export(
        model,
        dummy,
        out,
        input_names=["input"],
        output_names=["logits"],
        opset_version=11,
    )

    print(f"[OK] ONNX 导出成功: {out}")


def export_torchscript(
    checkpoint_path: str,
    out: str = "model.pt",
    num_classes: int = 10,
):
    """导出 TorchScript 模型（使用 trace 方式）。

    Args:
        checkpoint_path: 训练生成的 Lightning checkpoint 路径。
        out: 导出 TorchScript 的文件名。
        num_classes: 分类类别数（需与训练时一致）。
    """

    model = SimpleCNN(num_classes=num_classes)
    state_dict = load_state_dict_from_lightning_ckpt(checkpoint_path)
    model.load_state_dict(state_dict)
    model.eval()

    example_input = torch.randn(1, 3, 32, 32)
    traced = torch.jit.trace(model, example_input)
    traced.save(out)

    print(f"[OK] TorchScript 导出成功: {out}")


if __name__ == "__main__":
    # 手动导出的简单示例使用方式：
    #   python deploy/export.py
    ckpt_path = os.environ.get("CKPT_PATH", "checkpoints/best.ckpt")

    if not os.path.exists(ckpt_path):
        print("[WARN] 未找到 checkpoint 文件，可以通过环境变量 CKPT_PATH 指定路径。")
        print("       当前默认尝试路径: checkpoints/best.ckpt")
    else:
        export_onnx(ckpt_path, "model.onnx")
        export_torchscript(ckpt_path, "model.pt")
