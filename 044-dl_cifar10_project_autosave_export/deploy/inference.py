"""本地推理脚本：使用导出的 TorchScript 模型对单张图片进行预测。

示例使用方式：
    python deploy/inference.py --image path/to/image.jpg --model_path model.pt
"""

import argparse

import torch
from PIL import Image
from torchvision import transforms


def load_model(model_path: str = "model.pt") -> torch.jit.ScriptModule:
    """加载 TorchScript 模型。"""

    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()
    return model


def build_transform():
    """构建与训练时大致一致的预处理流程。

    注意：这里的预处理需要与你训练数据的分布相匹配，
    这里只是做一个简单示例（Resize + ToTensor）。
    """

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    return transform


def preprocess(img_path: str) -> torch.Tensor:
    """加载图片并做预处理，返回模型输入张量。"""

    transform = build_transform()
    img = Image.open(img_path).convert("RGB")
    x = transform(img)       # [3, 32, 32]
    x = x.unsqueeze(0)       # [1, 3, 32, 32] -> 增加 batch 维度
    return x


def predict(img_path: str, model_path: str = "model.pt") -> int:
    """对单张图片进行预测，并返回预测的类别 index。"""

    model = load_model(model_path)
    x = preprocess(img_path)

    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1).item()

    print(f"[RESULT] 图片 {img_path} 的预测类别 index 为: {pred}")
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="待预测的图片路径（例如一张 CIFAR-10 类似的 32x32 彩色图像）",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="model.pt",
        help="TorchScript 模型路径，默认是当前目录下的 model.pt",
    )
    args = parser.parse_args()

    predict(args.image, args.model_path)


if __name__ == "__main__":
    main()
