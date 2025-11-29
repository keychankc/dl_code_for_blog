# CIFAR-10 Lightning + Hydra + 自动保存 & 自动导出部署工程

## 1. 目录结构

```text
project/
│── configs/
│   └── config.yaml           # 模型 / 数据 / 训练 配置
│
│── src/
│   ├── data.py               # 数据管线：Dataset & DataLoader
│   ├── model.py              # 模型定义：SimpleCNN（含中文注释）
│   ├── lit_module.py         # LightningModule：训练/验证逻辑
│   ├── train.py              # 组装 Trainer & 模型（含 ModelCheckpoint）
│   ├── utils.py              # 辅助工具（如随机种子）
│   └── __init__.py
│
│── deploy/
│   ├── export.py             # 导出 ONNX / TorchScript 模型（可单独使用）
│   ├── inference.py          # 本地单张图片推理脚本
│   └── api.py                # FastAPI 在线推理服务
│
│── run.py                    # 项目入口（训练 + 自动导出 model.pt）
│── requirements.txt
│── README.md
```

## 2. 环境准备

建议使用虚拟环境（conda / venv）：

```bash
pip install -r requirements.txt
```

> ⚠️ 如果你需要 GPU / MPS，加 `torch` / `torchvision` 时，请根据自己的环境从 PyTorch 官网复制命令安装。

## 3. 启动训练（自动保存 best.ckpt + 自动导出 model.pt）

```bash
python run.py
```

训练结束后，你会在：

- `checkpoints/best.ckpt`     —— Lightning 的最佳 checkpoint
- `model.pt`                  —— 从 best.ckpt 自动导出的 TorchScript 模型

这两个文件就是后续部署和推理的基础。

## 4. 修改超参数（无需改 Python 代码）

例如修改 batch size / 学习率 / 训练轮数：

```bash
python run.py data.batch_size=128 train.lr=5e-4 train.max_epochs=20
```

## 5. 单独导出 ONNX / 重新导出 TorchScript（可选）

如果你之后改了模型或想导出 ONNX，可以运行：

```bash
python deploy/export.py
```

默认会从 `checkpoints/best.ckpt` 读取模型，并导出：

- `model.onnx`
- `model.pt`

你也可以手动指定 checkpoint 路径（用环境变量）：

```bash
CKPT_PATH=some_other.ckpt python deploy/export.py
```

## 6. 本地单张图片推理

在确保当前目录下已有 `model.pt` 的前提下：

```bash
python deploy/inference.py --image path/to/your_image.jpg --model_path model.pt
```

程序会打印类似：

```text
[RESULT] 图片 xxx 的预测类别 index 为: 3
```

> 注：当前预处理假设你的图片大致与 CIFAR-10 类似（32x32、RGB 小图），
> 真实场景建议根据需要调整 `deploy/inference.py` 中的 transforms。

## 7. 启动 FastAPI 在线服务

确保项目根目录下有 `model.pt`（训练完会自动生成，也可以手动导出），然后：

```bash
uvicorn deploy.api:app --host 0.0.0.0 --port 8000 --reload
```

测试接口：

```bash
curl -X POST -F "file=@path/to/your_image.jpg" http://localhost:8000/predict
```

会得到类似响应：

```json
{"prediction": 3}
```

你也可以在浏览器打开：

- `http://localhost:8000/docs`  查看自动生成的 Swagger API 文档

---