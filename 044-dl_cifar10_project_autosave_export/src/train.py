"""训练相关的组装逻辑。

职责：
- 根据配置构建 Trainer（内置自动保存 best.ckpt）
- 根据配置构建模型（并可选开启 torch.compile）
"""

import torch
import pytorch_lightning as L

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from .model import SimpleCNN
from .lit_module import LitClassifier


def build_trainer(cfg) -> L.Trainer:
    """根据配置构建 Lightning Trainer。

    - 包含 AMP（precision）
    - 包含多设备控制（accelerator / devices）
    - 可选 W&B 日志记录
    - 自动保存 val_loss 最小的 best.ckpt 到 ./checkpoints/best.ckpt
    """

    logger = None
    if cfg.train.use_wandb:
        # 使用 W&B 记录实验（需要提前 `wandb login`）
        logger = WandbLogger(project=cfg.train.wandb_project)

    # accelerator=auto 时，这里做一个简单判断
    if cfg.train.accelerator == "auto":
        if torch.cuda.is_available():
            accelerator = "gpu"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            accelerator = "mps"
        else:
            accelerator = "cpu"
    else:
        accelerator = cfg.train.accelerator

    # ModelCheckpoint：自动保存最好的模型
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",      # 保存目录
        filename="best",            # 文件名 -> best.ckpt
        save_top_k=1,               # 只保留一个最好的
        monitor="val_loss",         # 根据验证集上的 val_loss 选择最优
        mode="min",                 # val_loss 越小越好
        save_last=True,             # 额外保存 last.ckpt（最后一轮）
    )

    trainer = L.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=accelerator,
        devices=cfg.train.devices,
        precision=cfg.train.precision,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    return trainer


def build_model(cfg) -> L.LightningModule:
    """根据配置构建模型，并封装为 LightningModule。

    - 创建 SimpleCNN
    - 可选地对模型进行 torch.compile 加速
    - 封装进 LitClassifier，附带学习率等超参数
    """

    model = SimpleCNN(num_classes=cfg.model.num_classes)

    # 可选的 torch.compile（PyTorch 2.0+）
    if getattr(cfg.train, "compile", False):
        try:
            from torch import compile as torch_compile

            model = torch_compile(model)
            print("[INFO] 使用 torch.compile 对模型进行了加速编译。")
        except Exception as e:
            print("[WARN] torch.compile 不可用或失败，使用原始模型。", e)

    lit_model = LitClassifier(model, lr=cfg.train.lr)
    return lit_model
