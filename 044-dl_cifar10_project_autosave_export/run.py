"""项目入口。

职责：
- 读取 Hydra 配置
- 构建 DataLoader / 模型 / Trainer
- 启动训练
- 自动保存 best.ckpt，并在训练结束后自动导出 model.pt（TorchScript）
"""

import os

import hydra
from omegaconf import DictConfig

from src.data import build_dataloader
from src.train import build_trainer, build_model
from src.utils import seed_everything
from deploy.export import export_torchscript


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # 0. 设定随机种子，保证实验可复现
    seed_everything(42)

    # 1. 数据：只负责构建 DataLoader
    train_loader, val_loader = build_dataloader(
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        data_dir=cfg.data.data_dir,
    )

    # 2. 模型：nn.Module + LightningModule
    lit_model = build_model(cfg)

    # 3. Trainer：Lightning 接管训练循环 / 日志 / AMP 等
    trainer = build_trainer(cfg)

    # 4. 启动训练
    trainer.fit(lit_model, train_loader, val_loader)

    # 5. 训练结束后：自动从 best.ckpt 导出 TorchScript 模型 model.pt
    best_ckpt_path = os.path.join("checkpoints", "best.ckpt")
    if os.path.exists(best_ckpt_path):
        print(f"[INFO] 训练完成，发现 best.ckpt: {best_ckpt_path}")
        print("[INFO] 正在从 best.ckpt 导出 TorchScript 模型为 model.pt ...")
        export_torchscript(
            checkpoint_path=best_ckpt_path,
            out="model.pt",
            num_classes=cfg.model.num_classes,
        )
        print("[INFO] 导出完成，可以直接使用 deploy/inference.py 或 deploy/api.py。")
    else:
        print("[WARN] 未找到 best.ckpt，跳过自动导出 model.pt。预期路径: checkpoints/best.ckpt")


if __name__ == "__main__":
    main()
