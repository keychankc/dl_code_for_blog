"""LightningModule：封装训练 / 验证等逻辑。

这个模块的职责：
- 定义前向传播（forward）
- 定义训练步骤（training_step）
- 定义验证步骤（validation_step）
- 定义优化器（configure_optimizers）

注意：这里不关心 DataLoader，也不关心配置系统。
"""

import pytorch_lightning as L
import torch.nn.functional as F
import torch.optim as optim


class LitClassifier(L.LightningModule):
    """将 nn.Module 包装成 LightningModule，托管训练细节。"""

    def __init__(self, model, lr: float):
        super().__init__()

        # 真实的 PyTorch 模型（SimpleCNN）
        self.model = model
        # 学习率
        self.lr = lr

        # 将超参数保存到 checkpoint 中，方便恢复和可视化
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x):
        """推理 / 前向过程，直接调用内部模型。"""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """单个训练 step 的逻辑。

        Args:
            batch: 一个 batch 的数据，包含 (x, y)。
            batch_idx: 当前 batch 的索引。

        Returns:
            loss 张量，Lightning 会自动帮你做 backward。
        """
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # self.log 会自动将指标记录到 logger（TensorBoard / W&B 等）
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """单个验证 step 的逻辑。"""
        x, y = batch
        logits = self(x)
        val_loss = F.cross_entropy(logits, y)
        self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        """配置优化器（和可选的学习率调度器）。"""
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
