import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time

from utils import get_time_dif


def evaluate(config, model, data_iter, _test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if _test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding'):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)  # 适用于 sigmoid/tanh/RNN 网络，保持输入和输出的方差一致
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)  # 适用于 ReLU 及其变种，避免梯度消失
                else:
                    nn.init.normal_(w)  # 一般情况，但不如 Xavier 或 Kaiming 稳定
            elif 'bias' in name:
                nn.init.constant_(w, 0)  # 偏置一般不需要复杂初始化，设为 0 即可

def _eval_result(config, model, test_iter):
    model.load_state_dict(torch.load(model.save_path, weights_only=True))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, _test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

def train(config, model, train_iter, dev_iter, test_iter, writer):
    """
    :param config: 参数配置对象
    :param model: 深度学习模型
    :param train_iter: 训练数据集的迭代器
    :param dev_iter: 验证数据集的迭代器
    :param test_iter: 测试数据集的迭代器
    :param writer: TensorBoard 记录器
    :return:
    """
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate)

    # 用 ReduceLROnPlateau 替代 ExponentialLR
    # 验证集损失不下降时才调整学习率，避免不必要的衰减
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    total_batch = 0  # 记录进行到多少 batch
    dev_best_loss = float('inf')  # 记录最优验证 loss
    last_improve = 0  # 记录上次验证集 loss 下降的 batch 数
    flag = False  # 记录是否长时间未提升
    train_loss_sum, train_acc_sum, batch_count = 0, 0, 0  # 累积 loss 和 acc 计算整个 epoch 的平均值

    for epoch in range(model.num_epochs):
        print(f'Epoch [{epoch + 1}/{model.num_epochs}]')

        # 训练
        for _, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)

            optimizer.zero_grad()  # 梯度清空，防止累计导致的梯度混合

            loss = F.cross_entropy(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播 计算损失相对于模型参数的梯度
            optimizer.step()  # 先优化参数

            # 计算 batch 级别的训练准确率
            # .cpu()：将数据从 GPU 移动到 CPU（用于计算准确率）
            labels_cpu = labels.data.cpu()
            predict = torch.max(outputs.data, 1)[1].cpu()  # 预测类别
            train_acc = metrics.accuracy_score(labels_cpu, predict)  # 计算准确率

            # 累计 loss 和 acc 计算整个 epoch 的平均值
            train_loss_sum += loss.item()
            train_acc_sum += train_acc
            batch_count += 1

            if total_batch % 100 == 0:  # 100 batch 进行一次验证
                dev_acc, dev_loss = evaluate(config, model, dev_iter)

                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), model.save_path)
                    improve = '*'  # 记录模型有提升
                    last_improve = total_batch
                else:
                    improve = ''

                time_dif = get_time_dif(start_time)
                msg = ('Iter: {0:>6},  Train Loss: {1:>5.2f},  Train Acc: {2:>6.2%},  '
                       'Val Loss: {3:>5.2f},  Val Acc: {4:>6.2%},  Time: {5} {6}')
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))

                # 记录 loss 和 acc 到 TensorBoard
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)

                model.train()

                # 调整学习率 (基于 dev_loss)
                scheduler.step(dev_loss)  # ReduceLROnPlateau 需要 loss 作为输入

            total_batch += 1

            # 如果 long time no improvement，则 early stop
            if total_batch - last_improve > model.require_improvement:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break

        if flag:
            break

        # 计算整个 epoch 平均训练 loss 和 acc
        avg_train_loss = train_loss_sum / batch_count
        avg_train_acc = train_acc_sum / batch_count
        print(f"Epoch [{epoch + 1}/{model.num_epochs}] - Avg Train Loss: {avg_train_loss:.4f},"
              f" Avg Train Acc: {avg_train_acc:.4%}")

        writer.add_scalar("epoch_loss/train", avg_train_loss, epoch)
        writer.add_scalar("epoch_acc/train", avg_train_acc, epoch)

    writer.close()
    _eval_result(config, model, test_iter)
