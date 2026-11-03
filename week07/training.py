# -*- coding: utf-8 -*-
"""
training.py —— DataLoader、完整评分的 MSE、训练循环与最佳轮次

这里沿用第五周的"清梯度 → 前向 → 损失 → 反向 → 更新"流程，
只增加两点本周特有的要求：

1. 每轮结束后在【评价模式 + 关闭梯度】下重新评价完整训练集和验证集，
   MSE 按"全部目标的误差平方和 ÷ 目标总数"计算，而不是各批次的平均再平均。
2. 保存验证 MSE 最小那一轮的参数副本；完全相同则保留较早轮次。

为什么不用最后 40 轮的参数：
    训练轮数越多，训练误差几乎一定下降，但验证误差可能先降后升。
    用"验证集上最好的一轮"才是可比较的选择方式。
"""
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

BATCH_SIZE = 128
EPOCHS = 40
LEARNING_RATE = 1e-3
SEED = 42


def make_loader(X, y, batch_size=BATCH_SIZE, shuffle=False, seed=SEED):
    """
    把 numpy 数组包成 DataLoader。

    参数
    ----
    X, y    : (N, 24, 1) 和 (N, 1)，float32
    shuffle : 训练集打乱样本顺序；打乱的是"整条窗口"，窗口内部 24 步顺序不变
    seed    : 组会要求每组 DataLoader 各建一个种子为 42 的生成器，
              这样两种隐藏维数看到的样本顺序完全一致，比较才公平
    """
    X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
    y_t = torch.from_numpy(np.asarray(y, dtype=np.float32))
    dataset = TensorDataset(X_t, y_t)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,      # 最后一批不足 128 也保留
        num_workers=0,        # Windows 上 0 最稳
        generator=generator,
    )


def dataset_mse(model, loader):
    """
    在评价模式下计算整个数据集的 MSE。

    专门按"误差平方和 ÷ 目标总数"来算：
    如果先算每批的平均、再把批平均求平均，最后一批样本少时会权重偏大。
    """
    model.eval()                       # 切换到评价模式
    total_sq = 0.0
    total_n = 0
    with torch.no_grad():              # 评价不需要计算图
        for xb, yb in loader:
            pred = model(xb)
            total_sq += float(((pred - yb) ** 2).sum())
            total_n += int(yb.numel())
    return total_sq / total_n


def train_lstm(model, train_loader, train_eval_loader, val_loader,
               epochs=EPOCHS, lr=LEARNING_RATE, log_every=10):
    """
    训练一个 LSTM，返回 (history, best)。

    history : [{epoch, train_mse, val_mse}, ...]
    best    : {epoch, val_mse, state_dict}
              state_dict 是 copy.deepcopy 的独立副本，
              否则后面继续训练会把这份参数改掉。

    训练循环五步（与第五周一致）：
      optimizer.zero_grad()  清掉上批梯度
      pred = model(x)        前向
      loss = criterion(...)  算损失
      loss.backward()        反向，梯度沿时间展开累积
      optimizer.step()       更新参数
    """
    criterion = nn.MSELoss()           # 训练时用批内平均 MSE
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    history = []
    best = {"epoch": 0, "val_mse": float("inf"), "state_dict": None}

    for epoch in range(1, epochs + 1):
        model.train()                  # 训练模式
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        # 每轮结束：用完整训练集、完整验证集重新评分（评价模式 + no_grad）
        train_mse = dataset_mse(model, train_eval_loader)
        val_mse = dataset_mse(model, val_loader)
        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse})

        if log_every and (epoch % log_every == 0 or epoch == 1):
            print(f"    epoch {epoch:>2}/{epochs}  训练 MSE {train_mse:.5f}"
                  f"  验证 MSE {val_mse:.5f}")

        # 只在【严格更小】时更新，完全相同时保留更早的那一轮
        if val_mse < best["val_mse"]:
            best = {
                "epoch": epoch,
                "val_mse": val_mse,
                "state_dict": copy.deepcopy(model.state_dict()),
            }

    # 训练结束后恢复到最佳轮次的参数
    model.load_state_dict(best["state_dict"])
    return history, best


def predict_scaled(model, X_seq):
    """
    用训练好的模型做预测，返回【标准化尺度】的一维 numpy 数组。

    必须在 eval + no_grad 下进行：本周是单层 LSTM，没有 dropout/batchnorm
    的差别，但这个习惯要保留，否则将来换模型容易出错。
    """
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(np.asarray(X_seq, dtype=np.float32))
        pred = model(X_t).cpu().numpy().reshape(-1)
    return pred
