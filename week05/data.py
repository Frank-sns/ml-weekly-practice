# -*- coding: utf-8 -*-
"""
data.py —— 数据读取、划分、标准化、张量与批次准备

职责：
1. 实践一：准备两个演示样本 x、y。
2. 实践二：读 diabetes，60/20/20 划分，标准化 X 和 y，
   并把 numpy 数组转成 torch 张量、打包成 DataLoader。

说明：这里只负责"把数据准备好"，不涉及模型和训练。
"""
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass


# ========== 实践一：两个演示样本 ==========
def make_step_samples():
    """
    实践一用的两个样本（形状都是 (2, 1)）：
      x = [[1.0], [2.0]]  两个输入
      y = [[2.0], [4.0]]  两个标签（正好满足 y = 2x）
    """
    x = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    y = torch.tensor([[2.0], [4.0]], dtype=torch.float32)
    return x, y


# ========== 实践二：diabetes 数据 ==========
@dataclass
class DiabetesData:
    """把实践二需要的数据打包成一个对象，字段名即含义，避免一堆裸数组乱传。"""
    X_train: np.ndarray      # 标准化特征（numpy），供 sklearn 基线用
    X_val: np.ndarray
    X_test: np.ndarray
    y_train_scaled: np.ndarray   # 标准化目标（numpy）
    y_val_scaled: np.ndarray
    y_test_scaled: np.ndarray
    y_train_raw: np.ndarray      # 原始目标（numpy），评价时恢复尺度用
    y_val_raw: np.ndarray
    y_test_raw: np.ndarray
    y_mean: float                # 训练目标的均值（标准化用，恢复尺度也用）
    y_std: float                 # 训练目标的标准差


def prepare_diabetes(random_state=42):
    """
    读 diabetes，划分 60/20/20，标准化 X 和 y。

    关键点（与前面几周一致）：
    - X 的标准化：只用训练集 fit 出均值和标准差，验证/测试集沿用同一套统计量。
    - y 的标准化：用训练目标的 y_mean、y_std，三部分统一算 (y - y_mean) / y_std。
      标准差用 np.std(ddof=0)（即除以 N，不除以 N-1），并检查必须大于 0。
    """
    dataset = load_diabetes(as_frame=True, scaled=False)
    df = dataset.frame
    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols].to_numpy()          # (442, 10)
    y = df["target"].to_numpy()              # (442,)

    # 60 / 20 / 20 划分，随机种子 42
    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.4, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=random_state)

    # X 标准化：fit 只在训练集上做
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # y 标准化：用训练目标的均值、标准差（ddof=0）
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train, ddof=0))
    assert y_std > 0, "训练目标标准差为 0，无法标准化，请检查数据"
    y_train_scaled = (y_train - y_mean) / y_std
    y_val_scaled = (y_val - y_mean) / y_std
    y_test_scaled = (y_test - y_mean) / y_std

    return DiabetesData(
        X_train=X_train_scaled, X_val=X_val_scaled, X_test=X_test_scaled,
        y_train_scaled=y_train_scaled, y_val_scaled=y_val_scaled, y_test_scaled=y_test_scaled,
        y_train_raw=y_train, y_val_raw=y_val, y_test_raw=y_test,
        y_mean=y_mean, y_std=y_std,
    )


def to_tensors(X_np, y_scaled_np):
    """
    把 numpy 的 X(N, d)、y(N,) 转成 torch.float32 张量。
    注意 y 会被 reshape 成列向量 (N, 1)，和模型输出形状一致，方便算损失。
    """
    Xt = torch.tensor(X_np, dtype=torch.float32)
    yt = torch.tensor(y_scaled_np, dtype=torch.float32).reshape(-1, 1)
    return Xt, yt


def make_loader(X_np, y_scaled_np, batch_size=32, seed=42):
    """
    用 TensorDataset 把特征和标签"配对"绑在一起，再包成 DataLoader 分批。
    这样取出来的每个 batch 是 (xb, yb)，其中 xb、yb 里样本一一对应、不会错位。

    shuffle=True 表示每个 epoch 重新打乱顺序；
    generator=torch.Generator().manual_seed(seed) 让打乱可复现（两组用相同 seed）；
    num_workers=0 表示在当前进程里读数据（简单、可复现）；
    drop_last=False 表示最后不足一批的样本也保留（265 / 32 有余数也照常训练）。
    """
    Xt, yt = to_tensors(X_np, y_scaled_np)
    dataset = TensorDataset(Xt, yt)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        generator=generator, num_workers=0, drop_last=False,
    )
