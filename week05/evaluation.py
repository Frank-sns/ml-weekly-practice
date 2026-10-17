# -*- coding: utf-8 -*-
"""
evaluation.py —— 只做评价，不更新参数

职责：
1. 前向预测（标准化目标尺度），内部切到 eval 模式并用 no_grad 关掉梯度。
2. 计算标准化目标上的 MSE（供每轮训练曲线用）。
3. 把标准化预测恢复到原始单位，计算 MAE / RMSE。

⚠️ 本文件里的函数都不会调用 backward() 或 optimizer.step()，
   评价阶段绝不改变模型参数。
"""
import numpy as np
import torch


def predict_scaled(model, X):
    """
    前向预测，返回"标准化目标尺度"上的预测值。

    两个关键设置：
    - model.eval()：切换到评估模式（对本 MLP 来说和训练模式等价，
      因为没有 dropout/batchnorm，但这是正确习惯）；
    - torch.no_grad()：关掉自动求导，这样不会为预测建计算图、不产生梯度。
    """
    model.eval()
    with torch.no_grad():
        return model(X)


def mse_scaled(model, X, y_scaled):
    """
    在标准化目标上算 MSE = mean((pred - y)^2)。
    用于每轮训练结束后、对"完整训练集/验证集"重新评价，画训练曲线。
    """
    pred = predict_scaled(model, X)
    return float(((pred - y_scaled) ** 2).mean().item())


def restore_scale(pred_scaled, y_mean, y_std):
    """
    把标准化预测恢复到原始单位：
        pred_raw = pred_scaled * y_std + y_mean
    这正是 data.py 里 y 标准化公式 (y - y_mean) / y_std 的逆运算。
    """
    return pred_scaled * y_std + y_mean


def mae_rmse(y_true, y_pred):
    """
    在原始单位上计算 MAE 和 RMSE（输入都转成一维 numpy 数组）。
    MAE  = mean(|y_true - y_pred|)
    RMSE = sqrt(mean((y_true - y_pred)^2))
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return mae, rmse


def evaluate_mlp_raw(model, X, y_raw, y_mean, y_std):
    """
    评价一个已训练好的 MLP，结果用原始单位：
      预测(标准化) -> 恢复原始单位 -> 和原始目标比 MAE/RMSE。
    X 是标准化特征张量，y_raw 是原始目标（numpy）。
    """
    pred_scaled = predict_scaled(model, X)
    pred_raw = restore_scale(pred_scaled, y_mean, y_std).numpy()
    return mae_rmse(y_raw, pred_raw)


def evaluate_mean_baseline(y_mean_pred, y_raw):
    """
    评价均值基线：不管输入，预测值恒为训练目标均值 y_mean_pred，
    直接和原始目标比 MAE/RMSE。
    """
    pred = np.full_like(np.asarray(y_raw), y_mean_pred, dtype=float)
    return mae_rmse(y_raw, pred)


def evaluate_linear_baseline(lr, X_np, y_raw):
    """
    评价线性回归基线：LinearRegression 是用"标准化输入 + 原始目标"拟合的，
    所以 lr.predict(X_np) 出来的直接就是原始单位预测。
    """
    pred = lr.predict(X_np)
    return mae_rmse(y_raw, pred)
