# -*- coding: utf-8 -*-
"""
evaluation.py —— MAE、RMSE 与预测结果核对

职责：只做指标计算和核对，不拟合模型、不绘图。
"""
import numpy as np
import pandas as pd


def mae_rmse(y_true, y_pred):
    """
    原始油温尺度上的 MAE 和 RMSE：
      MAE  = mean(|y_true - y_pred|)
      RMSE = sqrt(mean((y_true - y_pred)^2))
    输入都转成一维 numpy 数组。
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return mae, rmse


def evaluate_rule(rule_fn, X, y):
    """评价一个固定规则（函数）：先用规则算预测，再算 MAE/RMSE。"""
    return mae_rmse(y, rule_fn(X))


def evaluate_estimator(estimator, X, y):
    """评价一个 sklearn 管道模型：predict 后算 MAE/RMSE。"""
    return mae_rmse(y, estimator.predict(X))


def verify_metrics_from_predictions(rows):
    """
    从保存的预测记录（predictions.csv 的每一行）重新计算 MAE/RMSE，
    按 (model, split) 分组，用于和汇总表核对：如果两处一致，
    说明保存的逐条预测与汇总指标确实来自同一批计算。
    """
    df = pd.DataFrame(rows)
    result = {}
    for (model, split), g in df.groupby(["model", "split"]):
        mae, rmse = mae_rmse(g["y_true"].to_numpy(), g["y_pred"].to_numpy())
        result[(model, split)] = (mae, rmse)
    return result
