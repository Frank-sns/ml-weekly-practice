# -*- coding: utf-8 -*-
"""
evaluation.py —— 原始尺度上的 MAE / RMSE，以及"从逐条预测重算指标"

这个文件不训练、不绘图，只做指标计算和核对。
"""
import numpy as np
import pandas as pd


def mae_rmse(y_true, y_pred):
    """
    原始油温尺度上的 MAE 和 RMSE：
      MAE  = mean(|y_true - y_pred|)
      RMSE = sqrt(mean((y_true - y_pred)^2))

    两个指标都要报：MAE 好理解，RMSE 对大误差更敏感。
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return mae, rmse


def verify_metrics_from_predictions(rows):
    """
    从 predictions.csv 的每一行重算 MAE/RMSE，按 (model, split) 分组。

    作用：如果这里算出来的数和汇总表一致，说明汇总表没有抄错，
    而且两处确实来自同一批预测。
    """
    df = pd.DataFrame(rows)
    result = {}
    for (model, split), group in df.groupby(["model", "split"]):
        mae, rmse = mae_rmse(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        result[(model, split)] = (mae, rmse)
    return result
