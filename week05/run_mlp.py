# -*- coding: utf-8 -*-
"""
run_mlp.py —— 实践二入口：训练一个小型 MLP，并和两个基线对照

流程（每个步骤都调用对应组件，入口本身不写训练循环和绘图实现）：
1. 准备数据（划分 + 标准化，X 和 y 都标准化）。
2. 用同一份初始参数建两个 MLP，分别配 SGD、Adam，各训练 200 轮。
3. 每轮结束后用完整训练/验证集评价，记录最佳验证轮次与参数。
4. 拟合均值基线和线性回归基线。
5. 按验证 RMSE 在两组 MLP 中选一组（相同选 SGD），算测试集指标。
6. 交给 reporting 出 CSV 和图。
7. 做一次 hidden_dim=8 的前向形状检查（只检查，不重新训练）。

运行（在 week05/ 目录下）：python run_mlp.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")   # 修 Windows 控制台中文乱码

import copy
import numpy as np
import torch
import torch.nn as nn

import data
import models
import optimizers
import training
import evaluation
import reporting


# 两组实验配置（学习率不同，比较的是两套具体配置）
CONFIGS = [
    {"name": "SGD",  "optimizer": "sgd",  "lr": 0.01},
    {"name": "Adam", "optimizer": "adam", "lr": 0.001},
]
BATCH_SIZE = 32
EPOCHS = 200


def main():
    # ---- 1. 准备数据 ----
    d = data.prepare_diabetes()
    # 转成 torch 张量（y 变 (N,1)），供"完整数据"评价用
    Xtr_t, ytr_t = data.to_tensors(d.X_train, d.y_train_scaled)
    Xva_t, yva_t = data.to_tensors(d.X_val, d.y_val_scaled)
    Xte_t, yte_t = data.to_tensors(d.X_test, d.y_test_scaled)

    # ---- 2. 建两个初始参数完全相同的 MLP ----
    # 固定随机种子创建初始网络，再用 deepcopy 得到两组参数相同的独立模型
    torch.manual_seed(42)
    base = models.make_mlp(hidden_dim=16)
    mlps = {
        "SGD": copy.deepcopy(base),
        "Adam": copy.deepcopy(base),
    }

    loss_fn = nn.MSELoss()

    # ---- 3. 训练两组，收集逐轮历史与最佳轮次 ----
    histories = {}
    best_epochs = {}
    for cfg in CONFIGS:
        name = cfg["name"]
        model = mlps[name]
        optimizer = optimizers.make_optimizer(model, cfg["optimizer"], cfg["lr"])
        loader = data.make_loader(d.X_train, d.y_train_scaled, batch_size=BATCH_SIZE, seed=42)
        history, best_epoch, best_val = training.train_model(
            model, optimizer, loss_fn, loader,
            Xtr_t, ytr_t, Xva_t, yva_t, EPOCHS,
        )
        histories[name] = history
        best_epochs[name] = best_epoch
        print(f"{name} 训练完成：最佳验证 MSE = {best_val:.4f}（第 {best_epoch} 轮）")

    # ---- 4. 拟合两个基线 ----
    mean_pred = training.fit_mean_baseline(d.y_train_raw)               # 均值基线（一个数）
    linear = training.fit_linear_baseline(d.X_train, d.y_train_raw)     # 线性回归基线

    # ---- 5. 计算各组训练/验证 MAE、RMSE（原始单位），先算训练/验证 ----
    metric_rows = []

    # 均值基线
    tr_mae, tr_rmse = evaluation.evaluate_mean_baseline(mean_pred, d.y_train_raw)
    va_mae, va_rmse = evaluation.evaluate_mean_baseline(mean_pred, d.y_val_raw)
    te_mae, te_rmse = evaluation.evaluate_mean_baseline(mean_pred, d.y_test_raw)
    metric_rows.append({"模型": "均值基线", "学习率": "", "最佳轮次": "",
                        "训练MAE": round(tr_mae, 2), "训练RMSE": round(tr_rmse, 2),
                        "验证MAE": round(va_mae, 2), "验证RMSE": round(va_rmse, 2),
                        "测试MAE": round(te_mae, 2), "测试RMSE": round(te_rmse, 2),
                        "选定标记": ""})

    # 线性回归基线
    tr_mae, tr_rmse = evaluation.evaluate_linear_baseline(linear, d.X_train, d.y_train_raw)
    va_mae, va_rmse = evaluation.evaluate_linear_baseline(linear, d.X_val, d.y_val_raw)
    te_mae, te_rmse = evaluation.evaluate_linear_baseline(linear, d.X_test, d.y_test_raw)
    metric_rows.append({"模型": "线性回归", "学习率": "", "最佳轮次": "",
                        "训练MAE": round(tr_mae, 2), "训练RMSE": round(tr_rmse, 2),
                        "验证MAE": round(va_mae, 2), "验证RMSE": round(va_rmse, 2),
                        "测试MAE": round(te_mae, 2), "测试RMSE": round(te_rmse, 2),
                        "选定标记": ""})

    # 两组 MLP（用各自最佳轮次参数评价；测试列先留空，选定后再补）
    mlp_val_rmse = {}
    for cfg in CONFIGS:
        name = cfg["name"]
        model = mlps[name]
        tr_mae, tr_rmse = evaluation.evaluate_mlp_raw(model, Xtr_t, d.y_train_raw, d.y_mean, d.y_std)
        va_mae, va_rmse = evaluation.evaluate_mlp_raw(model, Xva_t, d.y_val_raw, d.y_mean, d.y_std)
        mlp_val_rmse[name] = va_rmse
        metric_rows.append({"模型": f"{name} MLP", "学习率": cfg["lr"], "最佳轮次": best_epochs[name],
                            "训练MAE": round(tr_mae, 2), "训练RMSE": round(tr_rmse, 2),
                            "验证MAE": round(va_mae, 2), "验证RMSE": round(va_rmse, 2),
                            "测试MAE": "", "测试RMSE": "", "选定标记": ""})

    # ---- 6. 按验证 RMSE 选一组（相同固定选 SGD），补测试指标 + 标记 ----
    selected = "SGD" if mlp_val_rmse["SGD"] <= mlp_val_rmse["Adam"] else "Adam"
    for row in metric_rows:
        if row["模型"] == f"{selected} MLP":
            te_mae, te_rmse = evaluation.evaluate_mlp_raw(
                mlps[selected], Xte_t, d.y_test_raw, d.y_mean, d.y_std)
            row["测试MAE"] = round(te_mae, 2)
            row["测试RMSE"] = round(te_rmse, 2)
            row["选定标记"] = "选定"
    print(f"\n依据验证 RMSE 选定：{selected} MLP")

    # ---- 7. 组装 history 记录（加配置、学习率两列）----
    history_rows = []
    for cfg in CONFIGS:
        name = cfg["name"]
        for h in histories[name]:
            history_rows.append({
                "配置": name, "学习率": cfg["lr"], "epoch": h["epoch"],
                "训练MSE(标准化)": round(h["train_mse"], 6),
                "验证MSE(标准化)": round(h["val_mse"], 6),
            })

    # ---- 8. 交给 reporting 出结果 ----
    reporting.save_history(history_rows)
    reporting.save_metrics(metric_rows)
    reporting.plot_loss_curves(histories["SGD"], histories["Adam"],
                               best_epochs["SGD"], best_epochs["Adam"])
    reporting.plot_rmse_comparison(metric_rows)

    # ---- 9. 隐藏层宽度修改练习：hidden_dim=8 前向形状检查 ----
    print("\n" + "=" * 56)
    print("隐藏层宽度修改检查：hidden_dim=8（只做前向，不重新训练）")
    print("=" * 56)
    m8 = models.make_mlp(input_dim=10, hidden_dim=8, output_dim=1)
    X_check = torch.randn(32, 10)   # 假设一批 32 条、每条 10 个特征
    models.forward_shape_chain(m8, X_check)
    print("（正确形状应为 (32,10) -> (32,8) -> (32,8) -> (32,1)）")

    print("\n全部完成！结果都在 week05/results/ 目录下。")


if __name__ == "__main__":
    main()
