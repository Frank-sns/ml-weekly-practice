# -*- coding: utf-8 -*-
"""
reporting.py —— 把已经算好的结果保存成 CSV 和图片

职责：只负责"接收结果 -> 写文件 / 画图"，不训练模型、不计算指标。
这样入口（run_*.py）里就看不到训练循环和绘图实现，各文件职责清晰。
"""
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")   # 无窗口后端，直接把图存成文件
import matplotlib.pyplot as plt

# 中文字体，让图里的中文正常显示
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

RESULTS = "results"


def _ensure(path):
    """确保目标文件所在目录存在。"""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    return path


# ============================================================
# 实践一
# ============================================================
def save_step_trace(trace, path=f"{RESULTS}/step_trace.csv"):
    """保存实践一的更新记录（三行：清梯度后 / 反传后 / 更新后）。"""
    df = pd.DataFrame(trace)
    df = df[["stage", "weight", "bias", "weight_grad", "bias_grad", "loss"]]
    df.to_csv(_ensure(path), index=False)
    print("一次梯度更新记录：")
    print(df.to_string(index=False))
    print(f"\n已保存：{path}")


# ============================================================
# 实践二
# ============================================================
def save_history(history_rows, path=f"{RESULTS}/history.csv"):
    """保存两组的逐轮训练/验证 MSE（每组 200 行，共 400 行）。"""
    df = pd.DataFrame(history_rows)
    df = df[["配置", "学习率", "epoch", "训练MSE(标准化)", "验证MSE(标准化)"]]
    df.to_csv(_ensure(path), index=False)
    print(f"已保存：{path}（共 {len(df)} 行，MSE 为标准化目标上的值）")


def save_metrics(metric_rows, path=f"{RESULTS}/metrics.csv"):
    """保存四组模型的指标表（均为原始单位）。"""
    df = pd.DataFrame(metric_rows)
    cols = ["模型", "学习率", "最佳轮次", "训练MAE", "训练RMSE",
            "验证MAE", "验证RMSE", "测试MAE", "测试RMSE", "选定标记"]
    df = df[cols]
    df.to_csv(_ensure(path), index=False)
    print("\n指标表（原始单位，MAE/RMSE 越小越好）：")
    print(df.to_string(index=False))
    print(f"\n已保存：{path}")


def plot_loss_curves(sgd_history, adam_history, sgd_best_epoch, adam_best_epoch,
                     path=f"{RESULTS}/loss_curves.png"):
    """双面板训练曲线：左 SGD、右 Adam，各画训练/验证 MSE，并标出最佳轮次。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    panels = [
        (axes[0], sgd_history, sgd_best_epoch, "SGD (lr=0.01)"),
        (axes[1], adam_history, adam_best_epoch, "Adam (lr=0.001)"),
    ]
    for ax, hist, best_epoch, title in panels:
        epochs = [h["epoch"] for h in hist]
        train_mse = [h["train_mse"] for h in hist]
        val_mse = [h["val_mse"] for h in hist]
        ax.plot(epochs, train_mse, label="训练 MSE")
        ax.plot(epochs, val_mse, label="验证 MSE")
        ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=0.8)
        ax.annotate(f"最佳轮次 {best_epoch}",
                    xy=(best_epoch, val_mse[best_epoch - 1]),
                    xytext=(best_epoch + 3, val_mse[best_epoch - 1] + 0.05))
        ax.set_xlabel("epoch")
        ax.set_ylabel("标准化目标 MSE")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(_ensure(path), dpi=150)
    plt.close()
    print(f"已保存：{path}")


def plot_rmse_comparison(metric_rows, path=f"{RESULTS}/rmse_comparison.png"):
    """四组模型（均值基线/线性回归/SGD MLP/Adam MLP）的训练/验证 RMSE 柱状图。"""
    names = [r["模型"] for r in metric_rows]
    train_rmse = [r["训练RMSE"] for r in metric_rows]
    val_rmse = [r["验证RMSE"] for r in metric_rows]
    x = np.arange(len(names))
    width = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, train_rmse, width, label="训练 RMSE")
    plt.bar(x + width / 2, val_rmse, width, label="验证 RMSE")
    plt.xticks(x, names)
    plt.ylabel("RMSE（原始目标单位）")
    plt.title("四组模型训练/验证 RMSE 对照（MLP 用各自最佳轮次参数）")
    plt.legend()
    plt.tight_layout()
    plt.savefig(_ensure(path), dpi=150)
    plt.close()
    print(f"已保存：{path}")
