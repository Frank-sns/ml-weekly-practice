# -*- coding: utf-8 -*-
"""
reporting.py —— 保存 CSV / 模型文件，绘制两张图

输出都放在 week07/results/ 下：
  rnn_trace.csv              实践一的状态记录
  history.csv                两组 LSTM 各 40 轮的训练/验证 MSE
  metrics.csv                各模型、各数据部分的 MAE / RMSE
  predictions.csv            逐条预测（原始尺度）
  loss_curves.png            两组训练验证曲线
  validation_predictions.png 验证首周预测对照
  lstm_selected.pt           选定 LSTM 的参数与标准化统计量
"""
import os

import matplotlib

matplotlib.use("Agg")           # 不弹窗口，直接存文件
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def _path(name):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, name)


def save_rnn_trace(rows):
    """实践一的五步记录，两组各 5 行。"""
    columns = ["case", "step", "source_index", "source_time", "z",
               "h_prev", "input_term", "memory_term", "h", "input_grad"]
    df = pd.DataFrame(rows)[columns]
    df.to_csv(_path("rnn_trace.csv"), index=False, encoding="utf-8-sig")
    print(f"  已保存 results/rnn_trace.csv（{len(df)} 行）")


def save_history(rows):
    """两组 LSTM 的逐轮记录，80 行；mse 是标准化尺度。"""
    columns = ["model", "hidden_size", "epoch", "train_mse", "val_mse"]
    df = pd.DataFrame(rows)[columns]
    df.to_csv(_path("history.csv"), index=False, encoding="utf-8-sig")
    print(f"  已保存 results/history.csv（{len(df)} 行）")


def save_metrics(rows):
    """每个模型 × 每个数据部分一行。"""
    columns = ["model", "split", "n_samples", "hidden_size",
               "best_epoch", "mae", "rmse", "chosen_lstm"]
    df = pd.DataFrame(rows)[columns]
    df.to_csv(_path("metrics.csv"), index=False, encoding="utf-8-sig")
    print(f"  已保存 results/metrics.csv（{len(df)} 行）")


def save_predictions(rows):
    """逐条预测，原始油温尺度。"""
    columns = ["model", "split", "target_index", "target_time", "y_true", "y_pred"]
    df = pd.DataFrame(rows)[columns]
    df.to_csv(_path("predictions.csv"), index=False, encoding="utf-8-sig")
    print(f"  已保存 results/predictions.csv（{len(df)} 行）")


def save_model_bundle(model, hidden_size, best_epoch, mu, sigma,
                      window_size=24, filename="lstm_selected.pt"):
    """
    保存选定 LSTM 的模型文件。

    除了参数，还要把"怎么用这份参数"需要的信息一起存：
      hidden_size  重建网络结构时要用
      best_epoch   说明这是第几轮的参数
      mu, sigma    输入标准化和输出还原都要用，缺了预测就对不上
    """
    bundle = {
        "state_dict": model.state_dict(),
        "hidden_size": int(hidden_size),
        "best_epoch": int(best_epoch),
        "mu": float(mu),
        "sigma": float(sigma),
        "window_size": int(window_size),
        "input_size": 1,
        "num_layers": 1,
        "bidirectional": False,
    }
    path = _path(filename)
    torch.save(bundle, path)
    print(f"  已保存 results/{filename}")


def plot_loss_curves(history_by_hidden, best_epoch_by_hidden):
    """
    两个子图：隐藏维数 16 和 32 各自的训练/验证 MSE 曲线。

    history_by_hidden : {16: [ {epoch, train_mse, val_mse}, ... ], 32: [...]}
    best_epoch_by_hidden : {16: 轮次, 32: 轮次}
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, hidden in zip(axes, sorted(history_by_hidden)):
        hist = history_by_hidden[hidden]
        epochs = [r["epoch"] for r in hist]
        train = [r["train_mse"] for r in hist]
        val = [r["val_mse"] for r in hist]
        best_epoch = best_epoch_by_hidden[hidden]

        ax.plot(epochs, train, label="训练 MSE", color="steelblue")
        ax.plot(epochs, val, label="验证 MSE", color="darkorange")
        ax.axvline(best_epoch, color="gray", linestyle=":", linewidth=1)
        best_val = val[best_epoch - 1]
        ax.scatter([best_epoch], [best_val], color="darkorange", zorder=5)
        # 标注放在最佳轮次的左上方，避免顶到图右侧边界
        ax.annotate(f"最佳轮次 {best_epoch}\n验证 MSE {best_val:.4f}",
                    xy=(best_epoch, best_val),
                    xytext=(best_epoch - 15, best_val + (max(val) - min(val)) * 0.45),
                    fontsize=9, color="darkorange",
                    arrowprops=dict(arrowstyle="->", color="darkorange"))

        ax.set_title(f"隐藏维数 H = {hidden}")
        ax.set_xlabel("训练轮次 epoch")
        ax.set_ylabel("MSE（标准化尺度）")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("两组 LSTM 的训练/验证 MSE（标准化尺度，越小越好）")
    plt.tight_layout()
    plt.savefig(_path("loss_curves.png"), dpi=150)
    plt.close()
    print("  已保存 results/loss_curves.png")


def plot_validation_predictions(rows, start, end):
    """
    验证首周的预测对照：真实油温、last_value、lag_linear、选定 LSTM。

    rows 是 predictions.csv 的行（只取 split == "val"），
    按 target_time 对齐到时间轴，纵轴是原始油温尺度。
    """
    df = pd.DataFrame(rows)
    df = df[df["split"] == "val"].copy()
    df["target_time"] = pd.to_datetime(df["target_time"])
    df = df[(df["target_time"] >= start) & (df["target_time"] <= end)]

    fig, ax = plt.subplots(figsize=(11, 4.5))

    # 真实值对每个模型都相同，取一份即可
    truth = (df[df["model"] == df["model"].iloc[0]]
             .sort_values("target_time")[["target_time", "y_true"]])
    t_axis = np.asarray(truth["target_time"].to_numpy(), dtype="datetime64[ns]")
    ax.plot(t_axis, truth["y_true"].to_numpy(), color="black", linewidth=2,
            label="真实油温")

    style = {
        "last_value": ("tab:gray", "--"),
        "lag_linear": ("tab:green", "-."),
    }
    for model in df["model"].unique():
        if model == "last_value":
            color, ls = style["last_value"]
        elif model == "lag_linear":
            color, ls = style["lag_linear"]
        else:                       # 选定 LSTM
            color, ls = "tab:red", "-"
        sub = df[df["model"] == model].sort_values("target_time")
        t = np.asarray(sub["target_time"].to_numpy(), dtype="datetime64[ns]")
        ax.plot(t, sub["y_pred"].to_numpy(), color=color, linestyle=ls,
                linewidth=1.6 if model.startswith("lstm") else 1.2,
                label=model)

    ax.set_title("验证首周（2017-06-26 00:00 ~ 2017-07-02 23:00）预测对照")
    ax.set_xlabel("目标时间")
    ax.set_ylabel("油温 OT（原始尺度）")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()             # 日期轴自动倾斜，避免重叠
    plt.tight_layout()
    plt.savefig(_path("validation_predictions.png"), dpi=150)
    plt.close()
    print("  已保存 results/validation_predictions.png")
