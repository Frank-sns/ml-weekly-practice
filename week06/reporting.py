# -*- coding: utf-8 -*-
"""
reporting.py —— 保存表格和图形

职责：接收已经算好的记录和指标，负责保存 CSV 与绘图，
不拟合模型、不计算指标。绘图数据全部来自调用方传入的记录。
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
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    return path


# ============================================================
# 实践一
# ============================================================
def save_window_examples(rows, path=f"{RESULTS}/window_examples.csv"):
    """保存两种窗口长度（L=3、L=5）的全部构造记录。"""
    df = pd.DataFrame(rows)
    cols = ["window_size", "input_start", "input_end", "target_index",
            "target_time", "input_values", "target_value"]
    df = df[cols]
    df.to_csv(_ensure(path), index=False)
    print(f"已保存：{path}（共 {len(df)} 行，L=3 与 L=5 各一组）")
    print(df.head(3).to_string(index=False))
    print("……")


# ============================================================
# 实践二
# ============================================================
def save_tuning(rows, path=f"{RESULTS}/tuning.csv"):
    """保存调参对照表：lag_linear 参照 + Ridge 三个候选。"""
    df = pd.DataFrame(rows)
    cols = ["model", "param_name", "param_value", "train_mae", "val_mae",
            "train_rmse", "val_rmse", "rmse_gap", "chosen_candidate"]
    df = df[cols]
    df.to_csv(_ensure(path), index=False)
    print("\n调参对照表（RMSE 越小越好，rmse_gap = val_rmse - train_rmse）：")
    print(df.to_string(index=False))
    print(f"\n已保存：{path}")


def save_metrics(rows, path=f"{RESULTS}/metrics.csv"):
    """保存四种最终方法的指标表（每个模型 × 训练/验证/测试各一行）。"""
    df = pd.DataFrame(rows)
    cols = ["model", "split", "n_samples", "mae", "rmse", "selected"]
    df = df[cols]
    df.to_csv(_ensure(path), index=False)
    print("\n最终指标表（原始油温尺度）：")
    print(df.to_string(index=False))
    print(f"\n已保存：{path}")


def save_predictions(rows, path=f"{RESULTS}/predictions.csv"):
    """保存每个模型、每个目标的逐条预测（含目标下标与真实时间）。"""
    df = pd.DataFrame(rows)
    cols = ["model", "split", "target_index", "target_time", "y_true", "y_pred"]
    df = df[cols]
    df.to_csv(_ensure(path), index=False)
    print(f"已保存：{path}（共 {len(df)} 行，4 个模型 × 14376 个目标）")


def plot_tuning_comparison(rows, path=f"{RESULTS}/tuning_comparison.png"):
    """
    调参对照图：横轴是不同【配置】（lag_linear + 3 个 Ridge 候选），
    纵轴是 RMSE（原始油温尺度），分组柱状图对比训练/验证误差。

    注意：这张图比较的是"不同配置"，不是第五周曲线里的"不同训练轮次"。
    """
    df = pd.DataFrame(rows)
    labels = []
    for _, r in df.iterrows():
        if r["model"] == "lag_linear":
            labels.append("线性回归\nlag_linear")
        else:
            labels.append(f"Ridge\nα={r['param_value']}")
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, df["train_rmse"], width, label="训练 RMSE")
    plt.bar(x + width / 2, df["val_rmse"], width, label="验证 RMSE")
    plt.xticks(x, labels)
    plt.ylabel("RMSE（原始油温尺度）")
    plt.title("不同配置的训练/验证 RMSE 对照（含线性回归参照）")
    plt.legend()
    plt.tight_layout()
    plt.savefig(_ensure(path), dpi=150)
    plt.close()
    print(f"已保存：{path}")


def plot_series_split(times, series, val_start, test_start, window_size=24,
                      path=f"{RESULTS}/series_split.png"):
    """
    实验时段油温图：前 14400 小时的 OT 曲线，
    标出验证起点、测试起点，以及形成首条样本所用的初始历史（前 24 小时）。
    """
    plt.figure(figsize=(13, 5))
    plt.plot(times, series, linewidth=0.8, label="油温 OT")
    plt.axvline(pd.Timestamp(val_start), color="orange", linestyle="--",
                linewidth=1.2, label="验证起点 2017-06-26")
    plt.axvline(pd.Timestamp(test_start), color="red", linestyle="--",
                linewidth=1.2, label="测试起点 2017-10-24")
    # 首条样本的输入历史 = 最开始的 window_size 个小时
    plt.axvspan(times[0], times[window_size - 1], color="green", alpha=0.25,
                label="首条样本的输入历史（24 小时）")
    plt.xlabel("date")
    plt.ylabel("油温 OT（原始数据尺度）")
    plt.title("前 14,400 小时油温与训练/验证/测试划分")
    plt.legend()
    plt.tight_layout()
    plt.savefig(_ensure(path), dpi=150)
    plt.close()
    print(f"已保存：{path}")


def plot_validation_predictions(val_week_rows, hufl_series, hufl_times,
                                path=f"{RESULTS}/validation_predictions.png"):
    """
    验证集首周（2017-06-26 00:00 ~ 2017-07-02 23:00）双面板图：
     上图：真实油温 + 四种方法的预测（横轴为真实时间）
     下图：同一时段的 HUFL 负载记录（来自原始 CSV，按时间对齐）

    指标仍在完整验证集上计算，这张图只是取首周做"局部观察"。
    """
    pred_df = pd.DataFrame(val_week_rows)
    # 每个模型一列，按目标时间对齐
    pivot = pred_df.pivot(index="target_time", columns="model", values="y_pred")
    pivot = pivot.sort_index()
    y_true = pred_df.drop_duplicates("target_time").set_index("target_time")["y_true"]
    y_true = y_true.loc[pivot.index]   # 与 pivot 行序对齐

    # 时间轴统一转成 numpy datetime64 数组再画，
    # 避免 pandas 对象直接传给 matplotlib 时的单位转换冲突
    t_axis = np.asarray(pivot.index.to_numpy(), dtype="datetime64[ns]")
    t_start = pd.Timestamp(pivot.index[0])
    t_end = pd.Timestamp(pivot.index[-1])

    # 负载曲线取同一时段（原始 CSV，按真实时间对齐）
    hufl_df = pd.DataFrame({"time": hufl_times, "HUFL": hufl_series})
    hufl_week = hufl_df[
        (hufl_df["time"] >= t_start) & (hufl_df["time"] <= t_end)
    ]
    t_hufl = np.asarray(hufl_week["time"].to_numpy(), dtype="datetime64[ns]")

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(t_axis, y_true.to_numpy(), color="black", linewidth=1.5,
                 label="真实油温")
    for col in pivot.columns:
        axes[0].plot(t_axis, pivot[col].to_numpy(), linewidth=0.9, label=col)
    axes[0].set_ylabel("油温 OT（原始数据尺度）")
    axes[0].set_title("验证集首周：真实油温与四种方法预测")
    axes[0].legend(loc="upper right", ncol=3)

    axes[1].plot(t_hufl, hufl_week["HUFL"].to_numpy(dtype=float),
                 color="steelblue", linewidth=0.9, label="HUFL")
    axes[1].set_ylabel("HUFL（原始数据尺度）")
    axes[1].set_title("同一时段的 HUFL 负载记录（仅做事后对照，未输入模型）")
    axes[1].legend()

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(_ensure(path), dpi=150)
    plt.close()
    print(f"已保存：{path}")
