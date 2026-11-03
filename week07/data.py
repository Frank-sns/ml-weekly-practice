# -*- coding: utf-8 -*-
"""
data.py —— 读取 ETTh1、构造窗口、按目标时刻划分、标准化

本周任务和第六周是同一个预测问题（过去 24 小时 → 下一小时），
所以窗口构造和划分逻辑直接沿用；真正不同的是数据交给模型时的形状：

    第六周：X 是 (N, 24)，24 个历史值当成 24 个"特征列"，
            线性回归 / MLP 一次性看到全部 24 个值。

    本周：  X 是 (N, 24, 1)，第 1 维是"时间步"，第 2 维是"该时刻的特征数"。
            RNN / LSTM 按时间步一个一个读，每次只吃一个数，
            同时接收上一步的隐藏状态。

除此之外，本周还要用【训练时段原始 OT】的均值 μ 和总体标准差 σ 做标准化。
注意"每个时刻只计一次"：不能用拍平后的窗口统计，
因为同一个油温值会同时出现在 24 个不同窗口里，会被重复计算 24 次。
"""
import os

import numpy as np
import pandas as pd

# 数据复用第六周的副本，不重复保存（任务书允许）
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.normpath(os.path.join(HERE, "..", "week06", "data", "ETTh1.csv"))

# 与第六周完全一致的时间边界（左闭右开）
TRAIN_START = pd.Timestamp("2016-07-01 00:00")
VAL_START = pd.Timestamp("2017-06-26 00:00")
TEST_START = pd.Timestamp("2017-10-24 00:00")
N_ROWS = 14400          # 只用前 14400 小时
WINDOW_SIZE = 24        # 历史窗口长度


def read_etth1(path=DATA_PATH):
    """读取 ETTh1.csv，date 列解析成日期时间。"""
    return pd.read_csv(path, parse_dates=["date"])


def check_etth1(times, series):
    """
    检查时间严格递增、无重复、相邻记录相差 1 小时、数值均有效（无 NaN）。

    返回 True / False，并逐项打印结果。
    """
    diff_hours = np.diff(times.astype("datetime64[h]").astype("int64"))
    checks = {
        "时间严格递增": bool(np.all(diff_hours > 0)),
        "无重复时间": bool(len(np.unique(times)) == len(times)),
        "相邻记录相差 1 小时": bool(np.all(diff_hours == 1)),
        "数值均有效（无 NaN）": bool(not np.isnan(series).any()),
    }
    ok = True
    for name, passed in checks.items():
        print(f"  {name}：{'通过' if passed else '未通过'}")
        ok = ok and passed
    return ok


def make_windows(series, window_size=WINDOW_SIZE):
    """
    把连续序列切成"历史窗口 → 下一小时目标"的样本（与第六周相同）。

    返回：
      X            : (N-L, L)，每行 = 连续 L 个历史油温（从旧到新）
      y            : (N-L,)，每行对应的下一小时油温
      target_index : (N-L,)，每个目标在原始序列中的下标（= i + L）

    样本数 = N - L：最后一个目标的下标是 N-1，它需要前 L 个值做历史，
    所以最早的目标下标是 L。
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    X = np.stack([series[i:i + window_size] for i in range(n - window_size)])
    y = series[window_size:]
    target_index = np.arange(window_size, n)
    return X, y, target_index


def split_by_target_time(times, target_index, val_start=VAL_START, test_start=TEST_START):
    """
    按【目标时刻】把样本分成训练/验证/测试（与第六周相同）。

    为什么按目标时刻而不是按样本序号：
    - 时间序列只能"用过去预测未来"，验证/测试目标必须晚于训练目标；
    - 先在整个序列上构造窗口、再按目标时间归类，跨边界的样本仍然拥有
      完整的 24 小时历史（这些历史来自边界之前，是真实观测，不算泄漏）。

    返回 {"train": 掩码, "val": 掩码, "test": 掩码}
    """
    target_times = times[target_index]
    val_start = pd.Timestamp(val_start)
    test_start = pd.Timestamp(test_start)
    return {
        "train": target_times < val_start,
        "val": (target_times >= val_start) & (target_times < test_start),
        "test": target_times >= test_start,
    }


def compute_train_scaler(times, series):
    """
    用【训练时段内的原始 OT】计算均值 μ 和总体标准差 σ（ddof=0）。

    关键点：每个时刻只计一次。
    训练时段 = [2016-07-01, 2017-06-26)，这段时间里每秒（每小时）的油温
    都只作为一个数参与统计。不能把窗口矩阵 X 拍平后再统计——
    同一个油温会同时出现在 24 个窗口里，拍平相当于给中间时刻加权。

    返回 (mu, sigma)，均为 Python float。
    """
    mask = (times >= TRAIN_START) & (times < VAL_START)
    values = series[mask]
    mu = float(values.mean())
    sigma = float(values.std(ddof=0))       # 总体标准差
    assert sigma > 0, "训练时段标准差为 0，无法标准化"
    print(f"  标准化统计量：训练时段 {len(values)} 个原始观测，"
          f"μ = {mu:.4f}，σ = {sigma:.4f}")
    return mu, sigma


def to_sequence_arrays(X, y, mu, sigma):
    """
    把窗口整理成 LSTM 需要的形状并标准化。

    输入 X: (N, L)   → 输出 X_seq: (N, L, 1)  float32
    输入 y: (N,)     → 输出 y_seq: (N, 1)     float32

    为什么多一个维度：nn.LSTM 的输入约定是 (batch, seq_len, input_size)。
    这里 input_size=1（每个时刻只有一个油温），seq_len=24。

    为什么用 float32：PyTorch 的 float32 是 CPU 上的默认类型，
    双精度会让计算变慢，而且本例不需要更高精度。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    X_seq = ((X - mu) / sigma).astype(np.float32)[:, :, None]
    y_seq = ((y - mu) / sigma).astype(np.float32)[:, None]
    return X_seq, y_seq


def inverse_scale(values, mu, sigma):
    """把标准化尺度还原成原始油温尺度：预测值 × σ + μ。"""
    return np.asarray(values, dtype=float) * sigma + mu


def prepare_etth1(n_rows=N_ROWS, window_size=WINDOW_SIZE):
    """
    一次完成数据准备，供两个入口共用。

    返回一个字典：
      df, series, times      : 原始数据与前 n_rows 小时的 OT / 时间
      X, y, target_index     : 窗口样本（原始尺度）
      masks                  : 三组掩码
      target_times           : 每个样本的目标时间
      mu, sigma              : 训练时段的标准化统计量
    """
    df = read_etth1()
    series = df["OT"].iloc[:n_rows].to_numpy(dtype=float)
    times = df["date"].iloc[:n_rows].to_numpy()

    X, y, target_index = make_windows(series, window_size)
    masks = split_by_target_time(times, target_index)
    mu, sigma = compute_train_scaler(times, series)

    return {
        "df": df, "series": series, "times": times,
        "X": X, "y": y, "target_index": target_index,
        "masks": masks, "target_times": times[target_index],
        "mu": mu, "sigma": sigma,
    }
