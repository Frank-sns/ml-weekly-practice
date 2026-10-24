# -*- coding: utf-8 -*-
"""
data.py —— 读取序列、窗口构造、按目标时刻划分

职责：
1. 读取 ETTh1.csv（解析 date 为日期时间）。
2. make_windows：把连续序列切成"历史窗口 → 下一小时目标"的训练样本。
3. split_by_target_time：按【目标时刻】把样本分成训练/验证/测试。
4. check_etth1：检查时间递增、无重复、间隔 1 小时、数值有效。
"""
import numpy as np
import pandas as pd

DATA_PATH = "data/ETTh1.csv"


def read_etth1(path=DATA_PATH):
    """读取 ETTh1.csv，date 列解析成日期时间。"""
    return pd.read_csv(path, parse_dates=["date"])


def make_windows(series, window_size):
    """
    把连续序列切成"窗口样本"（本周最核心的数据函数）。

    每条样本 = 连续 L 个历史值（按从旧到新排列）+ 紧接着的下一小时目标。
    例：L=3 时，第一条样本的输入是 series[0:3]（第 0、1、2 小时），
        目标是 series[3]（第 3 小时）；第二条输入是 series[1:4]，目标是 series[4]。
    相当于用一个长度 L 的窗口在序列上"一步一步往前滑"。

    返回：
      X            : (N-L, L)，每行一条样本的历史油温（从旧到新）
      y            : (N-L,)，各行对应的下一小时油温
      target_index : (N-L,)，各个目标在原始序列中的下标（都是 i+L）

    样本数量 = N - L：最后一个可作为目标的时刻是第 N-1 小时，
    它需要前 L 个小时做历史，所以最早的目标是第 L 小时。
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    # 对每个可以作为目标的下标 i+L，取其前 L 个值组成一行输入
    X = np.stack([series[i:i + window_size] for i in range(n - window_size)])
    y = series[window_size:]
    target_index = np.arange(window_size, n)
    return X, y, target_index


def split_by_target_time(times, target_index, val_start, test_start):
    """
    按【目标时刻】把窗口样本分成训练/验证/测试。

    times：原始序列里每个时刻的日期时间（长度 = 原始序列长度）。
    target_index：每个样本的目标在原始序列中的下标。
    所以每个样本的目标时刻 = times[target_index]。

    为什么按"目标时刻"而不是按"样本序号"划分（组会重点）：
    - 时间序列必须模拟真实场景：只能"用过去预测未来"。
    - 按目标时刻划分后，验证/测试目标的时间都晚于训练目标；
      每条样本的输入窗口都是"目标时刻之前"的真实历史，不会用到未来信息。
    - 而且我们先在【整个序列】上构造窗口、再按目标时间归类：
      这样第一条验证目标的输入窗口（来自前一天）是完整的；
      如果先按时间把序列截成三段再分别构造窗口，每段开头的样本会缺历史。

    边界（均为目标时刻，左闭右开）：
      训练 [2016-07-01, 2017-06-26)
      验证 [2017-06-26, 2017-10-24)
      测试 [2017-10-24, 2018-02-21)
    """
    target_times = times[target_index]                     # 每个样本的目标时间
    val_start = pd.Timestamp(val_start)
    test_start = pd.Timestamp(test_start)
    train_mask = target_times < val_start
    val_mask = (target_times >= val_start) & (target_times < test_start)
    test_mask = target_times >= test_start
    return {"train": train_mask, "val": val_mask, "test": test_mask}


def check_etth1(times, series):
    """检查时间递增、无重复、相邻记录相差 1 小时、数值均有效。"""
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


def prepare_etth1(window_size=24, val_start="2017-06-26", test_start="2017-10-24"):
    """
    实践二的完整数据准备：读文件 → 取前 14400 小时 → 构造窗口 → 按目标时刻划分。

    返回：
      X, y, target_index : 窗口样本（见 make_windows）
      times              : 前 14400 小时的日期时间（画图、对齐用）
      masks              : {"train"/"val"/"test": 布尔掩码}
    """
    df = read_etth1()
    series = df["OT"].iloc[:14400].to_numpy(dtype=float)   # 只用前 14400 小时
    times = df["date"].iloc[:14400].to_numpy()
    X, y, target_index = make_windows(series, window_size)
    masks = split_by_target_time(times, target_index, val_start, test_start)
    return X, y, target_index, times, masks
