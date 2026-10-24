# -*- coding: utf-8 -*-
"""
run_windows.py —— 实践一入口：把连续序列变成可以训练的样本

用前 12 小时油温（2016-07-01 00:00 ~ 11:00）演示 make_windows：
1. 查看这 12 小时的原始记录，并手动选出一条样本核对。
2. 用 L=3 构造窗口（连续 3 小时预测下一小时），打印形状与首尾样本。
3. 把 window_size 改成 5，对照两组结果的变化。
4. 把两次构造结果保存到 results/window_examples.csv。

运行（在 week06/ 目录下）：python run_windows.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")   # 修 Windows 控制台中文乱码

import data
import reporting


def describe_sample(series, times, target_idx, window_size):
    """打印一条样本的输入时间/数值与目标时间/数值（用于回原始表格核对）。"""
    start = target_idx - window_size
    print(f"    输入：{times[start]} ~ {times[target_idx - 1]}"
          f"（下标 {start}~{target_idx - 1}）")
    print(f"    输入数值：{[round(float(v), 3) for v in series[start:target_idx]]}")
    print(f"    目标：{times[target_idx]}（下标 {target_idx}），"
          f"数值 {float(series[target_idx]):.3f}")


def main():
    # ---- 1. 读前 12 小时并查看 ----
    df = data.read_etth1()
    series = df["OT"].iloc[:12].to_numpy(dtype=float)
    times = df["date"].iloc[:12].to_numpy()

    print("前 12 小时油温记录（2016-07-01 00:00 ~ 11:00）：")
    for t, v in zip(times, series):
        print(f"  {t}  OT = {v:.3f}")
    print("观察：00:00~05:00 油温持续下降（30.5 → 21.2），"
          "06:00~07:00 小幅回升到 23.1，09:00 又降到 17.4，"
          "10:00~11:00 回升到 20.1——中间有回升，不是单调变化。\n")

    # ---- 2. 手动选一条样本（L=3，取第一条：输入 00:00~02:00，目标 03:00）----
    print("手动核对一条样本（L=3，目标取 03:00、下标 3）：")
    describe_sample(series, times, target_idx=3, window_size=3)
    print()

    # ---- 3. L=3 构造窗口 ----
    X3, y3, idx3 = data.make_windows(series, window_size=3)
    print(f"L=3 构造结果：X 形状 {X3.shape}，y 形状 {y3.shape}，"
          f"target_index 形状 {idx3.shape}")
    print(f"  样本数 = {len(y3)}（12 小时、窗口 3 → 12-3 = 9 条）")
    print("  第一条样本：")
    describe_sample(series, times, int(idx3[0]), 3)
    print("  最后一条样本：")
    describe_sample(series, times, int(idx3[-1]), 3)
    print()

    # ---- 4. L=5 构造窗口并对照 ----
    X5, y5, idx5 = data.make_windows(series, window_size=5)
    print(f"L=5 构造结果：X 形状 {X5.shape}，y 形状 {y5.shape}，"
          f"target_index 形状 {idx5.shape}")
    print(f"  样本数 = {len(y5)}（12 小时、窗口 5 → 12-5 = 7 条）")
    print("  第一条样本：")
    describe_sample(series, times, int(idx5[0]), 5)
    print("  最后一条样本：")
    describe_sample(series, times, int(idx5[-1]), 5)
    print()
    print("两组对照：")
    print("  - 样本数：L=3 → 9 条，L=5 → 7 条。窗口越长，可构造的样本越少（N - L）。")
    print("  - 输入覆盖时长：L=3 覆盖 3 小时，L=5 覆盖 5 小时。")
    print("  - 最近一个输入与目标相隔：两组都是 1 小时（预测的都是紧接着的下一小时）。")
    print("  - 两条样本的目标都是 11:00（序列末尾），但输入起点不同（08:00 vs 06:00）。\n")

    # ---- 5. 保存记录 ----
    rows = []
    for ws, X, y, idx in [(3, X3, y3, idx3), (5, X5, y5, idx5)]:
        for i in range(len(idx)):
            rows.append({
                "window_size": ws,
                "input_start": int(idx[i] - ws),          # 窗口第一个值的下标
                "input_end": int(idx[i] - 1),             # 窗口最后一个值的下标
                "target_index": int(idx[i]),              # 目标在原始序列中的下标
                "target_time": str(times[idx[i]]),        # 目标的真实时间
                "input_values": str([round(float(v), 3) for v in X[i]]),  # 列表文本
                "target_value": round(float(y[i]), 3),
            })
    reporting.save_window_examples(rows)
    print(f"\n说明：L=3 有 9 行、L=5 有 7 行，合计 {len(rows)} 行。")


if __name__ == "__main__":
    main()
