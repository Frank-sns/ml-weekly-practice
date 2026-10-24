# -*- coding: utf-8 -*-
"""
run_baselines.py —— 实践二入口：比较预测基线，完成调参与泛化检查

流程（入口只负责"连接"，具体实现都在各组件里）：
1. 数据准备：前 14400 小时 + 完整性检查 + L=24 窗口 + 按目标时刻划分。
2. 三种基线：last_value（最近观测）、window_mean（窗口均值）、
   lag_linear（标准化+线性回归，学 24 个历史位置的权重）。
3. 所选调参路线：Ridge（三个候选 alpha），在验证 RMSE 上选定一个。
4. 最终四种方法：训练/验证/测试指标；按验证 RMSE 选总体最优（selected）。
5. 保存每个模型、每个目标的逐条预测，并从预测表重算指标与汇总表核对。
6. 输出三张图：调参对照、时段划分、验证首周预测与负载对照。

运行（在 week06/ 目录下）：python run_baselines.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")   # 修 Windows 控制台中文乱码

import numpy as np
import pandas as pd

import data
import models
import evaluation
import reporting


# ========== 集中设置实验参数 ==========
WINDOW_SIZE = 24                        # 用过去 24 小时预测下一小时
VAL_START = pd.Timestamp("2017-06-26")  # 验证起点（目标时刻，含）
TEST_START = pd.Timestamp("2017-10-24") # 测试起点（目标时刻，含）
VAL_WEEK_END = pd.Timestamp("2017-07-02 23:00")  # 验证首周画图截止
ALPHAS = [0.1, 1.0, 10.0]               # Ridge 三个候选：弱/中/强正则化


def main():
    # ---- 1. 数据准备 ----
    df = data.read_etth1()
    series = df["OT"].iloc[:14400].to_numpy(dtype=float)   # 只用前 14400 小时
    times = df["date"].iloc[:14400].to_numpy()
    print("数据完整性检查（前 14400 小时）：")
    assert data.check_etth1(times, series)

    X, y, target_index = data.make_windows(series, WINDOW_SIZE)
    target_times = times[target_index]                     # 每个样本的目标时间
    masks = data.split_by_target_time(times, target_index, VAL_START, TEST_START)
    train_mask, val_mask, test_mask = masks["train"], masks["val"], masks["test"]

    counts = {k: int(m.sum()) for k, m in masks.items()}
    print(f"\n窗口 L={WINDOW_SIZE}，样本总数 = {len(y)}")
    print(f"  训练 {counts['train']} / 验证 {counts['val']} / 测试 {counts['test']}")
    assert counts["train"] + counts["val"] + counts["test"] == len(y), "样本有遗漏"
    overlap = int((train_mask & val_mask).sum() + (train_mask & test_mask).sum()
                  + (val_mask & test_mask).sum())
    print(f"  三组无重叠：{overlap == 0}（重复归组数 = {overlap}）")

    # 第一条验证/测试样本：注意它们的历史窗口来自"更早的时段"（跨边界取历史）
    for split_name, mask in [("验证", val_mask), ("测试", test_mask)]:
        first_pos = int(np.where(mask)[0][0])
        t_idx = int(target_index[first_pos])
        print(f"  第一条{split_name}样本：输入 {times[t_idx - WINDOW_SIZE]} ~ "
              f"{times[t_idx - 1]}（历史来自边界之前），目标 {times[t_idx]}")

    # 选一个目标时刻，回原始序列核对完整输入窗口
    k = 1000
    t_idx = int(target_index[k])
    check_ok = bool(np.allclose(X[k], series[t_idx - WINDOW_SIZE:t_idx]))
    print(f"  核对第 {k} 条样本（目标 {times[t_idx]}）："
          f"输入窗口与原始序列一致 → {check_ok}\n")

    # 切出各段（同一条样本只归一组，对应同一目标下标）
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # ---- 2. 三种基线 ----
    # 固定规则不用训练；线性回归只在训练窗口 fit
    lag = models.fit_lag_linear(X_train, y_train)

    # ---- 3. Ridge 调参：三个候选，同条件对照 ----
    tuning_rows = []
    mae_tr, rmse_tr = evaluation.evaluate_estimator(lag, X_train, y_train)
    mae_va, rmse_va = evaluation.evaluate_estimator(lag, X_val, y_val)
    tuning_rows.append({"model": "lag_linear", "param_name": "", "param_value": "",
                        "train_mae": round(mae_tr, 3), "val_mae": round(mae_va, 3),
                        "train_rmse": round(rmse_tr, 3), "val_rmse": round(rmse_va, 3),
                        "rmse_gap": round(rmse_va - rmse_tr, 3),
                        "chosen_candidate": ""})   # 参照不参加候选选择

    ridge_models = {}
    candidate_metrics = []   # (alpha, val_rmse)，用于选定
    for alpha in ALPHAS:
        est = models.fit_ridge(X_train, y_train, alpha)
        ridge_models[alpha] = est
        mae_tr, rmse_tr = evaluation.evaluate_estimator(est, X_train, y_train)
        mae_va, rmse_va = evaluation.evaluate_estimator(est, X_val, y_val)
        tuning_rows.append({"model": "ridge", "param_name": "alpha",
                            "param_value": alpha,
                            "train_mae": round(mae_tr, 3), "val_mae": round(mae_va, 3),
                            "train_rmse": round(rmse_tr, 3), "val_rmse": round(rmse_va, 3),
                            "rmse_gap": round(rmse_va - rmse_tr, 3),
                            "chosen_candidate": ""})
        candidate_metrics.append((alpha, rmse_va))

    # 选定：验证 RMSE 最小者；完全相同时取较小 alpha（min 按 (rmse, alpha) 排序）
    best_alpha, best_val_rmse = min(candidate_metrics, key=lambda t: (t[1], t[0]))
    for row in tuning_rows:
        if row["model"] == "ridge" and row["param_value"] == best_alpha:
            row["chosen_candidate"] = "是"
    print(f"Ridge 候选对照完成，选定 alpha = {best_alpha}"
          f"（验证 RMSE {best_val_rmse:.3f}）\n")
    ridge_tuned = ridge_models[best_alpha]

    # ---- 4. 最终四种方法 ----
    # (名称, 预测函数)；预测函数都接收窗口 X、返回一维预测
    final_methods = [
        ("last_value", models.predict_last_value),
        ("window_mean", models.predict_window_mean),
        ("lag_linear", lag.predict),
        ("ridge_tuned", ridge_tuned.predict),
    ]
    preds = {name: {"train": fn(X_train), "val": fn(X_val), "test": fn(X_test)}
             for name, fn in final_methods}

    metric_rows = []
    val_rmse_by_model = {}
    for name, _ in final_methods:
        for split, (XX, yy) in [("train", (X_train, y_train)),
                                ("val", (X_val, y_val)),
                                ("test", (X_test, y_test))]:
            mae, rmse = evaluation.mae_rmse(yy, preds[name][split])
            metric_rows.append({"model": name, "split": split,
                                "n_samples": int(len(yy)),
                                "mae": round(mae, 3), "rmse": round(rmse, 3),
                                "selected": ""})
        val_rmse_by_model[name] = metric_rows[-3]["rmse"]   # val 行在每组最后加的是 test，取 val 行

    # 总体最优：按验证 RMSE，完全相同时按顺序保留先出现的（严格小于才替换）
    best_name, best_rmse = None, float("inf")
    for name, _ in final_methods:
        rmse = [r["rmse"] for r in metric_rows
                if r["model"] == name and r["split"] == "val"][0]
        if rmse < best_rmse:
            best_rmse, best_name = rmse, name
    for row in metric_rows:
        if row["model"] == best_name:
            row["selected"] = "是"
    print(f"四种方法按验证 RMSE 选出的总体最优：{best_name}（{best_rmse:.3f}）\n")

    # ---- 5. 逐条预测记录 + 核对 ----
    pred_rows = []
    split_info = [("train", train_mask, y_train),
                  ("val", val_mask, y_val),
                  ("test", test_mask, y_test)]
    for name, _ in final_methods:
        for split_name, mask, yy in split_info:
            idx = target_index[mask]
            tt = times[idx]
            pp = preds[name][split_name]
            for i in range(len(idx)):
                pred_rows.append({
                    "model": name, "split": split_name,
                    "target_index": int(idx[i]),
                    "target_time": str(tt[i]),
                    "y_true": float(yy[i]),
                    "y_pred": float(pp[i]),
                })

    print("从 predictions 重算指标，与汇总表核对：")
    verified = evaluation.verify_metrics_from_predictions(pred_rows)
    all_ok = True
    for row in metric_rows:
        mae, rmse = verified[(row["model"], row["split"])]
        ok = abs(mae - row["mae"]) < 0.001 and abs(rmse - row["rmse"]) < 0.001
        all_ok = all_ok and ok
        if not ok:
            print(f"  ✗ {row['model']} / {row['split']} 不一致")
    print(f"  核对结果：{'全部一致 ✓' if all_ok else '存在不一致 ✗'}\n")

    # ---- 6. 输出 ----
    reporting.save_tuning(tuning_rows)
    reporting.save_metrics(metric_rows)
    reporting.save_predictions(pred_rows)
    reporting.plot_tuning_comparison(tuning_rows)
    reporting.plot_series_split(times, series, VAL_START, TEST_START, WINDOW_SIZE)

    # 验证首周的逐条预测（画图用）
    val_week_rows = []
    for r in pred_rows:
        if r["split"] != "val":
            continue
        t = pd.Timestamp(r["target_time"])
        if VAL_START <= t <= VAL_WEEK_END:
            val_week_rows.append(r)
    print(f"验证首周预测行数：{len(val_week_rows)}（应 = 168 小时 × 4 个模型）")
    hufl_series = df["HUFL"].iloc[:14400].to_numpy(dtype=float)
    reporting.plot_validation_predictions(val_week_rows, hufl_series, times)

    # ---- 7. 手动算一个时刻的两种固定规则（供 README 讲解用）----
    demo_t = pd.Timestamp("2017-06-28 12:00")
    val_t = target_times[val_mask]
    pos = np.where(val_t == demo_t)[0]
    if len(pos) > 0:
        k_val = int(pos[0])
        t_idx = int(target_index[val_mask][k_val])
        print(f"\n手动示例（目标 {demo_t}，下标 {t_idx}）：")
        win = series[t_idx - WINDOW_SIZE:t_idx]
        print(f"  最近 24 小时油温：{[round(float(v), 2) for v in win]}")
        print(f"  last_value 预测 = {win[-1]:.2f}（最近一次观测）")
        print(f"  window_mean 预测 = {win.mean():.2f}（24 小时均值）")
        print(f"  真实值 = {y[val_mask][k_val]:.2f}")

    print("\n全部完成！结果都在 week06/results/ 目录下。")


if __name__ == "__main__":
    main()
