# -*- coding: utf-8 -*-
"""
run_lstm.py —— 实践二入口：用 LSTM 学习序列表示并预测下一小时

运行（在 week07/ 目录下）：python run_lstm.py

流程：
1. 数据准备：前 14400 小时、L=24 窗口、按目标时刻划分、训练时段 μ/σ。
2. 训练前先看两组配置的形状与参数数量（H=16 / H=32）。
3. 两组各自独立训练 40 轮，保存验证 MSE 最小的一轮参数。
4. 比较两组最佳参数的验证 RMSE，选定一个 LSTM（相同取 H=16）。
5. 与上周基线 last_value、lag_linear 一起汇总指标；测试只评选定 LSTM。
6. 保存 history / metrics / predictions / 模型文件 / 两张图。
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import torch

import data
import evaluation
import models
import reporting
import training

HIDDEN_SIZES = [16, 32]
SEED = 42
VAL_WEEK_START = pd.Timestamp("2017-06-26 00:00")
VAL_WEEK_END = pd.Timestamp("2017-07-02 23:00")
DEMO_TARGET_TIME = pd.Timestamp("2017-06-28 12:00")


def main():
    # ================= 1. 数据准备 =================
    print("=" * 72)
    print("实践二：LSTM 序列预测（ETTh1，过去 24 小时 → 下一小时）")
    print("=" * 72)

    print("\n【1】数据完整性检查（前 14400 小时）")
    prepared = data.prepare_etth1()
    assert data.check_etth1(prepared["times"], prepared["series"])

    X, y = prepared["X"], prepared["y"]
    target_index = prepared["target_index"]
    times = prepared["times"]
    target_times = prepared["target_times"]
    masks = prepared["masks"]
    mu, sigma = prepared["mu"], prepared["sigma"]

    counts = {k: int(m.sum()) for k, m in masks.items()}
    print(f"\n  窗口 L={data.WINDOW_SIZE}，样本总数 = {len(y)}")
    print(f"    训练 {counts['train']} / 验证 {counts['val']} / 测试 {counts['test']}")
    assert counts["train"] + counts["val"] + counts["test"] == len(y), "样本有遗漏"
    overlap = int((masks["train"] & masks["val"]).sum()
                  + (masks["train"] & masks["test"]).sum()
                  + (masks["val"] & masks["test"]).sum())
    print(f"    三组无重叠：{overlap == 0}（重复归组数 = {overlap}）")

    # 整理成 LSTM 需要的形状并标准化
    X_seq, y_seq = data.to_sequence_arrays(X, y, mu, sigma)
    X_train, y_train = X_seq[masks["train"]], y_seq[masks["train"]]
    X_val, y_val = X_seq[masks["val"]], y_seq[masks["val"]]
    X_test, y_test = X_seq[masks["test"]], y_seq[masks["test"]]

    print(f"\n  形状：X {X_seq.shape}（N, 时间步, 特征数），y {y_seq.shape}；"
          f"dtype {X_seq.dtype}")
    print(f"  标准化：输入与目标都用 (OT - {mu:.4f}) / {sigma:.4f}")

    # 核对一条实际样本：数值与时间要能对上
    k = 1000
    t_idx = int(target_index[k])
    print(f"\n  核对第 {k} 条样本（目标时刻 {str(target_times[k])[:19]}）：")
    print(f"    原始窗口首值 {X[k][0]:.3f}（{str(times[t_idx - 24])[:19]}）→ "
          f"标准化后 {X_seq[k][0, 0]:+.4f}")
    print(f"    原始窗口末值 {X[k][-1]:.3f}（{str(times[t_idx - 1])[:19]}）→ "
          f"标准化后 {X_seq[k][-1, 0]:+.4f}")
    print(f"    原始目标 {y[k]:.3f} → 标准化后 {y_seq[k, 0]:+.4f}")

    # ================= 2. 训练前的形状与参数 =================
    print("\n【2】训练前：两组配置的形状与参数数量")
    sample_batch = torch.from_numpy(X_val[:8])     # 用 8 条真实验证样本
    shape_info = {}

    for hidden in HIDDEN_SIZES:
        torch.manual_seed(SEED)                    # 每组创建模型前都设种子
        model = models.LSTMRegressor(hidden)
        shapes = models.describe_shapes(model, sample_batch)
        same = models.check_last_step_equivalence(model, sample_batch)
        shape_info[hidden] = {"shapes": shapes, "same": same}

        print(f"\n  H = {hidden}")
        print(f"    输入 input      : {shapes['input']}   (batch, 时间步, 特征数)")
        print(f"    序列输出 output : {shapes['output']}   (batch, 时间步, 隐藏维数)")
        print(f"    最终隐藏 h_n    : {shapes['h_n']}   (层数×方向, batch, 隐藏维数)")
        print(f"    最终细胞 c_n    : {shapes['c_n']}   (层数×方向, batch, 隐藏维数)")
        print(f"    预测 pred       : {shapes['pred']}   (batch, 1)")
        print(f"    可训练参数总数  : {shapes['param_count']}"
              f"（LSTM {4 * hidden * (1 + hidden + 1)} + 输出层 {hidden + 1}）")
        print(f"    末步序列输出 == 最终隐藏状态：{same}")

    # ================= 3. 两组分别训练 40 轮 =================
    print("\n【3】两组各自独立训练 40 轮（Adam，lr=0.001，batch=128）")
    histories, bests, trained_models, val_rmse = {}, {}, {}, {}

    for hidden in HIDDEN_SIZES:
        print(f"\n  ---- H = {hidden} ----")
        torch.manual_seed(SEED)                    # 每组建模型前重新设种子
        model = models.LSTMRegressor(hidden)

        train_loader = training.make_loader(X_train, y_train, shuffle=True, seed=SEED)
        train_eval_loader = training.make_loader(X_train, y_train, shuffle=False)
        val_loader = training.make_loader(X_val, y_val, shuffle=False)

        history, best = training.train_lstm(model, train_loader,
                                            train_eval_loader, val_loader)
        histories[hidden] = history
        bests[hidden] = best
        trained_models[hidden] = model

        # 用还原到原始尺度的验证集 RMSE 比较两组
        pred_scaled = training.predict_scaled(model, X_val)
        pred_raw = data.inverse_scale(pred_scaled, mu, sigma)
        _mae, rmse = evaluation.mae_rmse(data.inverse_scale(y_val.reshape(-1), mu, sigma),
                                         pred_raw)
        val_rmse[hidden] = rmse
        print(f"    最佳轮次 {best['epoch']}，验证 MSE（标准化）{best['val_mse']:.6f}，"
              f"验证 RMSE（原始尺度）{rmse:.3f}")

    # ================= 4. 选定一个 LSTM =================
    # 按验证 RMSE 最小选；完全相同时选宽度 16（先比较 rmse，再比较 hidden 是否更小）
    chosen_hidden = min(HIDDEN_SIZES, key=lambda h: (val_rmse[h], h))
    print(f"\n【4】选定 LSTM：H = {chosen_hidden}"
          f"（验证 RMSE {val_rmse[chosen_hidden]:.3f}；"
          f"另一组 {[round(val_rmse[h], 3) for h in HIDDEN_SIZES if h != chosen_hidden][0]}）")

    chosen_model = trained_models[chosen_hidden]
    chosen_best_epoch = bests[chosen_hidden]["epoch"]

    # ================= 5. 汇总指标 =================
    print("\n【5】与上周基线比较")
    # 基线使用【原始油温】和各自的预处理，不套用 LSTM 的 μ/σ
    X_train_raw = X[masks["train"]]
    y_train_raw = y[masks["train"]]
    lag = models.fit_lag_linear(X_train_raw, y_train_raw)

    baseline_preds = {
        "last_value": {
            "train": models.predict_last_value(X[masks["train"]]),
            "val": models.predict_last_value(X[masks["val"]]),
            "test": models.predict_last_value(X[masks["test"]]),
        },
        "lag_linear": {
            "train": lag.predict(X[masks["train"]]),
            "val": lag.predict(X[masks["val"]]),
            "test": lag.predict(X[masks["test"]]),
        },
    }

    # 两组 LSTM 的训练/验证预测（都来自各自最佳轮次）
    lstm_preds = {}
    for hidden in HIDDEN_SIZES:
        m = trained_models[hidden]
        lstm_preds[f"lstm_h{hidden}"] = {
            "train": data.inverse_scale(training.predict_scaled(m, X_train), mu, sigma),
            "val": data.inverse_scale(training.predict_scaled(m, X_val), mu, sigma),
        }
    # 测试只评选定 LSTM
    selected_name = f"lstm_h{chosen_hidden}"
    lstm_preds[selected_name]["test"] = data.inverse_scale(
        training.predict_scaled(chosen_model, X_test), mu, sigma)

    # ---- 生成 metrics 行 ----
    split_arrays = {
        "train": (masks["train"], y[masks["train"]]),
        "val": (masks["val"], y[masks["val"]]),
        "test": (masks["test"], y[masks["test"]]),
    }
    metric_rows = []

    for name in ["last_value", "lag_linear"]:
        for split in ("train", "val", "test"):
            yy = split_arrays[split][1]
            mae, rmse = evaluation.mae_rmse(yy, baseline_preds[name][split])
            metric_rows.append({
                "model": name, "split": split, "n_samples": int(len(yy)),
                "hidden_size": "", "best_epoch": "",
                "mae": round(mae, 3), "rmse": round(rmse, 3), "chosen_lstm": "",
            })

    for hidden in HIDDEN_SIZES:
        name = f"lstm_h{hidden}"
        for split in ("train", "val", "test"):
            if split == "test" and name != selected_name:
                continue                     # 未选中的 LSTM 不评测试
            yy = split_arrays[split][1]
            mae, rmse = evaluation.mae_rmse(yy, lstm_preds[name][split])
            metric_rows.append({
                "model": name, "split": split, "n_samples": int(len(yy)),
                "hidden_size": hidden, "best_epoch": bests[hidden]["epoch"],
                "mae": round(mae, 3), "rmse": round(rmse, 3),
                "chosen_lstm": "是" if name == selected_name else "",
            })

    print("\n  metrics.csv 预览：")
    print(pd.DataFrame(metric_rows).to_string(index=False))

    # ================= 6. 逐条预测 + 核对 =================
    pred_rows = []
    for name in ["last_value", "lag_linear", "lstm_h16", "lstm_h32"]:
        if name.startswith("lstm") and name != selected_name:
            splits = ("train", "val")
        else:
            splits = ("train", "val", "test")
        for split in splits:
            mask = split_arrays[split][0]
            idx = target_index[mask]
            tt = times[idx]
            preds = (baseline_preds[name][split] if name in baseline_preds
                     else lstm_preds[name][split])
            yy = split_arrays[split][1]
            for i in range(len(idx)):
                pred_rows.append({
                    "model": name, "split": split,
                    "target_index": int(idx[i]),
                    "target_time": str(tt[i]),
                    "y_true": float(yy[i]),
                    "y_pred": float(preds[i]),
                })

    print("\n  从逐条预测重算指标，与汇总表核对：")
    verified = evaluation.verify_metrics_from_predictions(pred_rows)
    all_ok = True
    for row in metric_rows:
        mae, rmse = verified[(row["model"], row["split"])]
        ok = abs(mae - row["mae"]) < 0.001 and abs(rmse - row["rmse"]) < 0.001
        all_ok = all_ok and ok
        if not ok:
            print(f"    ✗ {row['model']} / {row['split']} 不一致")
    print(f"    核对结果：{'全部一致 ✓' if all_ok else '存在不一致 ✗'}")

    # 目标没有重复或遗漏
    for name in ["last_value", "lag_linear", "lstm_h16", "lstm_h32"]:
        for split in ("train", "val", "test"):
            sub = [r for r in pred_rows if r["model"] == name and r["split"] == split]
            if not sub:
                continue
            expected = int(split_arrays[split][0].sum())
            unique_targets = len({r["target_index"] for r in sub})
            assert unique_targets == expected == len(sub), f"{name}/{split} 目标不完整"
    print("    目标无重复、无遗漏 ✓")

    # ================= 7. 保存 =================
    print("\n【6】保存结果")
    history_rows = []
    for hidden in HIDDEN_SIZES:
        for r in histories[hidden]:
            history_rows.append({
                "model": f"lstm_h{hidden}", "hidden_size": hidden,
                "epoch": r["epoch"],
                "train_mse": round(r["train_mse"], 6),
                "val_mse": round(r["val_mse"], 6),
            })
    reporting.save_history(history_rows)
    reporting.save_metrics(metric_rows)
    reporting.save_predictions(pred_rows)
    reporting.save_model_bundle(chosen_model, chosen_hidden, chosen_best_epoch,
                                mu, sigma, window_size=data.WINDOW_SIZE)

    reporting.plot_loss_curves(histories, {h: bests[h]["epoch"] for h in HIDDEN_SIZES})

    val_week_rows = [r for r in pred_rows
                     if r["split"] == "val"
                     and VAL_WEEK_START <= pd.Timestamp(r["target_time"]) <= VAL_WEEK_END
                     and r["model"] in ("last_value", "lag_linear", selected_name)]
    print(f"  验证首周预测行数：{len(val_week_rows)}"
          f"（应 = 168 小时 × 3 个模型 = 504）")
    reporting.plot_validation_predictions(val_week_rows, VAL_WEEK_START, VAL_WEEK_END)

    # ================= 8. 手动解释一个目标 =================
    print("\n【7】解释一个具体目标：窗口怎样变成末步表示、再变成预测")
    val_pos = np.where(target_times[masks["val"]] == DEMO_TARGET_TIME)[0]
    if len(val_pos) > 0:
        pos = int(val_pos[0])
        t_idx = int(target_index[masks["val"]][pos])
        # X[i] 就是"目标下标 i+L"那条样本的 24 个小时窗口
        window_raw = X[t_idx - data.WINDOW_SIZE]
        x_one = torch.from_numpy(X_val[pos:pos + 1])

        chosen_model.eval()
        with torch.no_grad():
            pred_scaled, output, h_n, c_n = chosen_model.forward_with_states(x_one)

        print(f"  目标时刻 {DEMO_TARGET_TIME}（原下标 {t_idx}）")
        print(f"    输入窗口：{data.WINDOW_SIZE} 个小时，"
              f"原始值 {window_raw[0]:.2f} → {window_raw[-1]:.2f}")
        print(f"    序列输出 output 形状 {tuple(output.shape)}，"
              f"取最后一步 → {tuple(output[:, -1, :].shape)}")
        print(f"    最终隐藏状态 h_n 形状 {tuple(h_n.shape)}；"
              f"前 5 个分量 {np.round(h_n[-1, 0, :5].numpy(), 3)}")
        print(f"    最终细胞状态 c_n 形状 {tuple(c_n.shape)}（与 h_n 相同）")
        print(f"    输出层前向：pred = 标准化预测 {float(pred_scaled):+.4f}")
        print(f"    还原：{float(pred_scaled) * sigma + mu:.3f}  "
              f"（真实 {y[t_idx]:.3f}）")
        print("    说明：24 个小时被压缩成 H 维的末步表示，"
              "再由一个线性层映射成一个数；")
        print("          这个表示不是某一小时的油温，而是模型为降低预测误差学出来的中间量。")

    print("\n实践二完成！结果都在 week07/results/ 目录下。")


if __name__ == "__main__":
    main()
