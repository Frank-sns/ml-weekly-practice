# -*- coding: utf-8 -*-
"""
run_small_rnn.py —— 实践一入口：展开一个小 RNN，观察前向、反向与顺序

运行（在 week07/ 目录下）：python run_small_rnn.py

做的事：
1. 取最前面 6 小时油温，z[k] = (u[k] - u[1]) / 10。
2. 原顺序 [1,2,3,4,5]：五步前向（共享 wx/wh/b）→ 末端算一次损失 → 反向。
3. 打印三个参数梯度、五个输入梯度，并手算第 2 步核对。
4. 顺序对照 [1,4,3,2,5]：只做前向，不反向。
5. 保存 results/rnn_trace.csv（两组各 5 行）。

本实践不调用优化器、不更新参数、没有训练循环。
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")   # 修 Windows 控制台中文乱码

import numpy as np

import data
import reporting
import small_rnn


def main():
    # ---- 1. 准备演示数据 ----
    df = data.read_etth1()
    u, z, times, index = small_rnn.build_demo_data(df)

    print("=" * 70)
    print("实践一：把一个小 RNN 沿时间展开")
    print("=" * 70)
    print("\n【1】最前面 6 小时油温（u[1] ~ u[6]），保留原始下标与时间")
    for k in range(6):
        print(f"  下标 {index[k]:>2}  {str(times[k])[:19]}  u[{k+1}] = {u[k]:8.3f}"
              f"   z[{k+1}] = (u[{k+1}]-u[1])/10 = {z[k]:8.4f}")
    print(f"  输入：z[1] ~ z[5]；目标：z[6] = {z[5]:.4f}")

    # ---- 2. 原顺序：前向 + 反向 ----
    print("\n【2】原顺序 [1,2,3,4,5]：五步前向，参数五步共享")
    print(f"  初始：wx = {small_rnn.WX_INIT}, wh = {small_rnn.WH_INIT}, "
          f"b = {small_rnn.B_INIT}，h[0] = 0")
    print(f"  程序里只创建了 3 个参数张量；五步循环使用的是同一组对象。\n")

    orig_rows, orig_info = small_rnn.run_original(z, z[5], times, index)

    header = f"{'步':>2} {'原下标':>4} {'z':>9} {'h_prev':>10} {'wx*z':>10} {'wh*h_prev':>11} {'h':>10}"
    print(header)
    print("-" * len(header))
    for r in orig_rows:
        print(f"{r['step']:>2} {r['source_index']:>6} {r['z']:>9.5f} {r['h_prev']:>10.5f}"
              f" {r['input_term']:>10.5f} {r['memory_term']:>11.5f} {r['h']:>10.5f}")

    print(f"\n  末端：预测值 = 0.8 × h[5] + 0.1 = {orig_info['pred']:.6f}")
    print(f"        目标   = z[6]            = {orig_info['target']:.6f}")
    print(f"        损失   = (预测 - 目标)^2  = {orig_info['loss']:.6f}")

    # ---- 3. 反向传播：三个参数梯度 + 五个输入梯度 ----
    print("\n【3】只对这一次损失调用 backward()，不调用优化器")
    print(f"  ∂loss/∂wx = {orig_info['wx_grad']:+.6f}")
    print(f"  ∂loss/∂wh = {orig_info['wh_grad']:+.6f}")
    print(f"  ∂loss/∂b  = {orig_info['b_grad']:+.6f}")
    print("\n  五个输入分量的梯度（梯度 = 损失对该输入小变化的敏感程度，"
          "不等于输入值本身）：")
    for r, g in zip(orig_rows, orig_info["input_grads"]):
        print(f"    第 {r['step']} 步（原始下标 {r['source_index']}，z = {r['z']:+.4f}）"
              f"  ∂loss/∂z = {g:+.6f}")

    # 一个容易误判的点：z[1] = 0，但梯度不为 0
    print("\n  注意：z[1] 的值是 0（因为减的基准就是 u[1]），但它的梯度并不为 0——"
          "\n  梯度看的是 tanh 在该点的斜率，不是数值大小。")

    # ---- 4. 手算第 2 步核对 ----
    h2_manual, h2_program, diff = small_rnn.manual_check(orig_rows)
    print("\n【4】手算第 2 步核对")
    print(f"  h[1] = tanh({small_rnn.WX_INIT}×{orig_rows[0]['z']:.5f}"
          f" + {small_rnn.WH_INIT}×0 + {small_rnn.B_INIT}) = {orig_rows[0]['h']:.9f}")
    print(f"  h[2] = tanh({small_rnn.WX_INIT}×{orig_rows[1]['z']:.5f}"
          f" + {small_rnn.WH_INIT}×{orig_rows[0]['h']:.9f} + {small_rnn.B_INIT})")
    print(f"       手算 = {h2_manual:.9f}")
    print(f"       程序 = {h2_program:.9f}")
    print(f"       差值 = {diff:.2e}  → {'一致 ✓' if diff < 1e-9 else '不一致 ✗'}")

    # ---- 5. 顺序对照：只做前向 ----
    print("\n【5】顺序对照 [1,4,3,2,5]：相同参数，从零状态开始，只做前向")
    perm_rows, perm_info = small_rnn.run_permuted(z, z[5], times, index)
    print(header)
    print("-" * len(header))
    for r in perm_rows:
        print(f"{r['step']:>2} {r['source_index']:>6} {r['z']:>9.5f} {r['h_prev']:>10.5f}"
              f" {r['input_term']:>10.5f} {r['memory_term']:>11.5f} {r['h']:>10.5f}")

    print(f"\n  两组对比：")
    print(f"    第 1 步：输入都是 z[1] = {orig_rows[0]['z']:.5f}，"
          f"状态都是 {orig_rows[0]['h']:.5f}（起点相同）")
    print(f"    第 2 步：原顺序输入 z[2] = {orig_rows[1]['z']:+.5f}，"
          f"h_prev = {orig_rows[1]['h_prev']:+.5f}")
    print(f"            对照序输入 z[4] = {perm_rows[1]['z']:+.5f}，"
          f"h_prev = {perm_rows[1]['h_prev']:+.5f}")
    print(f"    第 5 步：两组输入都是 z[5] = {orig_rows[4]['z']:+.5f}，"
          f"但 h_prev 分别 {orig_rows[4]['h_prev']:+.5f} / {perm_rows[4]['h_prev']:+.5f}")
    print(f"\n  末端预测：原顺序 {orig_info['pred']:.6f}，"
          f"对照序 {perm_info['pred']:.6f}，差 {orig_info['pred'] - perm_info['pred']:+.6f}")
    print("  最后一步输入相同、上一状态不同 → 末端状态不同。")
    print("  这说明 RNN 对顺序的利用来自『状态逐步传递』这条计算关系，"
          "而不是只看最后一个输入。")
    if abs(orig_info["pred"] - perm_info["pred"]) < 1e-6:
        print("  （若两者很接近，也不能据此说模型没考虑顺序——"
              "应沿中间状态逐项检查。）")

    # ---- 6. 两个帮助理解的展开 ----
    coef = small_rnn.linear_chain_coefficients()
    print("\n【6】把 tanh 暂时去掉，展开前两步（判断参数固定后预测与输入的关系）")
    print(f"  h[1] = wx·z[1]")
    print(f"  h[2] = wx·z[2] + wh·h[1] = wx·z[2] + (wh·wx)·z[1]")
    print(f"  代入 wx={small_rnn.WX_INIT}, wh={small_rnn.WH_INIT}："
          f"z[1] 的系数 = {coef['z1_coef']:.3f}，z[2] 的系数 = {coef['z2_coef']:.3f}")
    print("  这里只有输入的一次项，没有 z[1]×z[2] 这样的输入乘积。")
    print("  参数之间相乘（wh·wx）是固定常数，不等于对输入有非线性；")
    print("  这也是 MLP 在层与层之间需要激活函数的原因。")

    decay = small_rnn.gradient_decay_example()
    print("\n【7】梯度沿路径连乘的小算例")
    for k, v in decay.items():
        print(f"  {k} = {v:.6g}")
    print("  每步乘小于 1 的数 → 越往前越小（衰减）；乘大于 1 的数 → 可能越来越大。")
    print("  真实 RNN 的局部导数还取决于权重和 tanh 的导数，这里只是示意。")

    # ---- 7. 保存记录 ----
    print("\n【8】保存状态记录")
    for row in orig_rows:
        row["case"] = "original"
    for row in perm_rows:
        row["case"] = "permuted"
    reporting.save_rnn_trace(orig_rows + perm_rows)

    print("\n实践一完成。三参数梯度："
          f"wx {orig_info['wx_grad']:+.4f} / wh {orig_info['wh_grad']:+.4f} / "
          f"b {orig_info['b_grad']:+.4f}；损失 {orig_info['loss']:.4f}。")


if __name__ == "__main__":
    main()
