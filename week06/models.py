# -*- coding: utf-8 -*-
"""
models.py —— 固定预测规则、线性回归及所选调参路线（Ridge）

本周四个方法分两类：
- 固定规则（不学参数，直接按当前窗口算）：
    last_value  ：ŷ[t] = series[t-1]（最近一次观测）
    window_mean ：ŷ[t] = 最近 24 次观测的均值
- 统计模型（只在训练集 fit 学参数）：
    lag_linear  ：标准化 + 线性回归（学习不同历史位置的权重）
    ridge_tuned ：标准化 + Ridge（在 lag_linear 基础上加 L2 正则，alpha 调参）

线性模型的公式（z 是用训练统计量标准化后的窗口）：
    ŷ[t] = b + w[0]*z[t,0] + w[1]*z[t,1] + … + w[23]*z[t,23]
即把过去 24 个小时的油温当作 24 个特征，学出每个历史位置对下一小时
油温的"权重"。注意：这些权重是统计拟合结果，不等于设备的物理传热参数。
"""
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def predict_last_value(X):
    """
    上一时刻预测：ŷ[t] = series[t-1]。

    输入 X 的列按从旧到新排列，所以"最近一次观测"就是每行的最后一列。
    这是"朴素预测"：假设下一小时油温≈当前油温。油温有热惯性、
    短期变化往往不大，所以这个方法常常已经不错。
    它不学任何参数，直接依据当前窗口计算。
    """
    return X[:, -1]


def predict_window_mean(X):
    """
    窗口均值预测：ŷ[t] = (series[t-24] + … + series[t-1]) / 24。

    对每条窗口算均值。连续升温时，把较早的低温一起平均，预测会
    低于最近观测；连续降温时则相反，预测会高于最近观测。
    同样不学参数。
    """
    return X.mean(axis=1)


def fit_lag_linear(X_train, y_train):
    """
    滞后特征线性回归：标准化 + LinearRegression 的管道。

    "滞后特征"：同一个变量（油温）的多个历史值，每个历史位置成为
    一个独立的特征列。X 的 24 列 = 24 个特征，LinearRegression 为
    每个位置学一个权重 w[i] 加偏置 b。

    管道只在训练集 fit：StandardScaler 在训练窗口上算均值/标准差，
    线性回归在标准化后的训练窗口上学权重；验证/测试用同一管道 predict，
    内部自动用训练统计量标准化，保证不泄漏。
    """
    return make_pipeline(StandardScaler(), LinearRegression()).fit(X_train, y_train)


def fit_ridge(X_train, y_train, alpha):
    """
    Ridge 路线：标准化 + Ridge(alpha) 的管道。

    alpha 控制 L2 正则强度：约束权重平方和（w 不能太"大"）。
    - alpha 越大，约束越强，权重越接近 0，模型越"简单"（防过拟合，但可能欠拟合）；
    - alpha 越小，越接近普通线性回归。
    滞后特征之间高度相关（相邻小时油温差不多），Ridge 的约束有助于
    让权重更平滑、更稳定。

    参数含义与第三周一致；这里通过候选 alpha 调参。
    """
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(X_train, y_train)
