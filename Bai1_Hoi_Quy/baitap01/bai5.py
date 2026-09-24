# -*- coding: utf-8 -*-
"""Bài tập 5: thử tốc độ học khác."""

import numpy as np
import pandas as pd

df = pd.read_csv("data/gia_nha.csv")

x_goc = df["dien_tich"].to_numpy()
y = df["gia"].to_numpy()

# Chuẩn hóa diện tích
x_tb = x_goc.mean()
x_do_lech = x_goc.std()
x = (x_goc - x_tb) / x_do_lech

w, b = 0.0, 0.0

toc_do_hoc = 1.02
so_vong = 200
n = len(x)

print("Vong       w       b      MSE")

for vong in range(1, so_vong + 1):
    y_du_doan = w * x + b
    chenh_lech = y_du_doan - y

    grad_w = (2 / n) * (chenh_lech * x).sum()
    grad_b = (2 / n) * chenh_lech.sum()

    w -= toc_do_hoc * grad_w
    b -= toc_do_hoc * grad_b

    if vong in (1, 2, 5, 10, 25, 50, 100, 200):
        mse = (((w * x + b) - y) ** 2).mean()
        print(f"{vong:4d} {w:7.4f} {b:7.4f} {mse:8.4f}")

print()

w_goc = w / x_do_lech
b_goc = b - w * x_tb / x_do_lech

print("Sau khi doi ve thang do met vuong:")
print(f"w = {w_goc:.6f}")
print(f"b = {b_goc:.6f}")