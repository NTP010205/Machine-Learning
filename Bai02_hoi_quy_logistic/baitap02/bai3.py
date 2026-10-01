# -*- coding: utf-8 -*-
"""Bài tập 3: Dự đoán cho một bạn cụ thể."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


df = pd.read_csv("data/sinh_vien.csv")
X = df[["gio_on"]]
y = df["qua_mon"]

mo_hinh = LogisticRegression().fit(X, y)
w = float(mo_hinh.coef_[0][0])
b = float(mo_hinh.intercept_[0])


def du_doan(gio):
    z = w * gio + b
    p = sigmoid(z)
    nhan = 1 if p >= 0.5 else 0
    print(f"Gio on = {gio:5.2f} | z = {z:+.6f} | P(qua) = {p:.6f} | nhan = {nhan}")


print("BAI 3 - DU DOAN CHO MOT BAN CU THE")
print(f"Mo hinh: z = {w:.6f} * gio_on + ({b:.6f})")
print()
for gio in [3, 8, 12.89, 18, 26]:
    du_doan(gio)

moc_05 = -b / w
print()
print(f"Moc xac suat 0.5 theo mo hinh la {-b / w:.4f} gio.")
print(
    "Giai thich: 12.89 gio rat gan moc -b/w, khi do z gan 0; "
    "ma sigmoid(0) = 0.5, nen xac suat ra gan 0.5."
)
