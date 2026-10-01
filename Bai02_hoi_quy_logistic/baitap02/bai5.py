# -*- coding: utf-8 -*-
"""Bài tập 5: Dò ngưỡng tốt nhất theo F1."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


df = pd.read_csv("data/sinh_vien.csv")
X = df[["gio_on"]]
y = df["qua_mon"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
mo_hinh = LogisticRegression().fit(X_train, y_train)
p = mo_hinh.predict_proba(X_test)[:, 1]

nguong_list = np.arange(0.05, 1.0, 0.05)
ket_qua = []

print("BAI 5 - DO NGUONG TOT NHAT THEO F1")
print("Nguong    F1")
for nguong in nguong_list:
    y_pred = (p >= nguong).astype(int)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    ket_qua.append((nguong, f1))
    print(f" {nguong:4.2f}   {f1:.4f}")

nguong_tot_nhat, f1_tot_nhat = max(ket_qua, key=lambda x: x[1])
print()
print(f"Nguong cho F1 cao nhat: {nguong_tot_nhat:.2f}")
print(f"F1 cao nhat           : {f1_tot_nhat:.4f}")

if np.isclose(nguong_tot_nhat, 0.5):
    print("Nhan xet: Nguong tot nhat theo F1 dung bang 0.5 trong lan chia du lieu nay.")
else:
    print("Nhan xet: Nguong tot nhat theo F1 khong bang 0.5; 0.5 chi la nguong mac dinh.")
