# -*- coding: utf-8 -*-
"""Bài tập 6: Đổi lớp dương sang lớp rớt môn (0) rồi chấm lại."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split


df = pd.read_csv("data/sinh_vien.csv")
X = df[["gio_on"]]
y = df["qua_mon"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
mo_hinh = LogisticRegression().fit(X_train, y_train)
y_pred = mo_hinh.predict(X_test)

precision_lop_1 = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
recall_lop_1 = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
precision_lop_0 = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
recall_lop_0 = recall_score(y_test, y_pred, pos_label=0, zero_division=0)

print("BAI 6 - DOI LOP DUONG")
print("Lop 1 = qua mon")
print(f"  Precision = {precision_lop_1:.4f}")
print(f"  Recall    = {recall_lop_1:.4f}")
print()
print("Lop 0 = rot mon")
print(f"  Precision = {precision_lop_0:.4f}")
print(f"  Recall    = {recall_lop_0:.4f}")
print()
print("Giai thich:")
print(
    "Khi doi pos_label, y nghia cua 'positive' doi theo. Precision va recall luc nay "
    "duoc tinh tren nhom rot mon thay vi nhom qua mon. Ma tran du doan khong doi, "
    "nhung tu so va mau so trong hai cong thuc doi, nen cac gia tri khac nhau."
)
