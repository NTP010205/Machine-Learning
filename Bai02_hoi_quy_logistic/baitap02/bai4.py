# -*- coding: utf-8 -*-
"""Bài tập 4: Tự tính accuracy, precision, recall và F1 từ TP, TN, FP, FN."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


df = pd.read_csv("data/sinh_vien.csv")
X = df[["gio_on"]]
y = df["qua_mon"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
mo_hinh = LogisticRegression().fit(X_train, y_train)
y_pred = mo_hinh.predict(X_test)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) else 0.0
recall = tp / (tp + fn) if (tp + fn) else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

print("BAI 4 - TU TINH BON THUOC DO")
print(f"TN = {tn}, FP = {fp}, FN = {fn}, TP = {tp}")
print()
print(f"Accuracy  = ({tp} + {tn}) / {tp + tn + fp + fn} = {accuracy:.4f}")
print(f"Precision = {tp} / ({tp} + {fp}) = {precision:.4f}")
print(f"Recall    = {tp} / ({tp} + {fn}) = {recall:.4f}")
print(f"F1        = 2 * {precision:.4f} * {recall:.4f} / ({precision:.4f} + {recall:.4f}) = {f1:.4f}")
print()
print("Doi chieu voi tai lieu:")
print("  Accuracy  = 0.8667")
print("  Precision = 0.8500")
print("  Recall    = 0.9444")
print("  F1        = 0.8947")
print("Neu dung dung sinh_vien.csv, random_state=17 va stratify=y thi hai ben phai khop.")
