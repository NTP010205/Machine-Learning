# -*- coding: utf-8 -*-
"""Bài 3: so ba activation với cấu hình cố định theo m3."""

# Cấu hình thực thi cục bộ; không sửa thư viện hoặc tham số mô hình.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cau_hinh_moi_truong import cau_hinh
cau_hinh()

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/hai_vang_trang.csv")
X, y = df[["x1", "x2"]], df["nhan"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
ket_qua = []
print("Activation | Accuracy train | Accuracy test", flush=True)
for activation in ["logistic", "tanh", "relu"]:
    mh = make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(100, 100), alpha=1.0, activation=activation,
        solver="lbfgs", max_iter=20000, random_state=0
    ))
    mh.fit(X_tr, y_tr)
    train = accuracy_score(y_tr, mh.predict(X_tr))
    test = accuracy_score(y_te, mh.predict(X_te))
    ket_qua.append((activation, train, test))
    print(f"{activation:10s} | {train:.4f}         | {test:.4f}", flush=True)
tot_nhat = max(row[2] for row in ket_qua)
ten = [row[0] for row in ket_qua if row[2] == tot_nhat]
print("Accuracy test cao nhat trong thi nghiem:", ", ".join(ten), f"({tot_nhat:.4f}).")
print("Sigmoid/logistic nam trong (0,1), tanh trong (-1,1); hai ham co the bao hoa o hai dau.")
print("ReLU giu phan duong, cat phan am; tao ham tung doan va khong bao hoa o mien duong.")
print("Activation doi ca bieu dien va qua trinh toi uu, nen cung so no-ron khong cho cung ket qua.")
print("Day la nhan xet ba cau hinh da dinh truoc; khong tiep tuc sua tham so theo test.")
