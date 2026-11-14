# -*- coding: utf-8 -*-
"""Bài 5: cân bằng lớp cho RF/SVC, giữ cùng ngưỡng xác suất 0.5 như h5."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

df = pd.read_csv("data/sinh_vien_bo_hoc.csv")
X = df.drop(columns="bo_hoc")
y = df["bo_hoc"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
ket_qua = {}
print("Giu quy tac predict_proba >= 0.5 nhu h5, ke ca SVC.")
print("Mo hinh | Can bang | Precision | Recall | F1")
for ten in ["Random Forest", "SVC RBF"]:
    for trong_so in [None, "balanced"]:
        if ten == "Random Forest":
            mh = RandomForestClassifier(n_estimators=200, random_state=0, class_weight=trong_so)
        else:
            mh = make_pipeline(StandardScaler(), SVC(
                kernel="rbf", probability=True, random_state=0, class_weight=trong_so
            ))
        mh.fit(X_tr, y_tr)
        du = (mh.predict_proba(X_te)[:, 1] >= 0.5).astype(int)
        diem = np.array([
            precision_score(y_te, du, zero_division=0),
            recall_score(y_te, du, zero_division=0),
            f1_score(y_te, du, zero_division=0),
        ])
        ket_qua[(ten, trong_so)] = diem
        print(f"{ten:13s} | {str(trong_so):8s} | " + " | ".join(f"{v:.4f}" for v in diem))

print("\nThay doi (balanced - thuong):")
do_doi = {}
for ten in ["Random Forest", "SVC RBF"]:
    delta = ket_qua[(ten, "balanced")] - ket_qua[(ten, None)]
    do_doi[ten] = float(np.abs(delta).sum())
    print(f"{ten}: delta Precision={delta[0]:+.4f}, Recall={delta[1]:+.4f}, F1={delta[2]:+.4f}")
lon_nhat = max(do_doi.values())
ten_doi = [ten for ten, v in do_doi.items() if np.isclose(v, lon_nhat)]
print("Thay doi nhieu hon theo tong |delta P|+|delta R|+|delta F1|:", ", ".join(ten_doi))
print("Balanced tang trong so loi lop hiem khi hoc, khong bao dam Recall test luon tang.")
print("SVC hieu chinh xac suat khi probability=True: nguong xac suat 0.5 co the khac predict().")
print("Vi vay khong tron hai quy tac du doan khi so sanh, va khong chon lai nguong tren test.")
