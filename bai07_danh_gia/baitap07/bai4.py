# -*- coding: utf-8 -*-
"""Bài 4: vẽ ROC thật của hai mô hình với tham số cố định theo h5."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/sinh_vien_bo_hoc.csv")
X = df.drop(columns="bo_hoc")
y = df["bo_hoc"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
mo_hinh = {
    "Logistic": make_pipeline(StandardScaler(), LogisticRegression()),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=0),
}
duong = {}
fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
for ten, mh in mo_hinh.items():
    xs = mh.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, xs)
    auc = roc_auc_score(y_te, xs)
    duong[ten] = (fpr, tpr, auc)
    ax.plot(fpr, tpr, linewidth=2, label=f"{ten} (AUC = {auc:.4f})")
    print(f"{ten}: AUC={auc:.4f}, {len(fpr)} diem ROC")
ax.plot([0, 1], [0, 1], "--", color="gray", label="Đoán ngẫu nhiên (AUC = 0.5)")
ax.set(xlabel="Tỷ lệ báo động nhầm (FPR)", ylabel="Recall (TPR)",
       title="ROC trên tập kiểm tra: dự đoán sinh viên bỏ học",
       xlim=(-0.015, 1.015), ylim=(-0.015, 1.015))
ax.grid(alpha=0.25)
ax.legend(loc="lower right")
Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/roc.png", dpi=180)
plt.close(fig)
print("Da luu figures/roc.png")

# So thứ tự hai đoạn thẳng ROC ở giữa mọi cặp hoành độ kế tiếp.
# Tránh đánh giá ngay trên đoạn thẳng đứng (một FPR có nhiều TPR).
fpr_lr, tpr_lr, auc_lr = duong["Logistic"]
fpr_rf, tpr_rf, auc_rf = duong["Random Forest"]
moc = np.unique(np.r_[fpr_lr, fpr_rf])
giua = (moc[:-1] + moc[1:]) / 2
lr_giua = np.interp(giua, fpr_lr, tpr_lr)
rf_giua = np.interp(giua, fpr_rf, tpr_rf)
chenh = lr_giua - rf_giua
lr_cao = np.flatnonzero(chenh > 1e-12)
rf_cao = np.flatnonzero(chenh < -1e-12)
cat_nhau = bool(lr_cao.size and rf_cao.size)
print("Hai duong co doi thu tu tren/duoi (cat nhau):", "CO" if cat_nhau else "KHONG")
for nhom in [lr_cao, rf_cao]:
    if nhom.size:
        i = nhom[0]
        print(f"Vi du FPR={giua[i]:.4f}: TPR Logistic={lr_giua[i]:.4f}, RF={rf_giua[i]:.4f}")
ten_cao = "Logistic" if auc_lr > auc_rf else "Random Forest" if auc_rf > auc_lr else "Hai mo hinh"
print(f"{ten_cao} co AUC cao hon (hoac bang neu hoa): {max(auc_lr, auc_rf):.4f}.")
print("AUC cao hon the hien dien tich tong the lon hon, gan goc tren trai hon ve tong the;")
print("khong co nghia duong do luon o tren duong con lai tai moi FPR.")
print("Day la so sanh tren test voi tham so da co dinh, khong dung de dieu chinh mo hinh.")
