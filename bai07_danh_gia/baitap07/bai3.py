# -*- coding: utf-8 -*-
"""Bài 3: ngưỡng lớn nhất đạt Recall OOF >= 0.8; test chỉ dùng sau khi chọn."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/sinh_vien_bo_hoc.csv")
X = df.drop(columns="bo_hoc")
y = df["bo_hoc"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
mh = make_pipeline(StandardScaler(), LogisticRegression())
# Scaler được fit riêng trong từng fold, không nhìn phần validation của fold.
xs_hoc = cross_val_predict(mh, X_tr, y_tr, cv=5, method="predict_proba")[:, 1]
muc_tieu = 0.8
# Với quy tắc xs >= t, dự đoán chỉ đổi khi vượt một xác suất quan sát được.
# Dò toàn bộ giá trị phân biệt để tìm ngưỡng lớn nhất, không chỉ lưới bước 0.05.
cac_nguong = np.unique(np.r_[0.0, xs_hoc, 1.0])
recall_hoc = np.array([
    recall_score(y_tr, xs_hoc >= t, zero_division=0) for t in cac_nguong
])
vi_tri = np.flatnonzero(recall_hoc >= muc_tieu)[-1]
nguong = cac_nguong[vi_tri]
print(f"Train: {len(y_tr)} sinh vien, {int(y_tr.sum())} ban bo hoc.")
print(f"Nguong lon nhat: {nguong:.12f} (tinh toan dung day du do chinh xac)")
print(f"Recall OOF: {recall_hoc[vi_tri]:.4f}")
recall_vuot = recall_score(y_tr, xs_hoc >= np.nextafter(nguong, np.inf))
print(f"Chi can vuot nguong nay: Recall OOF = {recall_vuot:.4f} < {muc_tieu}.")
# So sánh cách dò lưới của h4, cũng chỉ dùng train; không dùng để chọn lại test.
luoi = np.round(np.arange(0.05, 0.96, 0.05), 2)
hop_le = [t for t in luoi if recall_score(y_tr, xs_hoc >= t) >= muc_tieu]
if hop_le:
    print(f"Neu gioi han luoi buoc 0.05 nhu h4: nguong lon nhat la {max(hop_le):.2f}.")

# Đã chốt ngưỡng. Chỉ bây giờ mới fit toàn bộ train và dự đoán test một lần.
mh.fit(X_tr, y_tr)
xs_test = mh.predict_proba(X_te)[:, 1]
du = (xs_test >= nguong).astype(int)
p = precision_score(y_te, du, zero_division=0)
r = recall_score(y_te, du, zero_division=0)
f1 = f1_score(y_te, du, zero_division=0)
print(f"Test: Precision={p:.4f}, Recall={r:.4f}, F1={f1:.4f}")
print(f"So sinh vien bi bao: {int(du.sum())}/{len(du)}")
print(f"Recall test {'DAT' if r >= muc_tieu else 'KHONG DAT'} yeu cau >= {muc_tieu}.")
print("\nGiai thich:")
print("Moi xac suat OOF den tu mo hinh chua hoc sinh vien do; ca scaler cung nam trong fold.")
print("Nguong cang cao, so canh bao cang it va Recall khong tang; chon nguong lon nhat hop le.")
print("Dieu kien 80% chi duoc kiem chung tren du doan OOF, khong bao dam cho moi tap moi.")
print("Test la nhung sinh vien khac va mo hinh cuoi hoc tren toan bo train, nen Recall co the khac.")
print(f"Test chi co {int(y_te.sum())} ca duong: mot ca dung/sai lam Recall doi {1/y_te.sum():.4f}.")
print("Neu dung nhan test de sua nguong, test tro thanh tap chon tham so va diem bi lac quan.")
print("Vi vay giu nguyen nguong da chon, bao cao trung thuc ke ca khi Recall test khong dat 0.8.")
