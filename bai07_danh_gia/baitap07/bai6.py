# -*- coding: utf-8 -*-
"""Bài 6: thêm đúng 5 tỷ vào một bản sao dự đoán, giữ nguyên dữ liệu/mô hình."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/gia_nha.csv")
X = df[["dien_tich"]]
y = df["gia"]  # Đơn vị trong CSV: tỷ đồng.
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
mh = LinearRegression()
du = mh.fit(X_tr, y_tr).predict(X_te)
du_hong = du.copy()
# Chọn trước căn đầu tiên của test, không chọn căn dựa vào sai số đã quan sát.
vi_tri = 0
du_hong[vi_tri] += 5.0
assert np.count_nonzero(du_hong != du) == 1
assert np.isclose(du_hong[vi_tri] - du[vi_tri], 5.0)
print(f"Train={len(y_tr)}, test={len(y_te)} can nha. Don vi gia/MAE/RMSE: ty dong.")
print(f"Can dau test (vi tri {vi_tri}, index CSV {y_te.index[vi_tri]}): "
      f"that={y_te.iloc[vi_tri]:.4f}, doan={du[vi_tri]:.4f}, doan hong={du_hong[vi_tri]:.4f}")
print("Truong hop | MAE | RMSE | R2")
ket_qua = []
for ten, du_doan in [("Ban dau", du), ("Mot can +5 ty", du_hong)]:
    diem = np.array([
        mean_absolute_error(y_te, du_doan),
        np.sqrt(mean_squared_error(y_te, du_doan)),
        r2_score(y_te, du_doan),
    ])
    ket_qua.append(diem)
    print(f"{ten:14s} | " + " | ".join(f"{v:.4f}" for v in diem))
delta = ket_qua[1] - ket_qua[0]
print(f"Thay doi: MAE={delta[0]:+.4f} ty; RMSE={delta[1]:+.4f} ty; R2={delta[2]:+.4f}.")
print(f"MAE gap {ket_qua[1][0]/ket_qua[0][0]:.2f} lan; RMSE gap {ket_qua[1][1]/ket_qua[0][1]:.2f} lan.")
print("MAE=mean(|sai so|); RMSE=sqrt(mean(sai so**2)), nen loi lon bi binh phuong truoc.")
print("Vi vay mot du doan hong tac dong manh den RMSE; RMSE va MAE cung don vi ty dong.")
print("R2=1-SSE/SST: SST khong doi, SSE tang nen R2 giam; R2 khong co don vi.")
print("Khong so truc tiep do lon thay doi R2 voi MAE/RMSE vi khac thang do.")
print("Chi thay doi ban sao du doan; CSV, mo hinh va mang du doan ban dau duoc giu nguyen.")
