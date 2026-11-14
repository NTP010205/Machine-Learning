# -*- coding: utf-8 -*-
"""Bài 6: hồi quy giá nhà, so MAE và ngoại suy ở 250 m²."""

# Cấu hình thực thi cục bộ; không sửa thư viện hoặc tham số mô hình.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cau_hinh_moi_truong import cau_hinh
cau_hinh()

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/gia_nha.csv")
X, y = df[["dien_tich"]], df["gia"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
mo_hinh = {
    "MLPRegressor": make_pipeline(StandardScaler(), MLPRegressor(
        hidden_layer_sizes=(20,), max_iter=5000, random_state=0
    )),
    "LinearRegression": LinearRegression(),
}
moi = pd.DataFrame({"dien_tich": [250.0]})
ket_qua = []
print(f"Train={len(y_tr)}, test={len(y_te)}; chi dung dien_tich, gia tinh bang ty dong.")
print(f"Dien tich train: {X_tr.dien_tich.min():.1f} .. {X_tr.dien_tich.max():.1f} m2.")
print("Mo hinh | MAE test (ty) | Gia du doan 250 m2 (ty)")
for ten, mh in mo_hinh.items():
    mh.fit(X_tr, y_tr)
    mae = mean_absolute_error(y_te, mh.predict(X_te))
    gia = float(mh.predict(moi)[0])
    ket_qua.append((ten, mae, gia))
    print(f"{ten:16s} | {mae:.6f} | {gia:.6f}")
tot = min(ket_qua, key=lambda row: row[1])
print(f"MAE thap hon tren test: {tot[0]} ({tot[1]:.6f} ty).")
print("250 m2 nam ngoai pham vi train; day la ngoai suy, khong co gia that de cham sai so tai 250.")
print("LinearRegression tiep tuc duong thang voi he so goc da hoc; MLP ReLU la ham tuyen tinh tung doan.")
print("Cac trong so MLP hoc de giam loi trong vung du lieu, khong bi rang buoc giu xu huong ngoai vung do.")
print("Vi vay hai du doan ngoai suy co the khac nhau; MAE test thap khong bao dam ngoai suy tot.")
print("Scaler chi fit tren train trong pipeline, du doan test/khach moi dung lai scaler do.")
