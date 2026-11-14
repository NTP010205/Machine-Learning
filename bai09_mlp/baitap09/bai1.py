# -*- coding: utf-8 -*-
"""Bài 1: tự lan truyền tiến từ trọng số đã học, rồi mới đối chiếu sklearn."""

# Cấu hình thực thi cục bộ; không sửa thư viện hoặc tham số mô hình.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cau_hinh_moi_truong import cau_hinh
cau_hinh()

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def lan_truyen_tien(X, W1, b1, W2, b2):
    """Chỉ dùng NumPy: affine -> ReLU -> affine -> sigmoid."""
    Z1 = np.asarray(X, dtype=float) @ W1 + b1
    H = np.maximum(0, Z1)
    Z2 = H @ W2 + b2
    # Sigmoid viết hai nhánh tương đương để tránh overflow khi z âm rất lớn.
    P = np.empty_like(Z2)
    duong = Z2 >= 0
    P[duong] = 1 / (1 + np.exp(-Z2[duong]))
    ez = np.exp(Z2[~duong])
    P[~duong] = ez / (1 + ez)
    return Z1, H, Z2, P[:, 0]


def main():
    df = pd.read_csv("data/xor.csv")
    X, y = df[["x1", "x2"]], df["nhan"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )
    mlp = MLPClassifier(hidden_layer_sizes=(8,), solver="lbfgs", max_iter=5000, random_state=0)
    mlp.fit(X_tr, y_tr)
    W1, W2 = mlp.coefs_
    b1, b2 = mlp.intercepts_
    nam_diem = X_te.iloc[:5]
    Z1, H, Z2, tu_tinh = lan_truyen_tien(nam_diem.to_numpy(), W1, b1, W2, b2)
    # Chỉ gọi predict_proba ở bước đối chứng, không dùng trong phép tính NumPy.
    doi_chung = mlp.predict_proba(nam_diem)[:, 1]
    bang = nam_diem.copy()
    bang["p_tu_tinh"] = tu_tinh
    bang["p_sklearn"] = doi_chung
    bang["sai_lech"] = np.abs(tu_tinh - doi_chung)
    print("Nam diem dau tap test, giu index CSV ban dau:")
    print(bang.to_string(float_format=lambda v: f"{v:.12f}"))
    np.testing.assert_allclose(tu_tinh, doi_chung, atol=1e-12, rtol=1e-12)
    print(f"Sai lech lon nhat: {np.max(np.abs(tu_tinh-doi_chung)):.3e}; KHOP trong sai so 1e-12.")
    print(f"\nKich thuoc W1={W1.shape}, b1={b1.shape}, W2={W2.shape}, b2={b2.shape}")
    print("Buoc 1: X(5,2) @ W1(2,8) + b1(8,) -> Z1(5,8), bias cong vao tung hang.")
    print("Buoc 2: H=maximum(0,Z1), gia tri am thanh 0, gia tri duong giu nguyen.")
    print("Buoc 3: Z2=H(5,8) @ W2(8,1) + b2(1,) -> (5,1).")
    print("Buoc 4: p=1/(1+exp(-Z2)), la xac suat lop 1; lop 0 co xac suat 1-p.")
    print("Hai nhanh sigmoid chi tranh tran so; khong doi cong thuc hay cat xac suat.")
    print("\nChi tiet diem dau:")
    print("Z1 =", np.round(Z1[0], 6))
    print("H  =", np.round(H[0], 6))
    print(f"Z2 = {Z2[0, 0]:.9f}; sigmoid(Z2) = {tu_tinh[0]:.12f}")
    print("Khong chuan hoa o bai nay vi m2 hoc truc tiep tren x1,x2. Dung dung thu tu cot.")
    print("Trong so o day duoc HOC tu CSV, khac mang XOR bon diem co trong so tu dat o m1.")


if __name__ == "__main__":
    main()
