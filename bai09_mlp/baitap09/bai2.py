# -*- coding: utf-8 -*-
"""Bài 2: kiểm chứng một mạng XOR khác bằng NumPy, không huấn luyện sklearn."""

# Cấu hình thực thi cục bộ; không sửa thư viện hoặc tham số mô hình.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cau_hinh_moi_truong import cau_hinh
cau_hinh()

import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
h1 = np.maximum(0, X[:, 0] - X[:, 1])
h2 = np.maximum(0, X[:, 1] - X[:, 0])
y = h1 + h2
xor_that = np.logical_xor(X[:, 0], X[:, 1]).astype(int)
print("x1 | x2 | h1 | h2 | y | XOR that")
for x, a, b, du, that in zip(X, h1, h2, y, xor_that):
    print(f"{x[0]:2d} | {x[1]:2d} | {a:2d} | {b:2d} | {du:1d} | {that:8d}")
np.testing.assert_array_equal(y, xor_that)
print("Ket qua:", y.tolist(), "- KHOP XOR.")
print("\nGiai thich tung truong hop:")
print("(0,0) va (1,1): x1-x2=x2-x1=0, ca hai ReLU bang 0, tong bang 0.")
print("(1,0): x1-x2=1, x2-x1=-1; ReLU cho h1=1,h2=0, tong bang 1.")
print("(0,1): x1-x2=-1, x2-x1=1; ReLU cho h1=0,h2=1, tong bang 1.")
print("Hai no-ron an phat hien hai chieu khac nhau; lop ra chi cong, khong can sigmoid.")
print("Tong max(0,d)+max(0,-d)=|d| voi d=x1-x2. Tren dau vao nhi phan, |x1-x2| chinh la XOR.")
print("Voi dau vao thuc bat ky, tong nay la khoang cach tuyet doi, khong phai phep XOR Boolean.")
