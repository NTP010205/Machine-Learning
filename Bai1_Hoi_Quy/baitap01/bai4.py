# -*- coding: utf-8 -*-
"""Bài tập 4: thêm số phòng vào mô hình."""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/gia_nha.csv")

X = df[["dien_tich", "so_phong"]]
y = df["gia"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mo_hinh = LinearRegression()
mo_hinh.fit(X_train, y_train)

y_pred = mo_hinh.predict(X_test)

r2 = r2_score(y_test, y_pred)

print("So can de hoc:", len(X_train))
print("So can de kiem tra:", len(X_test))
print(f"R2 = {r2:.4f}")
print()

print("R2 cua mo hinh mot bien = 0.9622")
print(f"R2 cua mo hinh hai bien = {r2:.4f}")
print("Nhan xet: Them so_phong lam R2 tang len,"
      " nhung muc cai thien khong lon.")