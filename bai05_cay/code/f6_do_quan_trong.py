# -*- coding: utf-8 -*-
"""Bước 6: đọc độ quan trọng của đặc trưng, và thấy giới hạn của nó."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv( "data/khach_vay.csv")
X = df.drop(columns= "tra_dung_han")
y = df[ "tra_dung_han"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size= 0.25, random_state= 17, stratify=y
)

rung = RandomForestClassifier(n_estimators= 200, random_state= 0)
rung.fit(X_tr, y_tr)
qt = pd.Series(rung.feature_importances_, index=X.columns)
print("Do quan trong, bon cot that" )
print(qt.sort_values(ascending=False).round(3).to_string())
print("Tong:", round(qt.sum(), 3))
print()

# Thêm một cột toàn số ngẫu nhiên, chẳng liên quan gì tới việc trả nợ
rng = np.random.default_rng( 0)
X_tr_nn = X_tr.copy()
X_tr_nn["so_ngau_nhien"] = rng.random( len(X_tr))
rung_nn = RandomForestClassifier(n_estimators= 200, random_state= 0)
rung_nn.fit(X_tr_nn, y_tr)
qt_nn = pd.Series(rung_nn.feature_importances_, index=X_tr_nn.columns)
print("Do quan trong, them mot cot ngau nhien" )
print(qt_nn.sort_values(ascending=False).round(3).to_string())

# Nhận xét bổ sung theo yêu cầu; mã hướng dẫn phía trên giữ theo PDF.
print("\nNHAN XET:")
print("feature_importances_ cong muc giam Gini tren train, chuan hoa tong bang 1.")
print("Cot co nhieu gia tri khac nhau co nhieu nguong de thu, de tach nhieu tren train.")
print("Vi vay cot ngau nhien van co the duoc danh gia cao du khong giup du doan moi.")
print("Do quan trong khong chung minh quan he nhan qua; can doi chieu permutation")
print("importance tren test nhu Bai 5, khong chi nhin feature_importances_.")
