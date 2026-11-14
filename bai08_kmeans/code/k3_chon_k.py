# -*- coding: utf-8 -*-
"""Bước 3: chọn số cụm K bằng khuỷu tay và chỉ số silhouette."""

# Tương thích đường dẫn Unicode Windows; xem README, không đổi thuật toán PDF.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tuong_thich_windows import sua_duong_dan_dll
sua_duong_dan_dll()
# Kết thúc phần tương thích; mã mẫu PDF tiếp tục bên dưới.

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv( "data/khach_hang.csv")
Xs = StandardScaler().fit_transform(df[[ "thu_nhap", "diem_chi_tieu"]])

print(" K inertia giam so voi K-1 silhouette" )
truoc = None
for k in range(1, 11):
    km = KMeans(n_clusters=k, n_init= 10, random_state= 0).fit(Xs)
    giam = "" if truoc is None else f"{truoc - km.inertia_:8.1f}"
    # Silhouette cần ít nhất hai cụm
    sil = "" if k == 1 else f"{silhouette_score(Xs, km.labels_):.3f}"
    print(f" {k:2d} {km.inertia_:7.1f} {giam:>15s} {sil:>10s}" )
    truoc = km.inertia_
