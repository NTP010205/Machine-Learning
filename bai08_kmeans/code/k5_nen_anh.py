# -*- coding: utf-8 -*-
"""Bước 5: nén màu một bức ảnh bằng K-Means."""

# Tương thích đường dẫn Unicode Windows; xem README, không đổi thuật toán PDF.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tuong_thich_windows import sua_duong_dan_dll
sua_duong_dan_dll()
# Kết thúc phần tương thích; mã mẫu PDF tiếp tục bên dưới.

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image

anh = load_sample_image( "china.jpg") # ảnh có sẵn trong scikit-learn
cao, rong, _ = anh.shape
diem = anh.reshape(- 1, 3) / 255.0 # mỗi dòng là một điểm ảnh RGB
so_mau = len(np.unique(anh.reshape(-1, 3), axis= 0))
print(f"Anh {cao} x {rong}, co {cao * rong} diem anh, {so_mau} mau" )
print()

# Học trên 5000 điểm ảnh rút ngẫu nhiên cho nhanh, rồi gán cho mọi điểm
rng = np.random.default_rng( 0)
mau = diem[rng.choice( len(diem), 5000, replace= False)]

print(" K so bit moi diem nho hon sai lech trung binh" )
for k in [4, 16, 64]:
    km = KMeans(n_clusters=k, n_init= 4, random_state= 0).fit(mau)
    nhan = km.predict(diem)
    moi = km.cluster_centers_[nhan]
    bit = int(np.ceil(np.log2(k)))
    sai = np. abs(moi - diem).mean() * 255
    print(f" {k:2d} {bit} {24 / bit:4.1f} lan"
          f" {sai:5.2f} muc sang" )
