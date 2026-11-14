# -*- coding: utf-8 -*-
"""Bài 3: 50 seed cho mỗi cách khởi tạo; lưu toàn bộ inertia thực nghiệm."""

# Tương thích đường dẫn DLL Unicode trên Windows; không sửa môi trường hocmay.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tuong_thich_windows import sua_duong_dan_dll
sua_duong_dan_dll()

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/khach_hang.csv")
Xs = StandardScaler().fit_transform(df[["thu_nhap", "diem_chi_tieu"]])
ket_qua = {}
# 'Gần 23.0' là tiêu chí tham chiếu của đề, không thay thế kết quả mô hình.
moc = 23.0
sai_so = 0.05  # Inertia 23.02 được làm tròn một chữ số thành 23.0.
for khoi_tao in ["random", "k-means++"]:
    ket_qua[khoi_tao] = np.array([
        KMeans(n_clusters=5, init=khoi_tao, n_init=1, random_state=rs).fit(Xs).inertia_
        for rs in range(50)
    ])
bang = pd.DataFrame(ket_qua)
bang.index.name = "random_state"
print("Inertia day du cua 50 seed (khong sap xep lai seed):")
print(bang.to_string(float_format=lambda v: f"{v:.6f}"))
print(f"\nTieu chi dat: |inertia - {moc}| <= {sai_so}, rtol=0.")
print("Khoi tao   | So lan dat/50 | Nho nhat | Lon nhat | Trung binh")
for ten, day in ket_qua.items():
    dat = np.isclose(day, moc, atol=sai_so, rtol=0)
    print(f"{ten:10s} | {int(dat.sum()):2d}/50         | {day.min():.6f} | {day.max():.6f} | {day.mean():.6f}")
Path("figures").mkdir(exist_ok=True)
bang.to_csv("figures/khoi_tao_50_lan.csv", float_format="%.12f")
print("Da luu figures/khoi_tao_50_lan.csv (du lieu thuc nghiem, khong phai dataset thay the).")
print("\nNhan xet:")
print("Random co the chon nhieu tam trong cung mot nhom, bo trong nhom khac.")
print("Hai buoc gan/dich tam chi cai thien cuc bo, nen co the dung o cuc tieu kem.")
print("K-means++ uu tien diem xa cac tam da chon (theo binh phuong khoang cach),")
print("nen thuong phu cac nhom tot hon va giam rui ro khoi tao xau; khong bao dam toi uu toan cuc.")
print("n_init la so lan khoi tao trong MOT lan fit; thu vien giu ket qua co inertia nho nhat.")
print("Thi nghiem giu n_init=1 de tach tac dong khoi tao. n_init=10 tang chi phi nhung giam rui ro.")
print("random_state chi co dinh chuoi ngau nhien de lap lai, khong tu lam ket qua tot hon.")
print("Khong dung nhan that; so lan dat o day la thuc nghiem tren 50 seed, khong phai bao dam cho moi du lieu.")
