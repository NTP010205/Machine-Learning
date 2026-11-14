# -*- coding: utf-8 -*-
"""Bài 5: DBSCAN trên x1/x2; chỉ đánh giá với nhãn thật sau khi fit."""

# Tương thích đường dẫn DLL Unicode trên Windows; không sửa môi trường hocmay.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tuong_thich_windows import sua_duong_dan_dll
sua_duong_dan_dll()

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score


def khop_mot_mot(nhan_that, nhan_cum):
    """Ánh xạ một-một cụm -> lớp tối ưu; nhiễu và cụm dư tính là không khớp.

    Mẫu số luôn là toàn bộ dữ liệu, không bỏ nhiễu để tăng điểm.
    Bài toán gán tổng quát vẫn hợp lệ khi DBSCAN tìm ra khác hai cụm.
    """
    that, cum = np.asarray(nhan_that), np.asarray(nhan_cum)
    hop_le = cum != -1
    if not hop_le.any():
        return 0.0
    bang = pd.crosstab(that[hop_le], cum[hop_le]).to_numpy()
    hang, cot = linear_sum_assignment(-bang)
    return float(bang[hang, cot].sum() / len(that))


df = pd.read_csv("data/hai_vang_trang.csv")
X = df[["x1", "x2"]]
# Chỉ x1, x2 được truyền vào tất cả mô hình, không chuẩn hóa để giữ eps đúng đề.
nhan_km = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)
cac_nhan = {e: DBSCAN(eps=e, min_samples=5).fit_predict(X) for e in [0.2, 0.3, 0.4]}
# Chỉ đọc nhãn thật để đánh giá sau khi các mô hình đã fit xong.
that = df["nhan"].to_numpy()
moc = khop_mot_mot(that, nhan_km)
print(f"KMeans K=2: khop toi uu {moc:.4f}, ARI={adjusted_rand_score(that, nhan_km):.4f}")
print("eps | So cum (bo -1) | So nhieu | Khop mot-mot / 300 | ARI")
for e, nhan in cac_nhan.items():
    so_cum = len(set(nhan) - {-1})
    so_nhieu = int(np.sum(nhan == -1))
    khop = khop_mot_mot(that, nhan)
    print(f"{e:.1f} | {so_cum:14d} | {so_nhieu:8d} | {khop:.4f} | {adjusted_rand_score(that, nhan):.4f}")
    print(f"    {'TOT HON' if khop > moc else 'KHONG TOT HON'} KMeans theo ty le khop mot-mot.")
print("\neps la ban kinh lan can; min_samples=5 tinh ca diem dang xet.")
print("eps qua nho: nhieu diem khong du hang xom, thanh nhieu hoac nhieu cum vuon.")
print("eps qua lon: cac vung mat do noi nhau, hai vang trang co the bi gop lam mot.")
print("DBSCAN co the theo cum cong khi hai vung mat do tach bach; nhieu/noise co the noi cau giua chung.")
print("Khong mac dinh DBSCAN tot hon: doc bang thuc nghiem; khong chon eps moi bang nhan that.")
print("Khong bo diem nhieu khi tinh ty le. ARI la chi so doi chieu them, coi -1 la mot nhom khi tinh ARI.")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 5, figsize=(16, 3.8), layout="constrained", sharex=True, sharey=True)
ds = [("Nhãn thật (chỉ đối chiếu)", that), (f"KMeans K=2\nKhớp={moc:.3f}", nhan_km)]
ds += [(f"DBSCAN eps={e}\n{len(set(n)-{-1})} cụm, {(n == -1).sum()} nhiễu", n) for e, n in cac_nhan.items()]
for ax, (ten, nhan) in zip(axes, ds):
    for k in np.unique(nhan):
        mask = nhan == k
        ax.scatter(X.loc[mask, "x1"], X.loc[mask, "x2"], s=12,
                   color="gray" if k == -1 else plt.get_cmap("tab10")(int(k) % 10),
                   marker="x" if k == -1 else "o")
    ax.set(title=ten, xlabel="x1")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)
axes[0].set_ylabel("x2")
Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/hai_vang_trang.png", dpi=170)
plt.close(fig)
print("Da luu figures/hai_vang_trang.png (mau cum khong co y nghia nhan co dinh).")
