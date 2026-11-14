# -*- coding: utf-8 -*-
"""Bài 2: thêm tuổi, chọn K bằng silhouette và so phân hoạch với hai đặc trưng."""

# Tương thích đường dẫn DLL Unicode trên Windows; không sửa môi trường hocmay.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tuong_thich_windows import sua_duong_dan_dll
sua_duong_dan_dll()

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/khach_hang.csv")
cot = ["tuoi", "thu_nhap", "diem_chi_tieu"]
sc = StandardScaler().fit(df[cot])
Xs = sc.transform(df[cot])
bang = []
mo_hinh = {}
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xs)
    mo_hinh[k] = km
    bang.append((k, km.inertia_, silhouette_score(Xs, km.labels_)))
bang = pd.DataFrame(bang, columns=["K", "Inertia", "Silhouette"])
print("BA COT: tuoi, thu_nhap, diem_chi_tieu")
print(bang.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
k_tot = int(bang.loc[bang["Silhouette"].idxmax(), "K"])
km = mo_hinh[k_tot]
tam = pd.DataFrame(sc.inverse_transform(km.cluster_centers_), columns=cot)
tam["so_khach"] = np.bincount(km.labels_, minlength=k_tot)
print(f"\nK tot nhat={k_tot}; con bang 5: {k_tot == 5}")
print("Tam cum ba cot, don vi goc:")
print(tam.round(3).to_string())

sc2 = StandardScaler().fit(df[["thu_nhap", "diem_chi_tieu"]])
Xs2 = sc2.transform(df[["thu_nhap", "diem_chi_tieu"]])
bang2 = []
for k in range(1, 11):
    mh2 = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xs2)
    bang2.append((k, mh2.inertia_, silhouette_score(Xs2, mh2.labels_) if k >= 2 else np.nan))
    if k == 5:
        km2 = mh2
bang2 = pd.DataFrame(bang2, columns=["K", "Inertia", "Silhouette"])
tam2 = pd.DataFrame(sc2.inverse_transform(km2.cluster_centers_), columns=["thu_nhap", "diem_chi_tieu"])
tam2["so_khach"] = np.bincount(km2.labels_)
tam2["tuoi_tb"] = df.groupby(km2.labels_)["tuoi"].mean()
print("\nHAI COT, K=5 (moc so sanh muc 3):")
print(tam2.round(3).to_string())
print("Bang giao giua hai cach chia (chi doi chieu SAU khi fit):")
print(pd.crosstab(pd.Series(km2.labels_, name="cum_2D"),
                  pd.Series(km.labels_, name="cum_3D")).to_string())
print(f"ARI giua hai phan hoach, khong phu thuoc so hieu cum: {adjusted_rand_score(km2.labels_, km.labels_):.6f}")
for ten, nhan in [("Hai cot", km2.labels_), ("Ba cot", km.labels_)]:
    tuoi_tb = df.groupby(nhan)["tuoi"].transform("mean")
    lech = np.sqrt(np.mean((df["tuoi"] - tuoi_tb) ** 2))
    print(f"{ten}: RMS do lech tuoi so voi trung binh cum={lech:.3f} tuoi")
print("Tuoi duoc chuan hoa nen co vai tro trong khoang cach nhu hai cot con lai.")
print("Bang tam va bang giao cho thay nhom nao tach/gop/doi thanh vien, khong so nhan cum truc tiep.")
print("RMS tuoi trong cum nho hon cho thay khach dong deu ve tuoi hon; khong tu dong la phan khuc tot hon.")
print("Silhouette cua khong gian 2D va 3D dung khoang cach khac nhau; khong so inertia truc tiep giua hai khong gian.")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
for b, ten in [(bang2, "2 đặc trưng"), (bang, "3 đặc trưng (thêm tuổi)")]:
    axes[0].plot(b.K, b.Inertia, "o-", label=ten)
    axes[1].plot(b.K, b.Silhouette, "o-", label=ten)
axes[0].set(title="Khuỷu tay (đọc riêng từng không gian)", ylabel="Inertia")
axes[1].set(title="Silhouette theo K", ylabel="Silhouette")
for ax in axes:
    ax.set_xlabel("Số cụm K")
    ax.set_xticks(range(1, 11))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/chon_k.png", dpi=170)
plt.close(fig)
print("Da luu figures/chon_k.png")
