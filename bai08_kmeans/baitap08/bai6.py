# -*- coding: utf-8 -*-
"""Bài 6: KMeans trên 64 pixel chữ số, không đưa nhãn thật vào huấn luyện."""

# Tương thích đường dẫn DLL Unicode trên Windows; không sửa môi trường hocmay.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tuong_thich_windows import sua_duong_dan_dll
sua_duong_dan_dll()

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits

du_lieu = load_digits()
X = du_lieu.data
km = KMeans(n_clusters=10, n_init=10, random_state=0).fit(X)
cum = km.labels_
# Nhãn thật chỉ xuất hiện sau bước fit; pixel cùng thang 0..16, giữ nguyên.
nhan_that = du_lieu.target
bang = pd.crosstab(pd.Series(nhan_that, name="chu_so"), pd.Series(cum, name="cum"))
print(f"Du lieu {X.shape}; K=10, n_init=10, random_state=0; inertia={km.inertia_:.4f}")
print("Bang crosstab day du (hang=chu so that, cot=cum):")
print(bang.to_string())
print("\nMuc tap trung cua tung chu so:")
print("Chu so | Cum chinh | So mau trong cum chinh/tong | Ty le | So cum co >=10% so mau chu so")
ty_le = bang.div(bang.sum(axis=1), axis=0)
for so in bang.index:
    chinh = int(bang.loc[so].idxmax())
    so_nhom = int((ty_le.loc[so] >= 0.1).sum())
    print(f"{so:6d} | {chinh:9d} | {bang.loc[so, chinh]:3d}/{bang.loc[so].sum():3d} | {ty_le.loc[so, chinh]:.4f} | {so_nhom}")
print("\nGom gon (>=90% mau chu so trong mot cum):", ty_le.max(axis=1).loc[lambda v: v >= 0.9].index.tolist())
print("Bi chia (it nhat hai cum moi cum chua >=10% mau chu so):",
      ty_le.index[(ty_le >= 0.1).sum(axis=1) >= 2].tolist())
print("\nThanh phan moi cum: hai chu so dong nhat (so mau, ty le trong cum):")
for k in bang.columns:
    dong_nhat = bang[k].nlargest(2)
    print(f"Cum {k}: " + "; ".join(f"chu so {so}: {n} ({n/bang[k].sum():.1%})" for so, n in dong_nhat.items()))
print("KMeans toi uu khoang cach Euclid giua pixel, khong hoc y nghia cua chu so.")
print("Cung chu so co nhieu net viet/do nghieng; khac chu so co the co hinh dang gan nhau.")
print("Do do mot chu so co the tach nhieu cum, mot cum co the gom nhieu chu so.")
print("K=10 chi chi dinh so tam, khong ep 10 cum tuong ung mot-mot 10 chu so; khong sua nhan cum.")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 5, figsize=(10, 4.5), layout="constrained")
for k, ax in enumerate(axes.flat):
    ax.imshow(km.cluster_centers_[k].reshape(8, 8), cmap="gray_r", vmin=0, vmax=16,
              interpolation="nearest")
    ax.set_title(f"Cụm {k} (n={int((cum == k).sum())})", fontsize=10)
    ax.axis("off")
fig.suptitle("Tâm cụm KMeans: trung bình pixel, không phải nhãn chữ số")
Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/tam_chu_so.png", dpi=170)
plt.close(fig)
print("Da luu figures/tam_chu_so.png")
