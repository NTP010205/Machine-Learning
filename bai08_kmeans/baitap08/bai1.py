# -*- coding: utf-8 -*-
"""Bài 1: tự tính silhouette từng khách bằng NumPy, giải thích điểm khó phân cụm."""

# Tương thích đường dẫn DLL Unicode trên Windows; không sửa môi trường hocmay.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tuong_thich_windows import sua_duong_dan_dll
sua_duong_dan_dll()

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler


def silhouette_mot_diem(X, nhan, i):
    """Trả về (a, b, s); cụm đơn có a=0 theo quy ước và s=0 như sklearn."""
    X = np.asarray(X, dtype=float)
    nhan = np.asarray(nhan)
    if X.ndim != 2 or nhan.ndim != 1 or len(X) != len(nhan):
        raise ValueError("X phai la ma tran, nhan la vector cung so mau.")
    if not np.isfinite(X).all():
        raise ValueError("X phai huu han, khong chua NaN/inf.")
    if not isinstance(i, (int, np.integer)) or not 0 <= i < len(X):
        raise ValueError("Vi tri i khong hop le.")
    cac_cum = np.unique(nhan)
    if not 2 <= len(cac_cum) < len(X):
        raise ValueError("Silhouette can 2 <= so cum <= so mau - 1.")
    khoang_cach = np.sqrt(((X - X[i]) ** 2).sum(axis=1))
    cung_cum = nhan == nhan[i]
    cung_cum[i] = False  # Không tính khoảng cách đến chính điểm i.
    a = float(khoang_cach[cung_cum].mean()) if cung_cum.any() else 0.0
    b = min(float(khoang_cach[nhan == k].mean()) for k in cac_cum if k != nhan[i])
    if not cung_cum.any() or max(a, b) == 0:
        s = 0.0
    else:
        s = (b - a) / max(a, b)
    return a, b, s


def main():
    df = pd.read_csv("data/khach_hang.csv")
    cot = ["thu_nhap", "diem_chi_tieu"]
    sc = StandardScaler().fit(df[cot])
    Xs = sc.transform(df[cot])
    km = KMeans(n_clusters=5, n_init=10, random_state=0).fit(Xs)
    ket_qua = np.array([silhouette_mot_diem(Xs, km.labels_, i) for i in range(len(Xs))])
    tu_tinh = ket_qua[:, 2].mean()
    thu_vien = silhouette_score(Xs, km.labels_)
    np.testing.assert_allclose(ket_qua[:, 2], silhouette_samples(Xs, km.labels_), atol=1e-12)
    print(f"Silhouette tu tinh: {tu_tinh:.10f}")
    print(f"Silhouette sklearn: {thu_vien:.10f}; chenh lech {abs(tu_tinh-thu_vien):.3e}")
    i = int(np.argmin(ket_qua[:, 2]))
    a, b, s = ket_qua[i]
    cum = int(km.labels_[i])
    khoang_cach = np.linalg.norm(Xs - Xs[i], axis=1)
    cum_khac = min((k for k in range(5) if k != cum),
                  key=lambda k: khoang_cach[km.labels_ == k].mean())
    tam = sc.inverse_transform(km.cluster_centers_)
    print(f"Khach silhouette nho nhat: vi tri iloc={i} (dem tu 0), index={df.index[i]}")
    print(df.iloc[i].to_string())
    print(f"Cum={cum}; a={a:.6f}; b={b:.6f}; s={s:.6f}")
    print(f"Tam cum duoc gan (thu nhap, chi tieu): {tam[cum].round(3).tolist()}")
    print(f"Cum khac gan nhat theo khoang cach TRUNG BINH: {cum_khac}, tam={tam[cum_khac].round(3).tolist()}")
    print("\nGiai thich:")
    print("a: trung binh khoang cach den cac khach KHAC trong cung cum, do do gan ket.")
    print("b: lay trung binh khoang cach den TUNG cum khac, roi chon trung binh nho nhat.")
    print("b khong phai khoang cach den tam cum hay den mot khach gan nhat.")
    print("s=(b-a)/max(a,b): gan 1 la tach bach; gan 0 la vung giao; am la gan cum khac hon.")
    print("Cum chi mot diem: a khong co trung binh hop le, quy uoc a=0 va s=0, KHONG gan s=1.")
    print("Khoang cach tinh tren du lieu da chuan hoa; toa do in ra van la don vi goc.")
    print("Doi chieu Hinh 3: khach duoc danh dau o ria cum cua minh, huong ve cum lan can;")
    print("doc vi tri theo thu nhap/diem chi tieu va tam tren, khong dong nhat so cum voi so tren hinh.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5), layout="constrained")
    for k in range(5):
        mask = km.labels_ == k
        ax.scatter(df.loc[mask, "thu_nhap"], df.loc[mask, "diem_chi_tieu"], s=28,
                   label=f"Cụm {k}", alpha=0.8)
    ax.scatter(tam[:, 0], tam[:, 1], marker="X", s=140, c="black", label="Tâm cụm")
    ax.scatter(df.iloc[i]["thu_nhap"], df.iloc[i]["diem_chi_tieu"], s=200,
               facecolors="none", edgecolors="red", linewidths=2, label=f"Silhouette thấp nhất: hàng {i}")
    ax.annotate(f"Hàng {i}, s={s:.3f}", (df.iloc[i]["thu_nhap"], df.iloc[i]["diem_chi_tieu"]),
                xytext=(-100, 18), textcoords="offset points", arrowprops={"arrowstyle": "->"})
    ax.set(xlabel="Thu nhập (triệu đồng/tháng)", ylabel="Điểm chi tiêu",
           title=f"210 khách hàng, K=5; silhouette trung bình={tu_tinh:.3f}")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4)
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/khach_hang_k5.png", dpi=170)
    plt.close(fig)
    print("Da luu figures/khach_hang_k5.png")


if __name__ == "__main__":
    main()
