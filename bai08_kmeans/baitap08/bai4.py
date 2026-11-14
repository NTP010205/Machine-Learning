# -*- coding: utf-8 -*-
"""Bài 4: lưu ảnh 16 màu, đo dung lượng PNG thật; ảnh cá nhân là phần riêng."""

# Tương thích đường dẫn DLL Unicode trên Windows; không sửa môi trường hocmay.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tuong_thich_windows import sua_duong_dan_dll
sua_duong_dan_dll()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image


def tim_anh_ca_nhan():
    thu_muc = Path(__file__).resolve().parents[1] / "data"
    for ten in ["anh_ca_nhan.png", "anh_ca_nhan.jpg", "anh_ca_nhan.jpeg"]:
        if (thu_muc / ten).is_file():
            return thu_muc / ten
    return next(iter(sorted(p for p in thu_muc.rglob("*")
                            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"})), None)


def chuan_hoa_rgb(anh):
    anh = np.asarray(anh)
    if anh.ndim != 3 or anh.shape[2] not in (3, 4):
        raise ValueError("Can anh RGB/RGBA; anh nay khong co 3 hoac 4 kenh.")
    if anh.shape[2] == 4:
        print("Anh RGBA: bo kenh alpha theo de bai.")
    rgb = anh[:, :, :3].astype(float)
    if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 255:
        raise ValueError("Gia tri mau phai huu han va nam trong 0..1 hoac 0..255.")
    if np.issubdtype(anh.dtype, np.integer) or rgb.max() > 1:
        rgb /= 255.0
    return rgb


def nen_va_luu(anh, ten_goc, ten_nen):
    rgb = chuan_hoa_rgb(anh)
    diem = rgb.reshape(-1, 3)
    rng = np.random.default_rng(0)
    mau = diem[rng.choice(len(diem), min(5000, len(diem)), replace=False)]
    if len(np.unique(mau, axis=0)) < 16:
        raise ValueError("Anh/mau co it hon 16 mau phan biet, khong du de tao 16 cum mau.")
    km = KMeans(n_clusters=16, n_init=4, random_state=0).fit(mau)
    anh_moi = km.cluster_centers_[km.predict(diem)].reshape(rgb.shape)
    Path("figures").mkdir(exist_ok=True)
    goc = Path("figures") / ten_goc
    nen = Path("figures") / ten_nen
    plt.imsave(goc, rgb)
    plt.imsave(nen, np.clip(anh_moi, 0, 1))
    byte_goc, byte_nen = goc.stat().st_size, nen.stat().st_size
    print(f"Mang anh nen: {anh_moi.shape}; so mau RGB sau luong tu: {len(np.unique(anh_moi.reshape(-1, 3), axis=0))}")
    print(f"{goc}: {byte_goc} byte = {byte_goc/1024:.2f} KB (1 KB=1024 byte)")
    print(f"{nen}: {byte_nen} byte = {byte_nen/1024:.2f} KB (1 KB=1024 byte)")
    print(f"Ty so dung luong PNG goc/nen: {byte_goc/byte_nen:.3f} lan; giam {100*(1-byte_nen/byte_goc):.2f}%")
    print(f"Sai lech tuyet doi trung binh moi kenh RGB: {np.abs(anh_moi-rgb).mean()*255:.4f}/255")
    print("Ly thuyet chi so 16 mau: 4 bit/pixel thay 24 bit, giam 6 lan (chua tinh bang mau).")
    print("PNG co nen, metadata va co the luu RGBA; ty so byte thuc khong nhat thiet la 6 lan.")


def main():
    print("PHAN 1: ANH MAU CHINA.JPG CUA SCIKIT-LEARN")
    nen_va_luu(load_sample_image("china.jpg"), "anh_goc.png", "anh_16_mau.png")
    print("\nPHAN 2: ANH CA NHAN")
    anh_ca_nhan = tim_anh_ca_nhan()
    if anh_ca_nhan is None:
        print("CHUA THUC HIEN: chua co anh ca nhan trong bai08_kmeans/data.")
        print("Bo sung data/anh_ca_nhan.png hoac data/anh_ca_nhan.jpg, sau do chay lai bai4.py.")
        print("Phan china.jpg thanh cong; Bai 4 CHUA HOAN THANH 100%.")
        return 2  # Trạng thái thiếu đầu vào tùy chọn, khác lỗi chương trình (1).
    print("Doc anh do nguoi dung dat trong data:", anh_ca_nhan)
    nen_va_luu(plt.imread(anh_ca_nhan), "anh_ca_nhan_goc.png", "anh_ca_nhan_16_mau.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
