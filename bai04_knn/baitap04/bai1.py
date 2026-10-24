"""Bài 1: tự viết KNN bằng numpy, sklearn chỉ dùng làm đối chứng."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def knn_du_doan(X_hoc, y_hoc, x_moi, k):
    """Tính khoảng cách, chọn k láng giềng rồi bỏ phiếu cho một mẫu."""
    X_hoc = np.asarray(X_hoc, dtype=float)
    y_hoc = np.asarray(y_hoc)
    x_moi = np.asarray(x_moi, dtype=float)
    if not 1 <= k <= len(X_hoc):
        raise ValueError("k phai tu 1 den so mau train.")
    khoang_cach = np.sqrt(((X_hoc - x_moi) ** 2).sum(axis=1))
    chi_so_gan = np.argsort(khoang_cach)[:k]
    cac_nhan, so_phieu = np.unique(y_hoc[chi_so_gan], return_counts=True)
    # np.unique sắp xếp nhãn; nếu hòa thì chọn nhãn nhỏ nhất như sklearn.
    return cac_nhan[np.argmax(so_phieu)]


def main():
    df = pd.read_csv("data/hai_vang_trang.csv")
    X = df[["x1", "x2"]].to_numpy(dtype=float)
    y = df["nhan"].to_numpy()
    X_hoc, X_test, y_hoc, y_test = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )
    k = 13
    du_doan_tay = np.array([knn_du_doan(X_hoc, y_hoc, mau, k) for mau in X_test])
    # Chỉ gọi mô hình thư viện sau khi phần tự viết đã dự đoán xong.
    doi_chung = KNeighborsClassifier(n_neighbors=k).fit(X_hoc, y_hoc)
    du_doan_thu_vien = doi_chung.predict(X_test)
    print("BAI 1 - TU VIET KNN")
    print(f"Train = {len(X_hoc)}, test = {len(X_test)}, k = {k}")
    print(f"Accuracy tu viet = {accuracy_score(y_test, du_doan_tay):.4f}")
    print(f"Accuracy sklearn = {accuracy_score(y_test, du_doan_thu_vien):.4f}")
    print(f"So du doan khac nhau = {np.count_nonzero(du_doan_tay != du_doan_thu_vien)}")
    print("\nNHAN XET:")
    print("KNN khong hoc trong so nhu hoi quy; no giu cac mau va nhan cua train.")
    print("Khi predict, ham tu viet tinh Euclid tu mau moi toi TAT CA mau train,")
    print("dung np.argsort lay 13 khoang cach nho nhat, dem phieu va chon lop da so.")
    print("Voi hai lop va k le, bo phieu deu nhau khong bi hoa.")
    print("Ca hai dung cung train/test, metric Euclid va k nen du doan can khop.")
    print("Neu co nhieu diem dong khoang cach o vi tri k, cach pha hoa co the")
    print("lam chon lang gieng khac nhau; can kiem tra tung du doan, khong chi accuracy.")


if __name__ == "__main__":
    main()
