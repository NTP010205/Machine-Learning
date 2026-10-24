"""Bài 2: chọn k với Manhattan bằng CV 5-fold chỉ trên train."""

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier


def main():
    df = pd.read_csv("data/hai_vang_trang.csv")
    X = df[["x1", "x2"]]
    y = df["nhan"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )
    print("BAI 2 - KHOANG CACH MANHATTAN")
    print("Bang minh hoa nhu e3 (khong dung diem test de chon k):")
    print(f"{'k':>4} {'Train':>10} {'Test':>10}")
    for k in [1, 5, 13, 31, 101]:
        mo_hinh = KNeighborsClassifier(n_neighbors=k, metric="manhattan").fit(X_train, y_train)
        print(f"{k:4d} {accuracy_score(y_train, mo_hinh.predict(X_train)):10.4f}"
              f" {accuracy_score(y_test, mo_hinh.predict(X_test)):10.4f}")

    diem_cv = {}
    for k in range(1, 52, 2):
        mo_hinh = KNeighborsClassifier(n_neighbors=k, metric="manhattan")
        diem_cv[k] = cross_val_score(mo_hinh, X_train, y_train, cv=5).mean()
    # Nếu bằng điểm, max chọn k nhỏ nhất do đã duyệt k tăng dần.
    k_tot = max(diem_cv, key=diem_cv.get)
    manhattan = KNeighborsClassifier(n_neighbors=k_tot, metric="manhattan").fit(X_train, y_train)
    acc_manhattan = accuracy_score(y_test, manhattan.predict(X_test))
    # Tính lại mốc Euclid của Lab trên cùng phép chia, không gán cứng accuracy.
    euclid = KNeighborsClassifier(n_neighbors=13, metric="euclidean").fit(X_train, y_train)
    acc_euclid = accuracy_score(y_test, euclid.predict(X_test))
    print(f"\nk tot nhat Manhattan = {k_tot}")
    print(f"CV 5-fold mean = {diem_cv[k_tot]:.4f}")
    print(f"Accuracy test Manhattan = {acc_manhattan:.4f}")
    print(f"Accuracy test Euclid (k=13) = {acc_euclid:.4f}")
    print("\nNHAN XET:")
    print(f"Manhattan - Euclid = {acc_manhattan - acc_euclid:+.4f}"
          f" ({100 * (acc_manhattan - acc_euclid):+.2f} diem phan tram).")
    if abs(acc_manhattan - acc_euclid) <= 1 / len(y_test) + 1e-12:
        print("Chenh lech nho tren test nay: khong qua mot mau phan loai dung.")
    print("Manhattan cong tri tuyet doi cac chenh lech; Euclid lay can tong binh phuong.")
    print("Doi metric lam thay doi thu tu lang gieng va co the doi k tot nhat.")
    print("Chenh lech tren chi danh gia phep chia nay, khong chung minh metric nao luon tot hon.")


if __name__ == "__main__":
    main()
