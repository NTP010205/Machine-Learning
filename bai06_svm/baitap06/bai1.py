"""Bài 1: tự tính f(x) và khoảng cách bằng phép nhân, cộng và căn bậc hai."""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC


def in_phep_tinh(ten, diem, w, b):
    # Không gọi decision_function; giữ hệ số đầy đủ để tránh sai số làm tròn.
    tich_1 = w[0] * diem[0]
    tich_2 = w[1] * diem[1]
    f = tich_1 + tich_2 + b
    do_dai_w = np.sqrt(w[0] ** 2 + w[1] ** 2)
    khoang_cach = abs(f) / do_dai_w
    print(f"\n{ten}: ({diem[0]:.6f}, {diem[1]:.6f})")
    print(f"  w1*x1 = {w[0]:.6f} * {diem[0]:.6f} = {tich_1:.6f}")
    print(f"  w2*x2 = {w[1]:.6f} * {diem[1]:.6f} = {tich_2:.6f}")
    print(f"  f(x) = {tich_1:.6f} + ({tich_2:.6f}) + {b:.6f} = {f:+.6f}")
    print(f"  d = abs({f:.6f}) / {do_dai_w:.6f} = {khoang_cach:.6f}")
    print(f"  Du doan theo dau f(x): lop {int(f > 0)}")


def main():
    X, y = make_blobs(n_samples=24, centers=[[2, 2], [6, 6]], cluster_std=0.9, random_state=0)
    y = 1 - y
    svm = SVC(kernel="linear", C=1000).fit(X, y)
    w, b = svm.coef_[0], svm.intercept_[0]
    do_dai_w = np.sqrt(w[0] ** 2 + w[1] ** 2)
    print("BAI 1 - KHOANG CACH TOI RANH GIOI")
    print(f"w1={w[0]:.6f}, w2={w[1]:.6f}, b={b:.6f}")
    print(f"||w|| = sqrt(({w[0]:.6f})^2 + ({w[1]:.6f})^2) = {do_dai_w:.6f}")
    print(f"Do rong le = 2/||w|| = {2 / do_dai_w:.6f}")
    print(f"Nua do rong le = 1/||w|| = {1 / do_dai_w:.6f}")
    for diem in [[3.0, 3.0], [5.0, 4.5]]:
        in_phep_tinh("Diem moi", diem, w, b)
    for i in svm.support_:
        in_phep_tinh(f"Vecto ho tro {i}, nhan {y[i]}", X[i], w, b)
    print("\nNHAN XET:")
    print("Tren bo 24 diem tach duoc nay, ba vecto ho tro co f(x) xap xi +1 hoac -1.")
    print("Do do d = |f(x)|/||w|| xap xi 1/||w||, bang nua do rong le 2/||w||.")
    print("Sai lech rat nho do dung sai toi uu so; dung he so day du cua mo hinh,")
    print("khong chi ba chu so thap phan in trong PDF, de tinh chinh xac hon.")
    print("Diem (3,3) nam ngoai le ve lop 1; (5,4.5) nam trong le ve lop 0.")
    print("f(x) khong phai xac suat. Voi le mem, SV co the nam trong le/sai phia,")
    print("nen khong phai moi SV trong moi bai toan deu cach bien bang nua le.")


if __name__ == "__main__":
    main()
