"""Bài 2: thêm riêng từng điểm để quan sát vai trò của vectơ hỗ trợ."""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC


def main():
    X, y = make_blobs(n_samples=24, centers=[[2, 2], [6, 6]], cluster_std=0.9, random_state=0)
    y = 1 - y
    goc = SVC(kernel="linear", C=1000).fit(X, y)
    w_goc, b_goc = goc.coef_[0], goc.intercept_[0]
    print("BAI 2 - THEM MOT DIEM")
    print(f"Goc: w1={w_goc[0]:.6f}, w2={w_goc[1]:.6f}, b={b_goc:.6f}")
    print("SV goc (chi so 0-based):", goc.support_)
    for diem in [[-1.0, -1.0], [4.5, 4.0]]:
        # Luôn ghép với 24 điểm GỐC, không cộng dồn hai thí nghiệm.
        X_them = np.vstack([X, diem])
        y_them = np.append(y, 1)
        moi = SVC(kernel="linear", C=1000).fit(X_them, y_them)
        w_moi, b_moi = moi.coef_[0], moi.intercept_[0]
        f_truoc = w_goc @ np.array(diem) + b_goc
        f_sau = w_moi @ np.array(diem) + b_moi
        print(f"\nThem diem lop 1 tai {tuple(diem)}:")
        print(f"  f_goc(diem moi) = {f_truoc:+.6f}; f_sau = {f_sau:+.6f}")
        print(f"  w1={w_moi[0]:.6f}, w2={w_moi[1]:.6f}, b={b_moi:.6f}")
        print(f"  Delta w = {w_moi - w_goc}; Delta b = {b_moi - b_goc:+.6f}")
        print("  Chi so SV:", moi.support_)
        print("  Diem moi la SV:", len(X) in moi.support_)
        print(f"  Do rong le sau = {2 / np.linalg.norm(w_moi):.6f}")
    print("\nNHAN XET:")
    print("Diem (-1,-1) o xa bien, dung phia lop 1 va ngoai le; no khong tao")
    print("rang buoc sat hon, nen bien gan nhu giu nguyen (co the lech do dung sai).")
    print("Diem (4.5,4) xam pham le cu cua lop 1, tro thanh SV va ep SVM doi bien.")
    print("Bien phu thuoc cac rang buoc tai SV, khong phai moi diem deu co vai tro ngang nhau.")
    print("So sanh bang he so chua lam tron va tap SV, khong chi nhin nhan du doan.")


if __name__ == "__main__":
    main()
