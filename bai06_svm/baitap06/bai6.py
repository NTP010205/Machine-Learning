"""Bài 6: đo đầy đủ thời gian fit của SVC() và LinearSVC() theo đề."""

import time

from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC, SVC


def main():
    print("BAI 6 - SVM TREN DU LIEU LON", flush=True)
    print("Chi do .fit(), khong tinh sinh du lieu; don vi giay. Mo hinh mac dinh.", flush=True)
    print(f"{'n':>8s} {'SVC fit':>12s} {'LinearSVC fit':>15s}", flush=True)
    ket_qua = []
    for n in [2000, 5000, 10000, 20000]:
        X, y = make_classification(n_samples=n, n_features=20, random_state=0)
        thoi_gian = []
        for mo_hinh in [SVC(), LinearSVC()]:
            bat_dau = time.perf_counter()
            mo_hinh.fit(X, y)
            thoi_gian.append(time.perf_counter() - bat_dau)
        ket_qua.append((n, *thoi_gian))
        print(f"{n:8d} {thoi_gian[0]:12.6f} {thoi_gian[1]:15.6f}", flush=True)
    print("\nMuc tang so voi kich thuoc truoc:")
    for truoc, sau in zip(ket_qua, ket_qua[1:]):
        print(f"{truoc[0]} -> {sau[0]}: n x{sau[0]/truoc[0]:.2f}; "
              f"SVC x{sau[1]/truoc[1]:.2f}; LinearSVC x{sau[2]/truoc[2]:.2f}")
    print("\nNHAN XET:")
    print("2000 -> 5000 la tang 2.5 lan; hai buoc sau moi la tang gap doi.")
    nhanh_hon = sum(tuyen < kernel for _, kernel, tuyen in ket_qua)
    print(f"LinearSVC nhanh hon SVC o {nhanh_hon}/{len(ket_qua)} kich thuoc trong lan do nay.")
    print("SVC mac dinh dung RBF, phai tinh tuong tac kernel giua cac mau;")
    print("LinearSVC toi uu bien tuyen tinh nen thuong phu hop hon voi nhieu mau.")
    print("Khong suy ra toc do nhanh hon dong nghia accuracy cao hon: hai mo hinh khac bien.")
    print("Moi co du lieu duoc sinh dung random_state=0 theo de, khong phai cac tap long nhau.")
    print("Do kho du lieu, so SV, dung sai, tai CPU va cache anh huong thoi gian;")
    print("mot lan do khong chung minh chinh xac bac phuc tap, ti le co the dao dong.")


if __name__ == "__main__":
    main()
