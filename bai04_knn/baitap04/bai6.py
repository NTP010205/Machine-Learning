"""Bài 6: đo thời gian thật trên dữ liệu ngẫu nhiên theo đề, không cần CSV."""

import time

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def main():
    rng = np.random.default_rng(0)
    # Dùng cùng 1000 mẫu truy vấn cho mọi cỡ train để so sánh công bằng.
    X_moi = rng.random((1000, 2))
    print("BAI 6 - DO THOI GIAN FIT VA PREDICT")
    print("k=5, algorithm='auto', 2 dac trung, 1000 mau predict; don vi: giay")
    print(f"{'so_mau':>10s} {'fit_time':>14s} {'predict_time':>16s}")
    ket_qua = []
    for so_mau in [1000, 10000, 100000]:
        X_hoc = rng.random((so_mau, 2))
        y_hoc = rng.integers(0, 2, size=so_mau)
        mo_hinh = KNeighborsClassifier(n_neighbors=5)
        # Không đưa thời gian tạo dữ liệu hoặc in màn hình vào phép đo.
        bat_dau = time.perf_counter()
        mo_hinh.fit(X_hoc, y_hoc)
        fit_time = time.perf_counter() - bat_dau
        bat_dau = time.perf_counter()
        mo_hinh.predict(X_moi)
        predict_time = time.perf_counter() - bat_dau
        ket_qua.append(predict_time)
        print(f"{so_mau:10d} {fit_time:14.6f} {predict_time:16.6f}")
    print("\nNHAN XET:")
    print("KNN la lazy learning: fit chu yeu luu du lieu, khong toi uu trong so.")
    print("Thu vien co the xay chi muc tim kiem nen fit van ton thoi gian,")
    print("dac biet voi 100000 mau; khong nen hieu fit la khong lam gi.")
    print("Predict phai tim lang gieng trong tap train: tap cang lon thi viec tim")
    print("kiem thuong ton them thoi gian. Cay tim kiem trong 2 chieu co the")
    print("giam so diem can xet nen thoi gian khong nhat thiet tang tuyen tinh.")
    if not all(a <= b for a, b in zip(ket_qua, ket_qua[1:])):
        print("Lan do nay predict_time khong tang deu; can doc dung cac so da do.")
    print("Thoi gian tuy may, tai CPU, cache va thuat toan; mot lan do co dao dong.")
    print("Nhan ngau nhien chi phuc vu do toc do, khong dung de ket luan chat luong.")


if __name__ == "__main__":
    main()
