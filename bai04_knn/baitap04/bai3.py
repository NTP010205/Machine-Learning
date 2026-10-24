"""Bài 3: chỉ đổi đơn vị trong bộ nhớ, giữ nguyên CSV và cùng phép chia."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    df = pd.read_csv("data/sinh_vien_mo_rong.csv")
    df_trieu = df.copy()
    df_trieu["chi_tieu"] = df_trieu["chi_tieu"] / 1000
    cot = ["gio_on", "diem_giua_ky", "chi_tieu"]
    y = df["qua_mon"]
    # Chia chỉ số một lần để mọi phép so sánh dùng cùng các sinh viên.
    chi_so_train, chi_so_test = train_test_split(
        df.index, test_size=0.25, random_state=17, stratify=y
    )
    y_train, y_test = y.loc[chi_so_train], y.loc[chi_so_test]
    ket_qua = {}
    du_doan_chuan = {}
    du_lieu_chuan = {}
    print("BAI 3 - DOI NGHIN DONG SANG TRIEU DONG")
    print(f"{'Don vi chi_tieu':20s} {'Khong chuan hoa':>18s} {'Co chuan hoa':>16s}")
    for ten, du_lieu in [("Nghin dong", df), ("Trieu dong", df_trieu)]:
        X_train = du_lieu.loc[chi_so_train, cot]
        X_test = du_lieu.loc[chi_so_test, cot]
        khong_chuan = KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)
        co_chuan = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7))
        co_chuan.fit(X_train, y_train)
        du_doan_chuan[ten] = co_chuan.predict(X_test)
        du_lieu_chuan[ten] = co_chuan.named_steps["standardscaler"].transform(X_test)
        acc_tho = accuracy_score(y_test, khong_chuan.predict(X_test))
        acc_chuan = accuracy_score(y_test, du_doan_chuan[ten])
        ket_qua[ten] = (acc_tho, acc_chuan)
        print(f"{ten:20s} {acc_tho:18.4f} {acc_chuan:16.4f}")

    print("\nNHAN XET:")
    print(f"Khong chuan hoa: accuracy thay doi"
          f" {ket_qua['Trieu dong'][0] - ket_qua['Nghin dong'][0]:+.4f}.")
    print("Chia chi_tieu cho 1000 chi doi don vi, khong doi muc chi tieu thuc te.")
    print("Nhung trong tong binh phuong khoang cach, dong gop cua chi_tieu giam")
    print("1.000.000 lan, nen thu tu lang gieng va ket qua bo phieu co the doi.")
    print("Sau StandardScaler, (x/1000 - mean/1000)/(std/1000) = (x-mean)/std.")
    print("Vi vay doi don vi duong khong doi toa do chuan hoa, tru sai so so thuc.")
    print("Hai pipeline chi fit scaler tren train; khong dung test de tinh mean/std.")
    print("So du doan sau chuan hoa khac nhau:",
          np.count_nonzero(du_doan_chuan["Nghin dong"] != du_doan_chuan["Trieu dong"]))
    print("Sai lech lon nhat cua toa do test sau chuan hoa:",
          f"{np.max(np.abs(du_lieu_chuan['Nghin dong'] - du_lieu_chuan['Trieu dong'])):.3e}")


if __name__ == "__main__":
    main()
