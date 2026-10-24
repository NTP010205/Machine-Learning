"""Bài 4: so sánh phiếu đều nhau và phiếu theo nghịch đảo khoảng cách."""

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def main():
    df = pd.read_csv("data/hai_vang_trang.csv")
    X = df[["x1", "x2"]]
    y = df["nhan"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )
    ket_qua = {}
    print("BAI 4 - LANG GIENG CO TRONG SO")
    print(f"{'k':>5s} {'uniform':>12s} {'distance':>12s}")
    for k in [1, 13, 101]:
        ket_qua[k] = {}
        for trong_so in ["uniform", "distance"]:
            mo_hinh = KNeighborsClassifier(n_neighbors=k, weights=trong_so)
            mo_hinh.fit(X_train, y_train)
            ket_qua[k][trong_so] = accuracy_score(y_test, mo_hinh.predict(X_test))
        print(f"{k:5d} {ket_qua[k]['uniform']:12.4f} {ket_qua[k]['distance']:12.4f}")
    chenh = ket_qua[101]["distance"] - ket_qua[101]["uniform"]
    print("\nNHAN XET:")
    print(f"Voi k=101, distance - uniform = {chenh:+.4f}.")
    if chenh > 0:
        print("Tren phep chia nay, distance giup giam tac dong underfit cua k qua lon.")
    elif chenh == 0:
        print("Tren phep chia nay, distance chua cai thien accuracy voi k=101.")
    else:
        print("Tren phep chia nay, distance cho accuracy thap hon voi k=101.")
    print("weights='distance' cho lang gieng gan phieu manh hon (ty le voi 1/d),")
    print("nen cac diem xa kho lan at bang chung gan. Diem trung khop duoc uu tien.")
    print("k=1 chi co mot lang gieng nen hai cach bo phieu cho cung du doan.")
    print("Trong so khoang cach khong bao dam tot hon tren moi bo du lieu.")


if __name__ == "__main__":
    main()
