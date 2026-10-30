"""Bài 6: rừng hồi quy, cây sâu 3 và đường thẳng trên cùng dữ liệu giá nhà."""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


def main():
    df = pd.read_csv("data/gia_nha.csv")
    X = df[["dien_tich"]]
    y = df["gia"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    cac_mo_hinh = [
        ("RandomForest 200 cay", RandomForestRegressor(n_estimators=200, random_state=0)),
        ("Cay sau 3", DecisionTreeRegressor(max_depth=3, random_state=0)),
        ("LinearRegression", LinearRegression()),
    ]
    can_moi = pd.DataFrame({"dien_tich": [250.0]})
    print("BAI 6 - RANDOM FOREST REGRESSION")
    print(f"Train={len(X_train)}, test={len(X_test)}; chi dung dien_tich")
    print(f"{'Mo hinh':24s} {'Test MAE':>12s} {'Gia 250 m2':>14s}")
    for ten, mo_hinh in cac_mo_hinh:
        mo_hinh.fit(X_train, y_train)
        mae = mean_absolute_error(y_test, mo_hinh.predict(X_test))
        gia = mo_hinh.predict(can_moi)[0]
        print(f"{ten:24s} {mae:12.4f} {gia:14.4f}")
    print("Don vi MAE va gia: ty dong.")
    print("\nNHAN XET:")
    print("Moi la cua cay hoi quy tra ve gia trung binh cua cac mau hoc trong la.")
    print("Rung lay trung binh du doan cua 200 cay; bootstrap lam cac cay khac nhau.")
    print("Can 250 m2 vuot vung dien tich train, nhung moi cay van dua no vao mot la")
    print("o bien. Gia tai la khong tiep tuc tang theo dien tich.")
    print("Trung binh cac gia tai la cung khong tao ra xu huong ngoai suy;")
    print("du doan rung nam trong mien gia train, khac duong thang co the keo dai.")


if __name__ == "__main__":
    main()
