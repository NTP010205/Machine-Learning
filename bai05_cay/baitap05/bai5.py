"""Bài 5: đối chiếu giảm Gini trên train với permutation importance trên test."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def main():
    df = pd.read_csv("data/khach_vay.csv")
    X = df.drop(columns="tra_dung_han")
    y = df["tra_dung_han"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )
    # Giữ đúng cột train như f6; dùng tiếp bộ sinh để tạo các số MỚI cho test.
    # Không dùng nhãn test, không ghi lại CSV, không fit mô hình trên test.
    rng = np.random.default_rng(0)
    X_train_nn, X_test_nn = X_train.copy(), X_test.copy()
    X_train_nn["so_ngau_nhien"] = rng.random(len(X_train))
    X_test_nn["so_ngau_nhien"] = rng.random(len(X_test))
    rung = RandomForestClassifier(n_estimators=200, random_state=0).fit(X_train_nn, y_train)
    ket_qua = permutation_importance(
        rung, X_test_nn, y_test, n_repeats=20, random_state=0, scoring="accuracy"
    )
    bang = pd.DataFrame({
        "Gini_importance": rung.feature_importances_,
        "Permutation_mean": ket_qua.importances_mean,
        "Permutation_std": ket_qua.importances_std,
    }, index=X_train_nn.columns)
    # Hạng 1 là cao nhất; nếu bằng điểm thì cùng hạng.
    bang["Hang_Gini"] = bang["Gini_importance"].rank(ascending=False, method="min").astype(int)
    bang["Hang_permutation"] = bang["Permutation_mean"].rank(ascending=False, method="min").astype(int)
    print("BAI 5 - PERMUTATION IMPORTANCE")
    print(f"Accuracy test cua rung 5 cot = {accuracy_score(y_test, rung.predict(X_test_nn)):.4f}")
    print(bang.sort_values("Permutation_mean", ascending=False).to_string(float_format="%.4f"))
    ngau_nhien = bang.loc["so_ngau_nhien"]
    print(f"\nCot ngau nhien: hang Gini = {int(ngau_nhien['Hang_Gini'])}/5; "
          f"hang permutation = {int(ngau_nhien['Hang_permutation'])}/5")
    print(f"Gini importance = {ngau_nhien['Gini_importance']:.4f}; "
          f"permutation mean = {ngau_nhien['Permutation_mean']:.4f}, "
          f"std = {ngau_nhien['Permutation_std']:.4f}")
    print("\nNHAN XET:")
    print("1. feature_importances_ cong muc giam Gini tren TRAIN, tong bang 1.")
    print("   Cot ngau nhien co nhieu gia tri/nguong de thu; cay sau co the dung no")
    print("   de tach cac mau nhieu va duoc cong diem du khong giup tong quat hoa.")
    print("2. Permutation importance = accuracy goc - accuracy sau xao tron mot cot")
    print("   tren TEST, trung binh 20 lan. Mo hinh da fit duoc giu nguyen.")
    print("   Cach do nay hoi cot do co giup du doan tren du lieu chua hoc hay khong.")
    print("3. Cot ngau nhien khong co tin hieu duoc thiet ke tu nhan; muc giam diem")
    print("   nho/gan 0 hoac am khong ung ho thu hang cao cua no theo Gini.")
    print("   Neu am, xao tron con cai thien diem mot chut, khong phai 'xac suat am'.")
    print("4. Hai cach khac thang do: Gini chuan hoa ve tong 1; permutation la muc")
    print("   thay doi accuracy, khong can tong bang 1 va co dao dong qua cac lan tron.")
    print("   Cot tuong quan co the thay the nhau, lam permutation importance thap.")
    print("5. Day la giai thich mo hinh da hoc, khong chung minh quan he nhan qua.")
    print("   Khong dung ket qua test nay de chon lai mo hinh roi bao diem nhu test moi.")


if __name__ == "__main__":
    main()
