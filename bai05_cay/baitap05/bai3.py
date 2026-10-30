"""Bài 3: chọn độ sâu cho Gini và entropy bằng CV trên cùng tập train."""

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier


def main():
    df = pd.read_csv("data/khach_vay.csv")
    X = df.drop(columns="tra_dung_han")
    y = df["tra_dung_han"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )
    print("BAI 3 - GINI HAY ENTROPY")
    diem_cv = {tieu_chi: {} for tieu_chi in ["gini", "entropy"]}
    print(f"{'Do sau':>8s} {'CV Gini':>12s} {'CV Entropy':>12s}")
    for do_sau in range(1, 16):
        for tieu_chi in diem_cv:
            cay = DecisionTreeClassifier(criterion=tieu_chi, max_depth=do_sau, random_state=0)
            diem_cv[tieu_chi][do_sau] = cross_val_score(
                cay, X_train, y_train, cv=5, error_score="raise"
            ).mean()
        print(f"{do_sau:8d} {diem_cv['gini'][do_sau]:12.4f} {diem_cv['entropy'][do_sau]:12.4f}")
    # Chốt bằng CV trước khi chấm test; nếu bằng điểm, giữ độ sâu nhỏ hơn.
    do_sau_tot = {tc: max(diem, key=diem.get) for tc, diem in diem_cv.items()}
    accuracy = {}
    print("\nKet qua sau khi chon:")
    for tieu_chi, do_sau in do_sau_tot.items():
        cay = DecisionTreeClassifier(criterion=tieu_chi, max_depth=do_sau, random_state=0)
        cay.fit(X_train, y_train)
        accuracy[tieu_chi] = accuracy_score(y_test, cay.predict(X_test))
        print(f"{tieu_chi:8s}: do sau={do_sau}, CV={diem_cv[tieu_chi][do_sau]:.4f}, "
              f"accuracy test={accuracy[tieu_chi]:.4f}")
    chenh = accuracy["entropy"] - accuracy["gini"]
    print("\nNHAN XET:")
    print(f"Entropy - Gini tren test = {chenh:+.4f} ({100 * chenh:+.2f} diem phan tram).")
    if abs(chenh) <= 1 / len(y_test) + 1e-12:
        print("Chenh lech nho tren test nay: khong qua mot khach duoc phan loai dung.")
    print("Hai tieu chi deu khuyen khich nut thuan, nhung cong thuc khac nhau")
    print("nen co the chon nguong va do sau khac. Ket luan chi cho phep chia nay.")
    print("Khong chon lai do sau sau khi thay diem test.")


if __name__ == "__main__":
    main()
