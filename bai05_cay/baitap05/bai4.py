"""Bài 4: chọn min_samples_leaf bằng CV, giữ cây không giới hạn độ sâu."""

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
    print("BAI 4 - CAT TIA BANG min_samples_leaf")
    print(f"{'min_samples_leaf':>18s} {'CV 5-fold':>12s}")
    diem_cv = {}
    for so_mau in [1, 2, 5, 10, 20, 40]:
        cay = DecisionTreeClassifier(min_samples_leaf=so_mau, random_state=0)
        diem_cv[so_mau] = cross_val_score(cay, X_train, y_train, cv=5, error_score="raise").mean()
        print(f"{so_mau:18d} {diem_cv[so_mau]:12.4f}")
    # Nếu bằng điểm CV, giữ giá trị nhỏ hơn đã được duyệt trước.
    tot = max(diem_cv, key=diem_cv.get)
    cay_tot = DecisionTreeClassifier(min_samples_leaf=tot, random_state=0).fit(X_train, y_train)
    acc_la = accuracy_score(y_test, cay_tot.predict(X_test))
    # Cây sâu 5 là mốc Lab cố định, không dò thêm tham số bằng test.
    cay_sau_5 = DecisionTreeClassifier(max_depth=5, random_state=0).fit(X_train, y_train)
    acc_sau = accuracy_score(y_test, cay_sau_5.predict(X_test))
    print(f"\nmin_samples_leaf tot nhat = {tot}; CV = {diem_cv[tot]:.4f}")
    print(f"Cay duoc chon: do sau={cay_tot.get_depth()}, so la={cay_tot.get_n_leaves()}")
    print(f"Accuracy test min_samples_leaf = {acc_la:.4f}")
    print(f"Accuracy test max_depth=5 = {acc_sau:.4f}")
    print("\nNHAN XET:")
    if acc_la > acc_sau:
        print("min_samples_leaf duoc chon cho ket qua test tot hon max_depth=5 tren bo nay.")
    elif acc_la < acc_sau:
        print("max_depth=5 cho ket qua test tot hon min_samples_leaf duoc chon tren bo nay.")
    else:
        print("Hai cach cho accuracy test bang nhau tren bo nay.")
    print("min_samples_leaf chan phep chia tao la qua it mau, giam hoc nhieu cuc bo.")
    print("max_depth gioi han so cau hoi tren moi duong di; hai rang buoc khac nhau.")
    print("Khong co cach luon tot hon: so mau toi thieu qua lon cung gay underfitting.")


if __name__ == "__main__":
    main()
