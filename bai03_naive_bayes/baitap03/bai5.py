"""Bài 5: chương trình chạy được chưa chứng minh chọn đúng biến thể."""

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB


def main():
    df = pd.read_csv("data/sinh_vien.csv")
    X = df[["gio_on", "diem_giua_ky"]]
    y = df["qua_mon"]
    # Cùng một lần chia để cả hai mô hình được chấm trên đúng cùng sinh viên.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )
    print("BAI 5 - CHON SAI BIEN THE NAIVE BAYES")
    for ten, mo_hinh in [("MultinomialNB", MultinomialNB()), ("GaussianNB", GaussianNB())]:
        mo_hinh.fit(X_train, y_train)
        print(f"{ten}: accuracy = {accuracy_score(y_test, mo_hinh.predict(X_test)):.4f}")
    print("\nNHAN XET:")
    print("Chuong trinh khong bao loi vi cac gia tri dau vao deu khong am.")
    print("MultinomialNB phu hop du lieu dem khong am, nhu so lan xuat hien cua tu.")
    print("gio_on va diem_giua_ky la cac dai luong lien tuc, khong phai so dem tu.")
    print("GaussianNB mo ta phan phoi lien tuc cua tung dac trung theo tung lop.")
    print("Chay duoc va co accuracy khong co nghia MultinomialNB dung ve nguyen tac.")


if __name__ == "__main__":
    main()
