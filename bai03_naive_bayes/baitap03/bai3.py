"""Bài 3: thay đổi alpha, giữ nguyên túi từ và phép chia của d3."""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def main():
    df = pd.read_csv("data/tin_nhan.csv")
    vec = CountVectorizer(token_pattern=r"(?u)\S+")
    X = vec.fit_transform(df["noi_dung"])
    y = df["la_rac"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )
    print("BAI 3 - DOI THAM SO LAM TRON")
    print(f"Train = {X_train.shape[0]}; test = {X_test.shape[0]}")
    print(f"{'alpha':>8s} {'Accuracy':>12s} {'Precision':>12s} {'Recall':>12s}")
    ket_qua = []
    for alpha in [0.001, 0.1, 1.0, 10.0, 100.0]:
        mo_hinh = MultinomialNB(alpha=alpha).fit(X_train, y_train)
        du_doan = mo_hinh.predict(X_test)
        diem = (accuracy_score(y_test, du_doan),
                precision_score(y_test, du_doan, zero_division=0),
                recall_score(y_test, du_doan, zero_division=0))
        ket_qua.append(diem)
        print(f"{alpha:8.3f} {diem[0]:12.4f} {diem[1]:12.4f} {diem[2]:12.4f}")

    print("\nNHAN XET:")
    if len(set(ket_qua)) == 1:
        print("Ca ba metric giu nguyen o tat ca cac muc alpha tren tap kiem tra nay.")
    else:
        print("Cac metric co thay doi theo alpha nhu bang ket qua thuc te o tren.")
    print("Alpha rat lon keo P(tu | lop) gan 1/V, lam giam su phan biet cua tu.")
    print("Dieu do khong dam bao accuracy luon giam: xac suat co the thay doi")
    print("nhung chua lam cac mau test vuot qua ranh gioi de doi nhan du doan.")
    print("Metric khong doi khong co nghia alpha khong anh huong den mo hinh.")


if __name__ == "__main__":
    main()
