"""Bài 6: lớp hiếm, xác suất tiên nghiệm và cái bẫy accuracy."""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def main():
    df = pd.read_csv("data/tin_nhan.csv")
    # Giữ thứ tự trong mỗi lớp: toàn bộ tin thường, rồi 20 tin rác ĐẦU TIÊN.
    # Chỉ tạo DataFrame trong bộ nhớ; không ghi lại hay sửa CSV nguồn.
    du_lieu = pd.concat([
        df[df["la_rac"] == 0],
        df[df["la_rac"] == 1].head(20),
    ], ignore_index=True)
    y = du_lieu["la_rac"]
    tin_train, tin_test, y_train, y_test = train_test_split(
        du_lieu["noi_dung"], y, test_size=0.25, random_state=17, stratify=y
    )
    vec = CountVectorizer(token_pattern=r"(?u)\S+")
    X_train = vec.fit_transform(tin_train)
    X_test = vec.transform(tin_test)
    mo_hinh = MultinomialNB(alpha=1.0).fit(X_train, y_train)
    du_doan = mo_hinh.predict(X_test)
    ma_tran = confusion_matrix(y_test, du_doan, labels=[0, 1])
    tn, fp, fn, tp = ma_tran.ravel()

    print("BAI 6 - LOP HIEM VA XAC SUAT TIEN NGHIEM")
    print("So tin tung lop (0: thuong, 1: rac):")
    print(y.value_counts().sort_index().to_string())
    print(f"Train = {len(tin_train)}; test = {len(tin_test)}")
    print("So tin train tung lop:")
    print(y_train.value_counts().sort_index().to_string())
    prior = mo_hinh.class_count_ / mo_hinh.class_count_.sum()
    print(f"Prior hoc tu train: thuong = {prior[0]:.4f}; rac = {prior[1]:.4f}")
    print(f"Accuracy = {accuracy_score(y_test, du_doan):.4f}")
    print(f"Recall lop rac = {recall_score(y_test, du_doan, pos_label=1):.4f}")
    print("Ma tran nham lan (hang: nhan that; cot: du doan; thu tu 0, 1):")
    print(ma_tran)
    print(f"TN = {tn}; FP = {fp}; FN = {fn}; TP = {tp}")

    # Mốc so sánh đơn giản, tính trực tiếp trên tập test đang dùng.
    du_doan_toan_thuong = [0] * len(y_test)
    print("\nNHAN XET:")
    print(f"Dung {tn + tp}/{len(y_test)} tin nhung chi bat duoc {tp}/{tp + fn} tin rac;")
    print(f"bo sot {fn}/{tp + fn} tin rac ({fn / (tp + fn):.0%} so tin rac trong test).")
    print(f"Neu luon doan thuong: accuracy = {accuracy_score(y_test, du_doan_toan_thuong):.4f},")
    print(f"nhung recall rac = {recall_score(y_test, du_doan_toan_thuong, pos_label=1):.4f}.")
    print("Accuracy bi chi phoi boi lop thuong dong hon, nen co the che giau lop hiem.")
    print("Naive Bayes cong log prior vao diem: khi bang chung tu ngu yeu,")
    print("prior 0.9 cua lop thuong lon hon prior 0.1 cua lop rac nen nghieng ve thuong.")
    print("Stratify chi giu ti le hai lop khi chia, khong lam du lieu can bang.")
    print("Can xem recall rieng lop rac va FN de biet bao nhieu tin rac bi bo sot;")
    print("dong thoi xem FP de tranh chan nham tin thuong. Khong chi nhin accuracy.")
    print("Test chi co it tin rac, nen mot tin bo sot cung lam recall thay doi nhieu.")


if __name__ == "__main__":
    main()
