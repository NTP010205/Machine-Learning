"""Bài 4: tự cài Multinomial Naive Bayes bằng Counter và math.log."""

from collections import Counter
from math import log

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB  # Chỉ dùng để đối chứng trong main.


def tach_tu(cau):
    r"""Giống CountVectorizer(lowercase=True, token_pattern=r'(?u)\S+')."""
    return cau.lower().split()


class NaiveBayesTuViet:
    """Tự học số đếm, prior và log likelihood, không gọi mô hình sklearn."""

    def __init__(self, alpha=1.0):
        if alpha <= 0:
            raise ValueError("Can alpha > 0 de tinh log xac suat co lam tron.")
        self.alpha = alpha

    def fit(self, cac_cau, cac_nhan):
        # Chỉ nhận train; test không tham gia lập từ điển hay tính prior.
        cac_cau = list(cac_cau)
        cac_nhan = list(cac_nhan)
        self.so_tin = Counter(cac_nhan)
        self.cac_lop = sorted(self.so_tin)
        self.dem_tu = {nhan: Counter() for nhan in self.cac_lop}
        for cau, nhan in zip(cac_cau, cac_nhan):
            self.dem_tu[nhan].update(tach_tu(cau))
        self.tu_dien = sorted({tu for dem in self.dem_tu.values() for tu in dem})
        self.tap_tu = set(self.tu_dien)
        self.tong_tu = {nhan: sum(dem.values()) for nhan, dem in self.dem_tu.items()}
        self.prior = {nhan: self.so_tin[nhan] / len(cac_nhan) for nhan in self.cac_lop}
        self.log_prior = {nhan: log(p) for nhan, p in self.prior.items()}
        self.log_xac_suat = {}
        for nhan in self.cac_lop:
            mau_so = self.tong_tu[nhan] + self.alpha * len(self.tu_dien)
            self.log_xac_suat[nhan] = {
                tu: log((self.dem_tu[nhan][tu] + self.alpha) / mau_so)
                for tu in self.tu_dien
            }
        return self

    def diem_log(self, cau):
        dem_trong_cau = Counter(tach_tu(cau))
        return {
            nhan: self.log_prior[nhan] + sum(
                so_lan * self.log_xac_suat[nhan][tu]
                for tu, so_lan in dem_trong_cau.items() if tu in self.tap_tu
            )
            for nhan in self.cac_lop
        }

    def predict(self, cac_cau):
        du_doan = []
        for cau in cac_cau:
            diem = self.diem_log(cau)
            # Nếu hòa, chọn lớp nhỏ hơn như argmax của sklearn trên classes_.
            du_doan.append(max(self.cac_lop, key=lambda nhan: diem[nhan]))
        return du_doan


def main():
    df = pd.read_csv("data/tin_nhan.csv")
    y = df["la_rac"]
    tin_train, tin_test, y_train, y_test = train_test_split(
        df["noi_dung"], y, test_size=0.25, random_state=17, stratify=y
    )
    tu_viet = NaiveBayesTuViet(alpha=1.0).fit(tin_train, y_train)
    du_doan_tay = tu_viet.predict(tin_test)

    # Từ điển đối chứng chính là từ điển đã tự xây trên train.
    # Cùng chữ thường, cùng cách tách từ và cùng cách bỏ qua từ ngoài từ điển.
    vocabulary = {tu: i for i, tu in enumerate(tu_viet.tu_dien)}
    vec = CountVectorizer(token_pattern=r"(?u)\S+", vocabulary=vocabulary)
    X_train = vec.fit_transform(tin_train)
    X_test = vec.transform(tin_test)
    doi_chung = MultinomialNB(alpha=1.0).fit(X_train, y_train)
    du_doan_sklearn = doi_chung.predict(X_test)

    print("BAI 4 - BO LOC TU VIET VA DOI CHUNG")
    print(f"Train = {len(tin_train)}; test = {len(tin_test)}")
    print(f"So tu trong vocabulary cua train: {len(tu_viet.tu_dien)}")
    for nhan in tu_viet.cac_lop:
        print(f"Lop {nhan}: {tu_viet.so_tin[nhan]} tin, "
              f"{tu_viet.tong_tu[nhan]} tu, prior = {tu_viet.prior[nhan]:.4f}")
    print(f"Accuracy tu viet = {accuracy_score(y_test, du_doan_tay):.4f}")
    print(f"Accuracy MultinomialNB = {accuracy_score(y_test, du_doan_sklearn):.4f}")
    so_khac = sum(a != b for a, b in zip(du_doan_tay, du_doan_sklearn))
    print(f"So du doan khac nhau = {so_khac}/{len(y_test)}")
    print("\nNHAN XET:")
    print("1. Cong log tranh tran duoi khi nhan nhieu xac suat rat nho.")
    print("   Log dong bien nen lop co diem lon nhat van duoc giu nguyen.")
    print("2. Laplace: (dem + alpha)/(tong tu lop + alpha*V).")
    print("   Tu co trong vocabulary nhung vang o mot lop van co xac suat duong;")
    print("   cong alpha*V o mau so giu tong xac suat cac tu bang 1.")
    print("3. Tu ngoai vocabulary train bi bo qua o CA HAI mo hinh.")
    print("   Khong them tu test vao vocabulary vi se lam ro ri thong tin test.")
    print("   Neu cau khong co tu nao da biet, chi prior quyet dinh du doan.")
    print("4. So sanh dung cung train/test, vocabulary, alpha va tien xu ly.")
    print("   Ket qua gan nhau vi ca hai cung thuc hien Multinomial Naive Bayes.")


if __name__ == "__main__":
    main()
