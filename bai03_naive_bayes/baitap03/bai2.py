"""Bài 2: xếp hạng từ bằng chênh lệch log xác suất giữa hai lớp."""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


def main():
    df = pd.read_csv("data/tin_nhan.csv")
    vec = CountVectorizer(token_pattern=r"(?u)\S+")
    X = vec.fit_transform(df["noi_dung"])
    mo_hinh = MultinomialNB(alpha=1.0).fit(X, df["la_rac"])
    # classes_ = [0, 1]: hiệu dương nghiêng về rác, hiệu âm về thường.
    chenh_lech = mo_hinh.feature_log_prob_[1] - mo_hinh.feature_log_prob_[0]
    bang = pd.DataFrame({"tu": vec.get_feature_names_out(), "chenh_lech_log": chenh_lech})
    print("BAI 2 - TU NAO TO CAO MANH NHAT")
    print(f"So tin huan luyen: {X.shape[0]}; so tu: {X.shape[1]}")
    print("\n5 tu nghieng manh nhat ve tin rac:")
    print(bang.nlargest(5, "chenh_lech_log").to_string(index=False, float_format="%.4f"))
    print("\n5 tu nghieng manh nhat ve tin thuong:")
    print(bang.nsmallest(5, "chenh_lech_log").to_string(index=False, float_format="%.4f"))
    print("\nNhan xet: xep hang theo ti so xac suat, khong chi theo so lan xuat hien.")


if __name__ == "__main__":
    main()
