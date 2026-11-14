# -*- coding: utf-8 -*-
"""Bài 1: tự tính bốn thước đo bằng NumPy, sau đó đối chiếu sklearn."""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def danh_gia(y_that, y_doan):
    """Trả về (accuracy, precision, recall, F1) cho nhãn dương 1.

    Chỉ nhận hai mảng nhãn nhị phân cùng kích thước, một chiều, không rỗng.
    Quy ước chỉ số có mẫu số bằng 0 nhận giá trị 0 (như zero_division=0).
    Đây là quy ước báo cáo, không phải khẳng định phép chia 0/0 bằng 0.
    """
    that = np.asarray(y_that)
    doan = np.asarray(y_doan)
    if that.ndim != 1 or doan.ndim != 1 or that.shape != doan.shape:
        raise ValueError("Hai mang phai mot chieu va co cung kich thuoc.")
    if that.size == 0:
        raise ValueError("Khong danh gia mang rong: accuracy khong xac dinh.")
    if not (np.isin(that, [0, 1]).all() and np.isin(doan, [0, 1]).all()):
        raise ValueError("Ham chi danh gia nhan nhi phan 0 va 1.")

    tp = np.sum((that == 1) & (doan == 1))
    fp = np.sum((that == 0) & (doan == 1))
    fn = np.sum((that == 1) & (doan == 0))
    tn = np.sum((that == 0) & (doan == 0))
    accuracy = (tp + tn) / that.size
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(accuracy), float(precision), float(recall), float(f1)


if __name__ == "__main__":
    df = pd.read_csv("data/sinh_vien_bo_hoc.csv")
    X = df.drop(columns="bo_hoc")
    y = df["bo_hoc"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )
    mo_hinh = {
        "Logistic": make_pipeline(StandardScaler(), LogisticRegression()),
        "Dummy": DummyClassifier(strategy="most_frequent"),
    }
    ket_qua = []
    print("TU TINH BANG NUMPY: Mo hinh | Accuracy | Precision | Recall | F1")
    for ten, mh in mo_hinh.items():
        du = mh.fit(X_tr, y_tr).predict(X_te)
        diem = danh_gia(y_te, du)
        ket_qua.append((ten, du, diem))
        print(f"{ten:10s} | " + " | ".join(f"{v:.4f}" for v in diem))

    # Chỉ dùng các hàm đánh giá sklearn ở bước đối chứng sau khi tự tính.
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    print("\nDOI CHUNG SKLEARN:")
    for ten, du, diem in ket_qua:
        doi_chung = (
            accuracy_score(y_te, du),
            precision_score(y_te, du, zero_division=0),
            recall_score(y_te, du, zero_division=0),
            f1_score(y_te, du, zero_division=0),
        )
        np.testing.assert_allclose(diem, doi_chung)
        print(f"{ten:10s} | " + " | ".join(f"{v:.4f}" for v in doi_chung) + " | KHOP")

    print("\nGiai thich:")
    print("TP: bao dung bo hoc; FP: bao nham; FN: bo sot; TN: doan dung hoc tiep.")
    print("Accuracy=(TP+TN)/N; Precision=TP/(TP+FP); Recall=TP/(TP+FN).")
    print("Dummy luon doan 0: TP=FP=0, khong co loi canh bao nao de do do chinh xac.")
    print("Precision=0/0 khong xac dinh. Ham quy uoc tra 0, doi chung zero_division=0.")
    print("Recall=0 vi co sinh vien bo hoc that nhung khong phat hien duoc ai.")
    print("Neu khong co nhan duong that, Recall cung khong xac dinh va quy uoc la 0.")
    print("Neu P+R=0, quy uoc F1=0; mang rong bi tu choi thay vi bao accuracy gia.")
    print("Accuracy cao cua Dummy chi phan anh lop 0 dong; khong chung to tim bo hoc tot.")
