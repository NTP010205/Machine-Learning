# -*- coding: utf-8 -*-
"""Bài 2: báo cáo từng chữ số, giữ SVM và cách chia của Lab 6."""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
mh = SVC(kernel="rbf")
du = mh.fit(X_tr, y_tr).predict(X_te)
print(classification_report(y_te, du, labels=np.arange(10), digits=4, zero_division=0))
bao_cao = classification_report(
    y_te, du, labels=np.arange(10), output_dict=True, zero_division=0
)
for chi_so in ["recall", "precision"]:
    thap_nhat = min(bao_cao[str(i)][chi_so] for i in range(10))
    chu_so = [i for i in range(10) if np.isclose(bao_cao[str(i)][chi_so], thap_nhat)]
    print(f"Chu so co {chi_so} thap nhat: {chu_so}, gia tri {thap_nhat:.4f}")
so_mau = [int(bao_cao[str(i)]["support"]) for i in range(10)]
print("So mau tung chu so 0..9:", so_mau)
print("Macro avg: moi lop co trong so 1/10; weighted avg: trong so la so mau lop/N.")
print(f"Moi lop co {min(so_mau)}..{max(so_mau)} mau, nen hai cach lay trung binh gan nhau.")
print("Dieu nay khong con dung neu so mau giua cac lop chenh lech lon.")
