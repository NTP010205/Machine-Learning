# -*- coding: utf-8 -*-
"""Bài 5: 12 cấu hình, 5-fold CV trên train; chỉ chấm test sau khi chọn."""

# Cấu hình thực thi cục bộ; không sửa thư viện hoặc tham số mô hình.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cau_hinh_moi_truong import cau_hinh
cau_hinh()

from time import perf_counter

import pandas as pd
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, ParameterGrid, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
mh = make_pipeline(StandardScaler(), MLPClassifier(
    solver="adam", max_iter=1000, random_state=0
))
luoi = {
    "mlpclassifier__hidden_layer_sizes": [(10,), (50,), (100,), (50, 50)],
    "mlpclassifier__alpha": [0.0001, 0.01, 1.0],
}
so_cau_hinh = len(list(ParameterGrid(luoi)))
print(f"{so_cau_hinh} cau hinh x 5 fold = {so_cau_hinh*5} lan fit CV, them 1 lan refit.", flush=True)
print(f"Chi tim tren {len(y_tr)} mau train; giu rieng {len(y_te)} mau test.", flush=True)
print("Moi fold tu fit StandardScaler trong pipeline. Khong an ConvergenceWarning.", flush=True)
bat_dau = perf_counter()
tim = GridSearchCV(mh, luoi, cv=5, scoring="accuracy", n_jobs=1,
                   refit=True, verbose=2, error_score="raise")
tim.fit(X_tr, y_tr)
thoi_gian = perf_counter() - bat_dau
ket = tim.cv_results_
bang = pd.DataFrame({
    "hidden_layer_sizes": [p["mlpclassifier__hidden_layer_sizes"] for p in ket["params"]],
    "alpha": [p["mlpclassifier__alpha"] for p in ket["params"]],
    "accuracy_cv": ket["mean_test_score"],
    "std_cv": ket["std_test_score"],
    "hang": ket["rank_test_score"],
})
print("\nKET QUA CV CHO DU 12 CAU HINH:")
print(bang.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
print("Tham so tot nhat:", tim.best_params_)
print(f"CV tot nhat: {tim.best_score_:.6f}")
print(f"Thoi gian CV + refit: {thoi_gian:.2f} giay (phu thuoc may).")
# best_estimator_ đã được refit trên toàn bộ train. Chỉ đánh giá test tại đây.
acc_mlp = accuracy_score(y_te, tim.predict(X_te))
svm = SVC(kernel="rbf").fit(X_tr, y_tr)  # Giữ cách dùng pixel gốc như Lab 6/m5.
acc_svm = accuracy_score(y_te, svm.predict(X_te))
print(f"Accuracy test MLP da chon: {acc_mlp:.4f}")
print(f"Accuracy test SVM RBF (tinh lai): {acc_svm:.4f}")
print("MLP", "VUOT" if acc_mlp > acc_svm else "BANG" if acc_mlp == acc_svm else "KHONG VUOT",
      "SVM tren tap test nay.")
print("Lua chon chi dua vao CV train; khong doi cau hinh sau khi xem so sanh test.")
Path("figures").mkdir(exist_ok=True)
bang.to_csv("figures/ket_qua_grid.csv", index=False)
print("Da luu figures/ket_qua_grid.csv (ket qua thuc nghiem, khong phai dataset thay the).")
