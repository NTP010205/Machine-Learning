"""Bài 3: chọn C cho từng bậc đa thức; chọn bậc bằng CV, không dùng test."""

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main():
    df = pd.read_csv("data/hai_vang_trang.csv")
    X, y = df[["x1", "x2"]], df["nhan"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )
    cac_tim = {}
    for bac in [2, 3, 4]:
        pipeline = make_pipeline(StandardScaler(), SVC(kernel="poly", degree=bac))
        tim = GridSearchCV(pipeline, {"svc__C": [0.1, 1, 10, 100]}, cv=5, error_score="raise")
        tim.fit(X_train, y_train)
        cac_tim[bac] = tim
    # Chọn xong bậc bằng CV trước khi đánh giá test của các mô hình.
    bac_tot = max(cac_tim, key=lambda bac: cac_tim[bac].best_score_)
    print("BAI 3 - KERNEL DA THUC, CO CHUAN HOA")
    print(f"{'Bac':>5s} {'C tot':>8s} {'CV':>10s} {'Test':>10s}")
    diem_test = {}
    for bac, tim in cac_tim.items():
        diem_test[bac] = tim.score(X_test, y_test)
        print(f"{bac:5d} {tim.best_params_['svc__C']:8g} {tim.best_score_:10.4f} {diem_test[bac]:10.4f}")
    # Mốc mục 5 dùng dữ liệu thô, C=10 và gamma=1 như PDF.
    rbf = SVC(kernel="rbf", C=10, gamma=1).fit(X_train, y_train)
    acc_rbf = accuracy_score(y_test, rbf.predict(X_test))
    print(f"\nBac tot nhat theo CV: {bac_tot}")
    print(f"RBF muc 5 (C=10, gamma=1): test = {acc_rbf:.4f}")
    print("\nNHAN XET:")
    chenh = diem_test[bac_tot] - acc_rbf
    print(f"Da thuc duoc chon - RBF = {chenh:+.4f}.")
    print("Co vuot RBF tren test nay:", "Co" if chenh > 0 else "Khong")
    print("Bac duoc chon bang CV tren train; bang test chi de bao cao, khong chon lai bac.")
    print("Pipeline fit scaler rieng trong moi fold, tranh ro ri du lieu validation.")
    print("Tang bac khong dam bao tot hon: kernel poly mac dinh co coef0=0,")
    print("cac bac chan/le co dang bien khac nhau; RBF co the hop bien cong hon.")


if __name__ == "__main__":
    main()
