"""Bài 4: dò lưới mịn và so với lưới thưa của PDF trên cùng phép chia."""

import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC


def main():
    df = pd.read_csv("data/hai_vang_trang.csv")
    X, y = df[["x1", "x2"]], df["nhan"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )
    luoi_min = {"C": [3, 10, 30], "gamma": [0.3, 1, 3]}
    tim = GridSearchCV(SVC(kernel="rbf"), luoi_min, cv=5, error_score="raise")
    tim.fit(X_train, y_train)
    # Tính lại mốc CV từ dữ liệu thật, không gán cứng kết quả tham chiếu.
    tho = GridSearchCV(SVC(kernel="rbf"),
                       {"C": [0.1, 1, 10, 100], "gamma": [0.01, 0.1, 1, 10]},
                       cv=5, error_score="raise").fit(X_train, y_train)
    bang = pd.DataFrame(tim.cv_results_["params"])
    bang["CV"] = tim.cv_results_["mean_test_score"]
    print("BAI 4 - DO LUOI MIN")
    print(bang.pivot(index="C", columns="gamma", values="CV").round(4).to_string())
    acc_min, acc_tho = tim.score(X_test, y_test), tho.score(X_test, y_test)
    print(f"\nLuoi min: {tim.best_params_}, CV={tim.best_score_:.4f}, test={acc_min:.4f}")
    print(f"Luoi tho: {tho.best_params_}, CV={tho.best_score_:.4f}, test={acc_tho:.4f}")
    print("\nNHAN XET:")
    print(f"Chenh lech CV = {tim.best_score_ - tho.best_score_:+.4f}; test = {acc_min - acc_tho:+.4f}.")
    print("Luoi min thu ky hon quanh vung tot cua luoi tho, co the tim duoc cap bi bo qua.")
    print("Diem CV cao hon khong dam bao diem test cao hon; khong dieu chinh luoi theo test.")
    print("Moi cap can 5 lan fit CV; thu qua nhieu cap ton thoi gian va co the qua khop CV.")
    print("CV hoa diem thi GridSearchCV giu cap dung truoc; khong pha hoa bang test.")


if __name__ == "__main__":
    main()
