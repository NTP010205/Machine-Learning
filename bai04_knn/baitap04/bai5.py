"""Bài 5: chọn k hồi quy theo CV MAE trên train, sau đó mới chấm test."""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor


def main():
    df = pd.read_csv("data/gia_nha.csv")
    X = df[["dien_tich"]]
    y = df["gia"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("BAI 5 - CHON k CHO KNN REGRESSION")
    print(f"Train = {len(X_train)}, test = {len(X_test)}")
    print(f"{'k':>4s} {'CV MAE (ty dong)':>20s}")
    mae_cv = {}
    for k in range(1, 21):
        # Dùng tìm kiếm chính xác bằng KD-tree. Trên môi trường Windows hiện tại,
        # auto chuyển sang brute ở k=19–20 và gặp lỗi đường dẫn DLL có dấu
        # trong threadpoolctl. Không sửa môi trường hay bỏ các giá trị k này.
        diem_am = cross_val_score(
            KNeighborsRegressor(n_neighbors=k, algorithm="kd_tree"), X_train, y_train,
            cv=5, scoring="neg_mean_absolute_error", error_score="raise"
        )
        # sklearn dùng điểm càng lớn càng tốt, nên MAE được trả về với dấu âm.
        mae_cv[k] = -diem_am.mean()
        print(f"{k:4d} {mae_cv[k]:20.4f}")
    k_tot = min(mae_cv, key=mae_cv.get)
    # Chọn xong k mới dự đoán test; không điều chỉnh k theo kết quả test.
    knn = KNeighborsRegressor(n_neighbors=k_tot, algorithm="kd_tree").fit(X_train, y_train)
    duong_thang = LinearRegression().fit(X_train, y_train)
    mae_knn = mean_absolute_error(y_test, knn.predict(X_test))
    mae_duong = mean_absolute_error(y_test, duong_thang.predict(X_test))
    print(f"\nk tot nhat = {k_tot}; CV MAE = {mae_cv[k_tot]:.4f}")
    print(f"KNN test MAE = {mae_knn:.4f}")
    print(f"LinearRegression test MAE = {mae_duong:.4f}")
    print("\nNHAN XET:")
    if mae_knn < mae_duong:
        print("KNN sau chon k VUOT LinearRegression tren test nay (MAE nho hon).")
    elif mae_knn > mae_duong:
        print("KNN sau chon k KHONG vuot LinearRegression tren test nay.")
    else:
        print("Hai mo hinh co MAE bang nhau tren test nay.")
    print("Gia nha tang gan tuyen tinh theo dien tich, nen duong thang la doi chung phu hop.")
    print("Chon k bang CV giup chon trong ho KNN, khong dam bao thang ho mo hinh khac.")


if __name__ == "__main__":
    main()
