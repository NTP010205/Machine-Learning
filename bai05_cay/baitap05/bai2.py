"""Bài 2: tự thử mọi câu hỏi chia nhị phân bằng chỉ số Gini."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def gini(nhan):
    """Gini bằng 1 trừ tổng bình phương tỷ lệ từng lớp."""
    if len(nhan) == 0:
        return 0.0
    _, so_luong = np.unique(nhan, return_counts=True)
    ty_le = so_luong / len(nhan)
    return 1.0 - np.sum(ty_le ** 2)


def tim_cau_hoi(X, y):
    """Trả về tên cột, ngưỡng và mức giảm Gini lớn nhất trên dữ liệu nhận vào."""
    nhan = np.asarray(y)
    if len(X) != len(nhan) or len(nhan) < 2:
        raise ValueError("X, y phai cung so mau va co it nhat hai mau.")
    gini_goc = gini(nhan)
    cot_tot, nguong_tot, giam_tot = None, None, -np.inf
    for cot in X.columns:
        gia_tri = X[cot].to_numpy(dtype=float)
        cac_muc = np.unique(gia_tri)
        # Giá trị trùng nhau không tạo thêm câu hỏi; cột hằng không có ngưỡng.
        for a, b in zip(cac_muc[:-1], cac_muc[1:]):
            nguong = (a + b) / 2
            di_trai = gia_tri <= nguong
            n_trai = np.count_nonzero(di_trai)
            if n_trai == 0 or n_trai == len(nhan):
                continue
            gini_sau = (n_trai / len(nhan) * gini(nhan[di_trai])
                        + (len(nhan) - n_trai) / len(nhan) * gini(nhan[~di_trai]))
            giam = gini_goc - gini_sau
            if giam > giam_tot:
                cot_tot, nguong_tot, giam_tot = cot, nguong, giam
    if cot_tot is None:
        raise ValueError("Khong co nguong chia hop le: cac cot deu hang.")
    return cot_tot, nguong_tot, giam_tot


def main():
    df = pd.read_csv("data/khach_vay.csv")
    X = df.drop(columns="tra_dung_han")
    y = df["tra_dung_han"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )
    cot, nguong, giam = tim_cau_hoi(X_train, y_train)
    print("BAI 2 - TU TIM CAU HOI GOC")
    print(f"So khach train = {len(X_train)}; Gini goc = {gini(y_train):.6f}")
    print(f"Tu viet: {cot} <= {nguong:.2f}, Gini giam = {giam:.6f}")
    # sklearn chỉ làm đối chứng sau khi thuật toán tự viết đã tìm xong.
    cay = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X_train, y_train)
    nut = cay.tree_
    trai, phai = nut.children_left[0], nut.children_right[0]
    giam_sklearn = nut.impurity[0] - sum(
        nut.n_node_samples[i] / nut.n_node_samples[0] * nut.impurity[i]
        for i in [trai, phai]
    )
    cot_sklearn = X.columns[nut.feature[0]]
    print(f"sklearn: {cot_sklearn} <= {nut.threshold[0]:.2f}, Gini giam = {giam_sklearn:.6f}")
    print("Khop cau hoi goc:", cot == cot_sklearn and np.isclose(nguong, nut.threshold[0]))
    print("\nNHAN XET:")
    print("1. Gini = 1 - tong(p_lop^2): nut cang thuan thi Gini cang nho.")
    print("2. Moi cot duoc sap cac gia tri khac nhau; thu trung diem giua hai muc lien ke.")
    print("   Cac nguong trong cung khoang cho cung phep chia, nen khong can thu vo han.")
    print("3. Gini sau chia = (n_trai/n)*Gini_trai + (n_phai/n)*Gini_phai.")
    print("   Phai can theo so mau, khong lay trung binh hai nhanh voi trong so 1/2.")
    print("4. Chon muc giam Gini lon nhat tren TAT CA cot; khong dung du lieu test.")
    print("   Day la chon tham lam tai nut goc, chua toi uu dong thoi ca cay.")
    print("   Neu hoa, ham giu cau hoi gap truoc; sklearn co the pha hoa khac.")


if __name__ == "__main__":
    main()
