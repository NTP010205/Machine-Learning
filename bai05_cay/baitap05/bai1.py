"""Bài 1: vẽ cây năm lá và cây sâu 2, giải thích phép chia nhánh trái."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Lưu hình trực tiếp, không mở cửa sổ chặn run_all.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


def ve_cay(cay, ten_cot, tieu_de, duong_dan):
    fig, ax = plt.subplots(figsize=(16, 9))
    plot_tree(
        cay, feature_names=ten_cot, class_names=["Trễ hạn", "Đúng hạn"],
        filled=True, rounded=True, precision=3, fontsize=11, ax=ax,
    )
    ax.set_title(tieu_de, fontsize=18, pad=20)
    fig.tight_layout()
    fig.savefig(duong_dan, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    df = pd.read_csv("data/khach_vay.csv")
    X = df.drop(columns="tra_dung_han")
    y = df["tra_dung_han"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )
    nam_la = DecisionTreeClassifier(max_leaf_nodes=5, random_state=0).fit(X_train, y_train)
    sau_hai = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X_train, y_train)
    Path("figures").mkdir(exist_ok=True)
    print("BAI 1 - TU VE CAY")
    for cay, tieu_de, tep in [
        (nam_la, "Cây quyết định năm lá", "figures/cay_nam_la.png"),
        (sau_hai, "Cây quyết định có độ sâu tối đa 2", "figures/cay_sau_2.png"),
    ]:
        ve_cay(cay, list(X.columns), tieu_de, tep)
        print(f"Da luu {tep}; do sau={cay.get_depth()}, so la={cay.get_n_leaves()}, "
              f"accuracy test={accuracy_score(y_test, cay.predict(X_test)):.4f}")

    # Xét HAI LÁ BÊN TRÁI của cây max_depth=2, không nhầm với cây năm lá.
    cau_truc = sau_hai.tree_
    nut_trai = cau_truc.children_left[0]
    la_trai = cau_truc.children_left[nut_trai]
    la_phai = cau_truc.children_right[nut_trai]
    nhan_hai_la = []
    print("\nHai la ben trai cua cay sau 2:")
    for la in [la_trai, la_phai]:
        ty_le = cau_truc.value[la][0] / cau_truc.value[la][0].sum()
        so_khach = np.rint(ty_le * cau_truc.n_node_samples[la]).astype(int)
        nhan = sau_hai.classes_[ty_le.argmax()]
        nhan_hai_la.append(nhan)
        print(f"Nut {la}: [tre han, dung han]={so_khach.tolist()}, "
              f"ket luan={'dung han' if nhan == 1 else 'tre han'}, "
              f"P(dung han)={ty_le[1]:.4f}, Gini={cau_truc.impurity[la]:.4f}")
    n = cau_truc.n_node_samples[nut_trai]
    gini_sau = sum(cau_truc.n_node_samples[la] / n * cau_truc.impurity[la]
                   for la in [la_trai, la_phai])
    print(f"Gini truoc chia = {cau_truc.impurity[nut_trai]:.4f}")
    print(f"Gini sau chia co trong so = {gini_sau:.4f}")
    print(f"Gini giam tai nut = {cau_truc.impurity[nut_trai] - gini_sau:.4f}")
    print("\nNHAN XET:")
    print("Hai la co cung ket luan:", "Co" if nhan_hai_la[0] == nhan_hai_la[1] else "Khong")
    print("Cay toi uu muc giam Gini, khong bat buoc hai la phai khac nhan da so.")
    print("Hai nhom co ty le tra dung han khac nhau, nen Gini trung binh giam")
    print("du ca hai van co nhieu khach tre han hon. Xac suat du doan cung khac nhau.")
    print("Neu cho phep sau hon, nhom nay con co the duoc chia tiep nhu cay nam la.")


if __name__ == "__main__":
    main()
