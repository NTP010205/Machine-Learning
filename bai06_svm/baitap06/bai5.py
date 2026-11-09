"""Bài 5: ma trận nhầm lẫn và ảnh sai thật từ đúng tập test của g5."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Lưu ảnh, không mở cửa sổ làm dừng run_all.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def main():
    X, y = load_digits(return_X_y=True)
    # Giữ chỉ số nguồn để có thể truy từng ảnh về bộ load_digits.
    X_train, X_test, y_train, y_test, _, chi_so_test = train_test_split(
        X, y, np.arange(len(y)), test_size=0.25, random_state=17, stratify=y
    )
    svm = SVC(kernel="rbf").fit(X_train, y_train)
    du_doan = svm.predict(X_test)
    ma_tran = confusion_matrix(y_test, du_doan, labels=np.arange(10))
    print("BAI 5 - PHAN TICH LOI CHU SO")
    print(f"Accuracy = {accuracy_score(y_test, du_doan):.4f}")
    print("Confusion matrix: hang = nhan that, cot = du doan (0..9)")
    print(pd.DataFrame(ma_tran, index=range(10), columns=range(10)).to_string())
    print("\nCac cap nham (that -> du doan):")
    for that in range(10):
        for doan in range(10):
            if that != doan and ma_tran[that, doan]:
                print(f"{that} -> {doan}: {ma_tran[that, doan]} anh")
    sai = np.flatnonzero(du_doan != y_test)
    print(f"\nSo anh sai: {len(sai)}/{len(y_test)}")
    Path("figures").mkdir(exist_ok=True)
    for i in sai:
        tep = Path("figures") / f"chu_so_sai_test_{i:03d}_that_{y_test[i]}_doan_{du_doan[i]}.png"
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(X_test[i].reshape(8, 8), cmap="gray_r", interpolation="nearest", vmin=0, vmax=16)
        ax.set_title(f"Thật: {y_test[i]} | Dự đoán: {du_doan[i]}", fontsize=15, pad=12)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.text(0.5, 0.03, f"Test #{i} | Chỉ số nguồn: {chi_so_test[i]}", ha="center", fontsize=10)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        fig.savefig(tep, dpi=160)
        plt.close(fig)
        print(f"Test #{i}, nguon #{chi_so_test[i]}: da luu {tep.as_posix()}")
    print("\nNHAN XET:")
    print("Cac o ngoai duong cheo cho biet huong nham; A -> B khong dong nghia B -> A.")
    print("Anh 8x8 co do phan giai thap: net mo, dut hoac khep vong khong ro co the")
    print("lam hinh dang gan voi lop khac. Can xem tung anh that da luu de doi chieu.")
    print("So loi rat it tren mot phep chia; khong du ket luan day la cac chu so")
    print("luon de nham nhat. Day la phan tich test, khong dung de chon lai tham so.")


if __name__ == "__main__":
    main()
