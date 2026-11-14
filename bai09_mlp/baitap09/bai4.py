# -*- coding: utf-8 -*-
"""Bài 4: bốn tốc độ học Adam; đường loss lấy trực tiếp từ loss_curve_."""

# Cấu hình thực thi cục bộ; không sửa thư viện hoặc tham số mô hình.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cau_hinh_moi_truong import cau_hinh
cau_hinh()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
bang = []
print("Adam, hidden=(100,), max_iter=1000, random_state=0, alpha=0.0001.", flush=True)
print("learning_rate_init | n_iter | loss cuoi | accuracy test", flush=True)
for toc_do in [0.0001, 0.001, 0.01, 0.1]:
    mh = make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(100,), solver="adam", max_iter=1000,
        random_state=0, learning_rate_init=toc_do
    ))
    mh.fit(X_tr, y_tr)
    mlp = mh[-1]
    loss = np.asarray(mlp.loss_curve_)
    acc = accuracy_score(y_te, mh.predict(X_te))
    bang.append((toc_do, mlp.n_iter_, loss[-1], acc))
    print(f"{toc_do:18g} | {mlp.n_iter_:6d} | {loss[-1]:.8f} | {acc:.4f}", flush=True)
    ax.plot(np.arange(1, len(loss) + 1), loss,
            label=f"lr={toc_do:g} ({mlp.n_iter_} vòng, test={acc:.4f})", linewidth=1.6)
ax.set(xlabel="Vòng học", ylabel="Loss trên train (thang log)", yscale="log",
       title="MLP 100 nơ-ron: ảnh hưởng của tốc độ học Adam")
ax.grid(alpha=0.25, which="both")
ax.legend(fontsize=9)
Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/toc_do_hoc.png", dpi=180)
plt.close(fig)
pd.DataFrame(bang, columns=["learning_rate_init", "n_iter", "loss_cuoi", "accuracy_test"]).to_csv(
    "figures/toc_do_hoc.csv", index=False
)
print("Da luu figures/toc_do_hoc.png va bang so lieu figures/toc_do_hoc.csv.")
print("Learning rate nho: buoc cap nhat nho, thuong can nhieu vong hon de giam loss.")
print("Learning rate lon: co the giam loss nhanh luc dau, nhung cung co the vuot vung tot va dao dong.")
print("Dung som hon khong dong nghia hoi tu tot hon; doc ca loss va accuracy, khong chi n_iter_.")
print("Adam dieu chinh buoc theo tung tham so, nhung learning_rate_init van quyet dinh muc buoc co so.")
print("Giu cung max_iter, seed, split va cac tham so khac de so sanh; khong tao duong loss gia.")
