# -*- coding: utf-8 -*-
"""Bài tập 2: Vẽ hàm sigmoid với z từ -8 tới 8."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


z = np.linspace(-8, 8, 400)
y = sigmoid(z)

Path("figures").mkdir(exist_ok=True)
duong_dan = Path("figures") / "sigmoid.png"

plt.figure(figsize=(8, 5))
plt.plot(z, y, label="sigmoid(z)")
plt.axhline(0.5, linestyle="--", label="y = 0.5")
plt.axvline(0, linestyle="--", label="z = 0")
plt.xlabel("z")
plt.ylabel("sigmoid(z)")
plt.title("Ham sigmoid")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(duong_dan, dpi=150)
plt.close()

print("BAI 2 - VE HAM SIGMOID")
print(f"Da luu hinh tai: {duong_dan}")
