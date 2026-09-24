# -*- coding: utf-8 -*-
"""Bài tập 2: vẽ biểu đồ phân tán theo số phòng."""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/gia_nha.csv")

plt.scatter(df["so_phong"], df["gia"])

plt.xlabel("So phong")
plt.ylabel("Gia (ty dong)")
plt.title("Gia nha theo so phong")

plt.savefig("bai2.png", dpi=150)

print("Nhan xet: Nhin chung so phong cang nhieu thi gia can ho co xu huong cao hon,"
      " tuy nhien van co su chong lan ve gia giua cac nhom.")

plt.show()