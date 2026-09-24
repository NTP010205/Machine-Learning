# -*- coding: utf-8 -*-
"""Bài tập 3: đổi biến đầu vào sang tuổi nhà."""

import numpy as np
import pandas as pd

df = pd.read_csv("data/gia_nha.csv")

x = df["tuoi_nha"].to_numpy()
y = df["gia"].to_numpy()

x_tb = x.mean()
y_tb = y.mean()

# Công thức bình phương tối thiểu cho đường thẳng một biến
tu_so = ((x - x_tb) * (y - y_tb)).sum()
mau_so = ((x - x_tb) ** 2).sum()

w = tu_so / mau_so
b = y_tb - w * x_tb

print(f"He so goc w = {w:.6f}")
print(f"He so chan b = {b:.6f}")
print()

print("Nhan xet: w mang dau am, nghia la theo mo hinh,"
      " khi tuoi nha tang thi gia nha co xu huong giam.")