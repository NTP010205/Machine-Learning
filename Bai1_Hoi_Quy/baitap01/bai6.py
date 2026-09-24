# -*- coding: utf-8 -*-
"""Bài tập 6: viết hàm dự đoán có cảnh báo."""

w = 0.078367
b = 0.401752


def du_doan_gia(dien_tich):
    if dien_tich < 35.5 or dien_tich > 117.5:
        print("Canh bao: dien tich nam ngoai khoang du lieu 35.5 - 117.5 m2.")

    gia_du_doan = w * dien_tich + b
    return gia_du_doan


for dien_tich in [60, 80, 200]:
    gia = du_doan_gia(dien_tich)
    print(f"Can {dien_tich} m2 duoc du doan: {gia:.3f} ty dong")