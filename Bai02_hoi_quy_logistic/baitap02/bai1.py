# -*- coding: utf-8 -*-
"""Bài tập 1: Thống kê theo nhóm."""

import pandas as pd


df = pd.read_csv("data/sinh_vien.csv")

nhom_7_tro_len = df[df["diem_giua_ky"] >= 7]
nhom_con_lai = df[df["diem_giua_ky"] < 7]

so_ban = len(nhom_7_tro_len)
ty_le_7 = nhom_7_tro_len["qua_mon"].mean()
ty_le_con_lai = nhom_con_lai["qua_mon"].mean()

print("BAI 1 - THONG KE THEO NHOM")
print(f"So ban co diem giua ky tu 7 tro len: {so_ban}")
print(f"Ty le qua mon cua nhom >= 7 diem: {ty_le_7:.4f}")
print(f"Ty le qua mon cua nhom < 7 diem : {ty_le_con_lai:.4f}")
print()

chenh_lech = ty_le_7 - ty_le_con_lai
if chenh_lech > 0:
    print(
        "Nhan xet: Nhom co diem giua ky >= 7 co ty le qua mon cao hon "
        f"{chenh_lech:.4f}, nen diem giua ky co kha nang phan biet hai nhom."
    )
elif chenh_lech < 0:
    print(
        "Nhan xet: Ket qua khong theo xu huong mong doi; can xem lai du lieu "
        "va muc do phan biet cua diem giua ky."
    )
else:
    print("Nhan xet: Hai nhom co ty le qua mon bang nhau, nen dac trung nay khong phan biet duoc hai nhom.")
