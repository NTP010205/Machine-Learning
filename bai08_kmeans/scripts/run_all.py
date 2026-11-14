# -*- coding: utf-8 -*-
"""Chạy 11 chương trình; exit 0=đủ, 1=lỗi, 2=thiếu phần ảnh cá nhân Bài 4."""

import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "code/k1_tu_viet.py", "code/k2_khach_hang.py", "code/k3_chon_k.py",
    "code/k4_ba_cai_bay.py", "code/k5_nen_anh.py",
    "baitap08/bai1.py", "baitap08/bai2.py", "baitap08/bai3.py",
    "baitap08/bai4.py", "baitap08/bai5.py", "baitap08/bai6.py",
]


def main():
    print("Interpreter:", sys.executable, flush=True)
    print("Thu muc lam viec:", ROOT, flush=True)
    for ten in ["khach_hang.csv", "hai_vang_trang.csv"]:
        if not (ROOT / "data" / ten).is_file():
            raise FileNotFoundError(f"Thieu CSV goc: {ROOT / 'data' / ten}")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    loi, chua_du = [], []
    for ten in FILES:
        print(f"\n===== CHAY {ten} =====", flush=True)
        try:
            # Kế thừa stdout/stderr để giữ đầy đủ traceback nếu bài nào lỗi.
            r = subprocess.run([sys.executable, "-u", ten], cwd=ROOT, env=env)
            if ten == "baitap08/bai4.py" and r.returncode == 2:
                chua_du.append(ten)
                print("CHUA DU: Bai 4 thieu anh ca nhan; phan china.jpg da chay.", flush=True)
            elif r.returncode != 0:
                loi.append((ten, r.returncode))
                print(f"LOI: {ten}, exit={r.returncode}", flush=True)
        except OSError:
            traceback.print_exc()
            loi.append((ten, "khong khoi dong duoc"))
    if loi:
        print("THAT BAI:", loi, "; phan chua du:", chua_du, flush=True)
        return 1
    if chua_du:
        print("\nDA CHAY 11/11 CHUONG TRINH, KHONG LOI THUC THI.", flush=True)
        print("CHUA HOAN THANH 100% LAB: Bai 4 con thieu anh ca nhan (exit 2).", flush=True)
        return 2
    print("\nHOAN TAT: 11/11 chuong trinh, du ca anh ca nhan Bai 4.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
