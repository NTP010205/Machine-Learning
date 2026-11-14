# -*- coding: utf-8 -*-
"""Chạy đủ 11 chương trình bằng interpreter hiện tại, không bỏ qua lỗi."""

import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "code/m1_lan_truyen_tien.py", "code/m2_xor.py", "code/m3_do_lon_mang.py",
    "code/m4_chuan_hoa.py", "code/m5_chu_so.py",
    "baitap09/bai1.py", "baitap09/bai2.py", "baitap09/bai3.py",
    "baitap09/bai4.py", "baitap09/bai5.py", "baitap09/bai6.py",
]


def main():
    print("Interpreter:", sys.executable, flush=True)
    print("Thu muc lam viec:", ROOT, flush=True)
    for name in ["xor.csv", "hai_vang_trang.csv", "sinh_vien_mo_rong.csv", "gia_nha.csv"]:
        if not (ROOT / "data" / name).is_file():
            raise FileNotFoundError(f"Thieu CSV bat buoc: {ROOT / 'data' / name}")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    loi = []
    for name in FILES:
        print(f"\n===== CHAY {name} =====", flush=True)
        if name == "baitap09/bai5.py":
            print("GridSearchCV: 12 cau hinh x 5 fold + refit; vui long doi chay du.", flush=True)
        try:
            # Kế thừa stdout/stderr: giữ đầy đủ cảnh báo và traceback từ bài con.
            r = subprocess.run([sys.executable, "-u", name], cwd=ROOT, env=env)
            if r.returncode:
                loi.append((name, r.returncode))
                print(f"LOI: {name}, exit={r.returncode}", flush=True)
        except OSError:
            traceback.print_exc()
            loi.append((name, "khong khoi dong duoc"))
    if loi:
        print("THAT BAI:", loi, flush=True)
        return 1
    print("\nHOAN TAT: 11/11 chuong trinh thanh cong. Can doc rieng canh bao neu co.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
