# -*- coding: utf-8 -*-
"""Chạy đủ 12 bài bằng chính interpreter hiện tại, trả mã lỗi nếu bài nào lỗi."""

import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "code/h1_do_bua.py",
    "code/h2_ma_tran.py",
    "code/h3_precision_recall.py",
    "code/h4_nguong.py",
    "code/h5_roc_auc.py",
    "code/h6_can_bang.py",
    "baitap07/bai1.py",
    "baitap07/bai2.py",
    "baitap07/bai3.py",
    "baitap07/bai4.py",
    "baitap07/bai5.py",
    "baitap07/bai6.py",
]


def main():
    print("Python:", sys.executable, flush=True)
    print("Thu muc lam viec:", ROOT, flush=True)
    for name in ["sinh_vien_bo_hoc.csv", "gia_nha.csv"]:
        path = ROOT / "data" / name
        if not path.is_file():
            raise FileNotFoundError(f"Thieu CSV: {path}. Can ban goc do giang vien cung cap.")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    loi = []
    for name in FILES:
        print(f"\n===== CHAY {name} =====", flush=True)
        try:
            # Không capture stderr: traceback của tiến trình con hiện trực tiếp.
            result = subprocess.run([sys.executable, "-u", name], cwd=ROOT, env=env)
            if result.returncode != 0:
                loi.append((name, result.returncode))
                print(f"LOI: {name}, exit={result.returncode}", flush=True)
        except OSError:
            traceback.print_exc()
            loi.append((name, "khong khoi dong duoc"))
    if loi:
        print(f"THAT BAI: {len(loi)}/{len(FILES)} chuong trinh bi loi:", loi, flush=True)
        return 1
    print(f"\nHOAN TAT: {len(FILES)}/{len(FILES)} chuong trinh thanh cong.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
