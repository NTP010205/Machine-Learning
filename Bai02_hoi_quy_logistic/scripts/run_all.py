# -*- coding: utf-8 -*-
"""Chạy lần lượt toàn bộ mã của Bài 2 từ thư mục gốc."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "sinh_vien.csv"

if not DATA.exists():
    print("Chua co data/sinh_vien.csv.")
    print("Hay chep tep sinh_vien.csv cua mon hoc vao thu muc data roi chay lai.")
    raise SystemExit(1)

files = [
    "code/c1_doc_du_lieu.py",
    "code/c2_sigmoid.py",
    "code/c3_khop_mo_hinh.py",
    "code/c4_danh_gia.py",
    "code/c5_doi_nguong.py",
    "code/c6_hai_bien.py",
    "baitap02/bai1.py",
    "baitap02/bai2.py",
    "baitap02/bai3.py",
    "baitap02/bai4.py",
    "baitap02/bai5.py",
    "baitap02/bai6.py",
]

for rel in files:
    print("\n" + "=" * 78)
    print(f"CHAY: {rel}")
    print("=" * 78)
    result = subprocess.run([sys.executable, rel], cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
