"""Chạy 5 phần thực hành và 6 bài tập bằng cùng Python interpreter."""

import os
from pathlib import Path
import subprocess
import sys


DANH_SACH = [
    "code/e1_khoang_cach.py",
    "code/e2_chuan_hoa.py",
    "code/e3_chon_k.py",
    "code/e4_so_sanh.py",
    "code/e5_knn_hoi_quy.py",
    "baitap04/bai1.py",
    "baitap04/bai2.py",
    "baitap04/bai3.py",
    "baitap04/bai4.py",
    "baitap04/bai5.py",
    "baitap04/bai6.py",
]


def main():
    thu_muc_goc = Path(__file__).resolve().parents[1]
    moi_truong = os.environ.copy()
    moi_truong["PYTHONIOENCODING"] = "utf-8"
    moi_truong["PYTHONDONTWRITEBYTECODE"] = "1"
    cac_bai_loi = []
    for tep in DANH_SACH:
        print(f"\n{'=' * 65}\nCHAY: {tep}\n{'=' * 65}", flush=True)
        try:
            ket_qua = subprocess.run(
                [sys.executable, "-u", tep], cwd=thu_muc_goc, env=moi_truong,
                capture_output=True, text=True, encoding="utf-8", check=False,
            )
        except OSError as loi:
            print(f"LOI {tep}: {loi}", file=sys.stderr, flush=True)
            cac_bai_loi.append(tep)
            continue
        print(ket_qua.stdout, end="", flush=True)
        if ket_qua.stderr:
            print(ket_qua.stderr, end="", file=sys.stderr, flush=True)
        if ket_qua.returncode != 0:
            print(f"LOI {tep}: exit code {ket_qua.returncode}", file=sys.stderr, flush=True)
            cac_bai_loi.append(tep)
    if cac_bai_loi:
        print("\nCAC BAI BI LOI: " + ", ".join(cac_bai_loi), file=sys.stderr)
        return 1
    print(f"\nHOAN TAT: {len(DANH_SACH)}/{len(DANH_SACH)} bai thanh cong.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
