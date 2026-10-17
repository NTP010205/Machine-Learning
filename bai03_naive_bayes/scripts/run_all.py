"""Chạy lần lượt 11 bài, dùng cùng Python và thư mục gốc Lab 3."""

import os
from pathlib import Path
import subprocess
import sys


DANH_SACH_BAI = [
    "code/d1_bayes_tay.py",
    "code/d2_tui_tu.py",
    "code/d3_loc_thu_rac.py",
    "code/d4_lam_tron.py",
    "code/d5_gaussian.py",
    "baitap03/bai1.py",
    "baitap03/bai2.py",
    "baitap03/bai3.py",
    "baitap03/bai4.py",
    "baitap03/bai5.py",
    "baitap03/bai6.py",
]


def main():
    thu_muc_goc = Path(__file__).resolve().parents[1]
    moi_truong = os.environ.copy()
    moi_truong["PYTHONIOENCODING"] = "utf-8"
    # Không tạo bytecode trong các thư viện của môi trường hocmay.
    moi_truong["PYTHONDONTWRITEBYTECODE"] = "1"
    bai_loi = []
    for tep in DANH_SACH_BAI:
        print(f"\n{'=' * 64}\nCHAY: {tep}\n{'=' * 64}", flush=True)
        try:
            ket_qua = subprocess.run(
                [sys.executable, "-u", tep], cwd=thu_muc_goc,
                env=moi_truong, capture_output=True, text=True, encoding="utf-8",
                check=False,
            )
        except OSError as loi:
            print(f"LOI {tep}: {loi}", file=sys.stderr, flush=True)
            bai_loi.append(tep)
            continue
        print(ket_qua.stdout, end="", flush=True)
        if ket_qua.stderr:
            print(ket_qua.stderr, end="", file=sys.stderr, flush=True)
        if ket_qua.returncode != 0:
            print(f"LOI {tep}: exit code {ket_qua.returncode}", file=sys.stderr, flush=True)
            bai_loi.append(tep)
    if bai_loi:
        print("\nCAC BAI BI LOI: " + ", ".join(bai_loi), file=sys.stderr)
        return 1
    print(f"\nHOAN TAT: {len(DANH_SACH_BAI)}/{len(DANH_SACH_BAI)} bai thanh cong.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
