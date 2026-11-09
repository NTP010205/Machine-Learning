"""Kiểm tra CSV và chạy đủ 11 bài với cùng Python, từ gốc Lab 6."""

import os
from pathlib import Path
import subprocess
import sys


DANH_SACH = [
    "code/g1_le_rong_nhat.py", "code/g2_tham_so_c.py", "code/g3_kernel.py",
    "code/g4_chon_tham_so.py", "code/g5_chu_so.py",
    "baitap06/bai1.py", "baitap06/bai2.py", "baitap06/bai3.py",
    "baitap06/bai4.py", "baitap06/bai5.py", "baitap06/bai6.py",
]


def main():
    thu_muc_goc = Path(__file__).resolve().parents[1]
    cac_csv = ["data/sinh_vien.csv", "data/sinh_vien_mo_rong.csv", "data/hai_vang_trang.csv"]
    thieu = [tep for tep in cac_csv if not (thu_muc_goc / tep).is_file()]
    if thieu:
        print("LOI: thieu CSV: " + ", ".join(thieu), file=sys.stderr)
        return 1
    moi_truong = os.environ.copy()
    moi_truong["PYTHONIOENCODING"] = "utf-8"
    moi_truong["PYTHONDONTWRITEBYTECODE"] = "1"
    cac_bai_loi = []
    for tep in DANH_SACH:
        print(f"\n{'=' * 65}\nCHAY: {tep}\n{'=' * 65}", flush=True)
        try:
            # Kế thừa stdout/stderr để hiện ngay kết quả, warning và traceback.
            ket_qua = subprocess.run([sys.executable, "-u", tep], cwd=thu_muc_goc,
                                     env=moi_truong, check=False)
        except OSError as loi:
            print(f"LOI {tep}: {loi}", file=sys.stderr, flush=True)
            cac_bai_loi.append(tep)
            continue
        if ket_qua.returncode != 0:
            print(f"LOI {tep}: exit code {ket_qua.returncode}", file=sys.stderr, flush=True)
            cac_bai_loi.append(tep)
    if cac_bai_loi:
        print("CAC BAI BI LOI: " + ", ".join(cac_bai_loi), file=sys.stderr)
        return 1
    print(f"\nHOAN TAT: {len(DANH_SACH)}/{len(DANH_SACH)} bai thanh cong.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
