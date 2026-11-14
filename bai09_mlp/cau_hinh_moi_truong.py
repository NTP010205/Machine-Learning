# -*- coding: utf-8 -*-
"""Cấu hình riêng của tiến trình Lab 9, không ghi vào môi trường hocmay.

Giới hạn BLAS/OpenMP một luồng trước khi import NumPy/sklearn: các mạng nhỏ
tránh chi phí chia việc cho quá nhiều luồng. Không đổi siêu tham số mô hình.
Sửa snapshot DLL Windows có đường dẫn 'Th?c hành' bằng nhánh Unicode sẵn có
của threadpoolctl, giống lỗi đã xác minh tại Lab 8. Không bỏ qua lỗi DLL thật.
"""

import os
from pathlib import Path


def cau_hinh():
    for ten in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ[ten] = "1"
    if os.name != "nt":
        return
    from threadpoolctl import ThreadpoolController
    goc = getattr(ThreadpoolController, "_snapshot_loaded_modules", None)
    if goc is None or getattr(goc, "_lab09_unicode", False):
        return

    def snapshot_unicode(self, kernel_32):
        return [(handle, path if path and Path(path).is_file() else "")
                for handle, path in goc(self, kernel_32)]

    snapshot_unicode._lab09_unicode = True
    ThreadpoolController._snapshot_loaded_modules = snapshot_unicode
