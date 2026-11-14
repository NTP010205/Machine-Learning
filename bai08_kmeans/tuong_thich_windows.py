# -*- coding: utf-8 -*-
"""Sửa đường dẫn DLL từ snapshot Windows trong bộ nhớ của tiến trình Lab 8.

Trên máy này, snapshot của threadpoolctl trả 'Th?c hành' dù đường dẫn thật là
'Thực hành'. Khi đường dẫn snapshot không tồn tại, trả chuỗi rỗng để thư viện
tự dùng GetModuleFileNameW/GetModuleFileNameExW qua handle DLL đã có.
Không sửa file thư viện, không tắt quản lý luồng, không bỏ qua lỗi DLL thật.
"""

import os
from pathlib import Path


def sua_duong_dan_dll():
    if os.name != "nt":
        return
    from threadpoolctl import ThreadpoolController

    goc = getattr(ThreadpoolController, "_snapshot_loaded_modules", None)
    if goc is None or getattr(goc, "_lab08_unicode", False):
        return

    def snapshot_unicode(self, kernel_32):
        return [
            (handle, path if path and Path(path).is_file() else "")
            for handle, path in goc(self, kernel_32)
        ]

    snapshot_unicode._lab08_unicode = True
    ThreadpoolController._snapshot_loaded_modules = snapshot_unicode
