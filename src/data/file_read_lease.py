"""Original read handles retained across hashing and actual lazy consumption."""
from __future__ import annotations

import hashlib
import os
import stat
import weakref
from contextlib import ExitStack
from pathlib import Path


def open_read_descriptor(path, *, allow_replace=False):
    path = Path(path)
    if os.name != "nt":
        return os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    create.restype = wintypes.HANDLE
    # All readers may coexist; writable handles are denied. BIDS/lazy readers
    # also deny path replacement for the entire live source lifetime.
    share = 1 | (4 if allow_replace else 0)
    handle = create(str(path.resolve()), 0x80000000, share, None, 3, 0x00200080, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel.CloseHandle(handle)
        raise


def file_identity(status):
    return status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns


class OriginalFileReadLease:
    """One actual hash per file, with its read handle alive until close()."""

    def __init__(self, paths=()):
        self._stack = ExitStack()
        self._finish = weakref.finalize(self, self._stack.close)
        self._files = {}
        self._closed = False
        try:
            for path in sorted(set(map(Path, paths))):
                self.add(path)
        except BaseException:
            self.close()
            raise

    def add(self, path):
        if self._closed:
            raise ValueError("the original read lease is closed")
        path = Path(path).resolve(strict=True)
        if path in self._files:
            self.assert_live()
            return
        path_before = file_identity(path.stat())
        stream = self._stack.enter_context(os.fdopen(open_read_descriptor(path), "rb"))
        before = file_identity(os.fstat(stream.fileno()))
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode) or before[:4] != path_before[:4]:
            raise ValueError("original read handle differs from its declared file object")
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
        self._files[path] = (stream, before, path_before, digest.hexdigest())
        self.assert_live()

    def assert_live(self):
        if self._closed:
            raise ValueError("the original read lease is closed")
        for path, (stream, descriptor_identity, path_identity, _) in self._files.items():
            if (file_identity(os.fstat(stream.fileno())) != descriptor_identity
                    or file_identity(path.stat()) != path_identity):
                raise ValueError("original source changed during its live read lease")

    def binding(self, path):
        self.assert_live()
        _, _, identity, digest = self._files[Path(path).resolve(strict=True)]
        return {"sha256": digest, "byte_size": identity[2], "file_identity": identity}

    def read_bytes(self, path):
        self.assert_live()
        stream, _, _, digest = self._files[Path(path).resolve(strict=True)]
        stream.seek(0)
        content = stream.read()
        self.assert_live()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError("original metadata bytes differ from the admitted read handle")
        return content

    def close(self):
        if not self._closed:
            self._closed = True
            self._finish()

    def __enter__(self):
        try:
            self.assert_live()
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *_):
        self.close()
