from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock


@contextmanager
def file_lock(lock_path: Path):
    with FileLock(str(lock_path)):
        yield
