from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock


@contextmanager
def file_lock(lock_path: Path) -> Generator[None, None, None]:
    with FileLock(str(lock_path)):
        yield
