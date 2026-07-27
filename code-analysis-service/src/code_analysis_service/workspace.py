from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def workspace_context() -> Iterator[Path]:
    tmp = tempfile.mkdtemp(prefix="code-analysis-")
    try:
        yield Path(tmp)
    finally:
        p = Path(tmp)
        if p.exists():
            import shutil
            shutil.rmtree(p, ignore_errors=True)
