"""Append-only JSONL audit trail: one line per pipeline step, with timing.

This is what you hand a hospital's security or compliance reviewer:
who sent what, where it went, how long each hop took, and what failed.
No patient names or IDs are written here -- only UIDs and step results.
"""
from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path


class Audit:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def log(self, study_uid: str, step: str, **detail) -> None:
        line = {"ts": round(time.time(), 3), "study": study_uid, "step": step, **detail}
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, default=str) + "\n")

    @contextmanager
    def step(self, study_uid: str, step: str, **detail):
        t0 = time.perf_counter()
        extra: dict = {}
        try:
            yield extra
        except Exception as e:
            self.log(study_uid, step, ok=False, ms=_ms(t0), error=f"{type(e).__name__}: {e}", **detail)
            raise
        self.log(study_uid, step, ok=True, ms=_ms(t0), **detail, **extra)


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)
