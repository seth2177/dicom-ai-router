"""SQLite state: study lifecycle and the de-identification crosswalk.

The crosswalk (real UID <-> pseudonymous UID) is what lets results come
back to the right patient. It never leaves the router.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS studies(
  study_uid TEXT PRIMARY KEY,
  calling_ae TEXT,
  state TEXT,              -- RECEIVING, QUEUED, PROCESSING, DONE, IGNORED, FAILED
  n_instances INTEGER DEFAULT 0,
  rule TEXT, model TEXT, summary TEXT, error TEXT,
  first_seen REAL, updated REAL
);
CREATE TABLE IF NOT EXISTS crosswalk(
  anon_uid TEXT PRIMARY KEY,
  orig_uid TEXT NOT NULL,
  kind TEXT NOT NULL      -- study, series, sop, frame
);
CREATE TABLE IF NOT EXISTS sent_sr(  -- every result SR delivered to PACS: the predecessor chain for revisions
  sop_uid TEXT PRIMARY KEY,
  study_uid TEXT NOT NULL, model TEXT NOT NULL,
  series_uid TEXT NOT NULL, sop_class TEXT NOT NULL,
  digest TEXT NOT NULL,    -- what the SR says (results.content_digest)
  predecessor TEXT,        -- sop_uid of the SR this one revised
  sent REAL
);
"""


class Store:
    def __init__(self, path: Path):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)

    def instance_received(self, study_uid: str, calling_ae: str) -> None:
        now = time.time()
        with self._lock, self._db:
            self._db.execute(
                """INSERT INTO studies(study_uid, calling_ae, state, n_instances, first_seen, updated)
                   VALUES(?,?, 'RECEIVING', 1, ?, ?)
                   ON CONFLICT(study_uid) DO UPDATE SET
                     n_instances = n_instances + 1, updated = excluded.updated,
                     state = CASE WHEN state IN ('DONE','IGNORED','FAILED','QUEUED') THEN 'RECEIVING' ELSE state END""",
                (study_uid, calling_ae, now, now),
            )

    def set_state(self, study_uid: str, state: str, **fields) -> None:
        cols = ", ".join(f"{k} = ?" for k in fields)
        sql = f"UPDATE studies SET state = ?, updated = ?{', ' + cols if cols else ''} WHERE study_uid = ?"
        with self._lock, self._db:
            self._db.execute(sql, (state, time.time(), *fields.values(), study_uid))

    def studies(self) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._db.execute("SELECT * FROM studies ORDER BY first_seen")]

    def put_crosswalk(self, pairs: list[tuple[str, str, str]]) -> None:
        with self._lock, self._db:
            self._db.executemany(
                "INSERT OR REPLACE INTO crosswalk(anon_uid, orig_uid, kind) VALUES(?,?,?)", pairs
            )

    def orig_uid(self, anon_uid: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT orig_uid FROM crosswalk WHERE anon_uid = ?", (anon_uid,)).fetchone()
        return row["orig_uid"] if row else None

    def record_sr(self, study_uid: str, model: str, sr, digest: str, predecessor: str | None) -> None:
        """After PACS accepted it. Re-sending the same SR moves it back to the head of the chain."""
        with self._lock, self._db:
            self._db.execute("DELETE FROM sent_sr WHERE sop_uid = ?", (str(sr.SOPInstanceUID),))
            self._db.execute(
                "INSERT INTO sent_sr(sop_uid, study_uid, model, series_uid, sop_class, digest, predecessor, sent) VALUES(?,?,?,?,?,?,?,?)",
                (str(sr.SOPInstanceUID), study_uid, model, str(sr.SeriesInstanceUID), str(sr.SOPClassUID), digest, predecessor, time.time()),
            )

    def latest_sr(self, study_uid: str, model: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM sent_sr WHERE study_uid = ? AND model = ? ORDER BY rowid DESC LIMIT 1",
                                   (study_uid, model)).fetchone()
        return dict(row) if row else None

    def sent_sr(self, sop_uid: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM sent_sr WHERE sop_uid = ?", (sop_uid,)).fetchone()
        return dict(row) if row else None
