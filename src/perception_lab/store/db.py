"""Event history in SQLite (standard library), with full-text search when FTS5 is available.

The full event is stored as JSON; a few columns are duplicated for filtering. The store is
safe to share between the pipeline, the VLM worker and dashboard threads.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from perception_lab.events.model import REVIEW_STATES, PetEvent

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    camera_id TEXT NOT NULL,
    rule TEXT NOT NULL,
    hypothesis TEXT NOT NULL,
    activity TEXT,
    interpretation_status TEXT NOT NULL,
    review_status TEXT NOT NULL,
    zone TEXT,
    timestamp_start_s REAL NOT NULL,
    timestamp_end_s REAL NOT NULL,
    detected_at TEXT NOT NULL,
    created_unix REAL NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_created ON events (created_unix);
CREATE INDEX IF NOT EXISTS events_rule ON events (rule);
"""


class EventStore:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(_SCHEMA)
            self.has_fts = self._init_fts()
            self._db.commit()

    def _init_fts(self) -> bool:
        try:
            self._db.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(event_id UNINDEXED, body)"
            )
            return True
        except sqlite3.OperationalError:
            return False

    # -- writes ---------------------------------------------------------------------

    def add(self, event: PetEvent, created_unix: float | None = None) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    event.event_id,
                    event.camera_id,
                    event.rule,
                    event.hypothesis,
                    event.activity,
                    event.interpretation_status,
                    event.review_status,
                    event.zone,
                    event.timestamp_start_s,
                    event.timestamp_end_s,
                    event.detected_at,
                    time.time() if created_unix is None else created_unix,
                    json.dumps(event.to_dict()),
                ),
            )
            self._index(event)
            self._db.commit()

    def update(self, event: PetEvent) -> None:
        with self._lock:
            cur = self._db.execute(
                "UPDATE events SET activity=?, interpretation_status=?, review_status=?, "
                "timestamp_end_s=?, payload=? WHERE event_id=?",
                (
                    event.activity,
                    event.interpretation_status,
                    event.review_status,
                    event.timestamp_end_s,
                    json.dumps(event.to_dict()),
                    event.event_id,
                ),
            )
            if cur.rowcount == 0:
                raise KeyError(f"unknown event {event.event_id}")
            self._index(event)
            self._db.commit()

    def set_review(self, event_id: str, status: str, label: str | None = None) -> PetEvent:
        if status not in REVIEW_STATES:
            raise ValueError(f"review status must be one of {', '.join(REVIEW_STATES)}")
        event = self.get(event_id)
        if event is None:
            raise KeyError(f"unknown event {event_id}")
        event.review_status, event.review_label = status, label
        self.update(event)
        return event

    def delete_created_before(self, cutoff_unix: float) -> int:
        with self._lock:
            ids = [
                r[0]
                for r in self._db.execute(
                    "SELECT event_id FROM events WHERE created_unix < ?", (cutoff_unix,)
                )
            ]
            self._db.executemany("DELETE FROM events WHERE event_id=?", [(i,) for i in ids])
            if self.has_fts:
                self._db.executemany("DELETE FROM events_fts WHERE event_id=?", [(i,) for i in ids])
            self._db.commit()
        return len(ids)

    def _index(self, event: PetEvent) -> None:
        if not self.has_fts:
            return
        interp = event.interpretation or {}
        body = " ".join(
            str(part)
            for part in (
                event.summary(),
                event.rule,
                event.hypothesis,
                event.zone or "",
                " ".join(event.animal_classes),
                interp.get("activity", ""),
                interp.get("description", ""),
                " ".join(interp.get("visual_observations", [])),
                event.review_label or "",
            )
        )
        self._db.execute("DELETE FROM events_fts WHERE event_id=?", (event.event_id,))
        self._db.execute("INSERT INTO events_fts VALUES (?, ?)", (event.event_id, body))

    # -- reads ----------------------------------------------------------------------

    def get(self, event_id: str) -> PetEvent | None:
        with self._lock:
            row = self._db.execute(
                "SELECT payload FROM events WHERE event_id=?", (event_id,)
            ).fetchone()
        return PetEvent.from_dict(json.loads(row["payload"])) if row else None

    def list(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        rule: str | None = None,
        created_after_unix: float | None = None,
        created_before_unix: float | None = None,
    ) -> list[PetEvent]:
        sql, args = "SELECT payload FROM events WHERE 1=1", []
        if rule:
            sql += " AND rule=?"
            args.append(rule)
        if created_after_unix is not None:
            sql += " AND created_unix >= ?"
            args.append(created_after_unix)
        if created_before_unix is not None:
            sql += " AND created_unix < ?"
            args.append(created_before_unix)
        sql += " ORDER BY created_unix DESC, timestamp_start_s DESC LIMIT ? OFFSET ?"
        args += [limit, offset]
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
        return [PetEvent.from_dict(json.loads(r["payload"])) for r in rows]

    def all_events(self) -> list[PetEvent]:
        """Every event in source-time order (for evaluation)."""
        with self._lock:
            rows = self._db.execute(
                "SELECT payload FROM events ORDER BY timestamp_start_s"
            ).fetchall()
        return [PetEvent.from_dict(json.loads(r["payload"])) for r in rows]

    def search(self, query: str, limit: int = 50) -> list[PetEvent]:
        terms = query.split()
        if not terms:
            return []
        with self._lock:
            if self.has_fts:
                match = " ".join('"' + t.replace('"', '""') + '"' for t in terms)
                rows = self._db.execute(
                    "SELECT e.payload FROM events_fts f JOIN events e USING (event_id) "
                    "WHERE events_fts MATCH ? ORDER BY e.created_unix DESC LIMIT ?",
                    (match, limit),
                ).fetchall()
            else:
                sql = "SELECT payload FROM events WHERE " + " AND ".join(
                    "payload LIKE ?" for _ in terms
                )
                rows = self._db.execute(
                    sql + " ORDER BY created_unix DESC LIMIT ?",
                    [f"%{t}%" for t in terms] + [limit],
                ).fetchall()
        return [PetEvent.from_dict(json.loads(r["payload"])) for r in rows]

    def count(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT COUNT(*) FROM events").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._db.close()
