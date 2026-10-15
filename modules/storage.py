# modules/storage.py
# M6: SQLite event store. Text only: no images, no embeddings of the user's day.

import os
import sqlite3
from datetime import datetime

import config as C
from modules.types import Event

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,          -- ISO-8601 local time
    object      TEXT NOT NULL,
    action      TEXT NOT NULL CHECK (action IN ('pickup', 'putdown')),
    landmark    TEXT,                   -- NULL = unknown
    confidence  REAL NOT NULL,
    note        TEXT,
    source      TEXT                    -- which video/session produced it
);
CREATE INDEX IF NOT EXISTS idx_events_obj_ts ON events(object, ts);
"""


class EventStore:
    def __init__(self, path: str = C.DB_PATH):
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)

    def add(self, ev: Event, source: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO events (ts, object, action, landmark, confidence, note, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ev.ts.isoformat(timespec="seconds"), ev.object, ev.action, ev.landmark,
             float(ev.confidence), ev.note, source))
        self.conn.commit()
        return cur.lastrowid

    @staticmethod
    def _to_event(r: sqlite3.Row) -> Event:
        return Event(ts=datetime.fromisoformat(r["ts"]), object=r["object"], action=r["action"],
                     landmark=r["landmark"], confidence=r["confidence"], note=r["note"])

    def latest(self, obj: str, action: str | None = None, before: datetime | None = None) -> Event | None:
        q, args = "SELECT * FROM events WHERE object = ?", [obj]
        if action:
            q += " AND action = ?"
            args.append(action)
        if before:
            q += " AND ts <= ?"
            args.append(before.isoformat(timespec="seconds"))
        # "latest event wins": order by time, then insertion order for ties
        r = self.conn.execute(q + " ORDER BY ts DESC, id DESC LIMIT 1", args).fetchone()
        return self._to_event(r) if r else None

    def between(self, start: datetime, end: datetime, obj: str | None = None) -> list[Event]:
        q = "SELECT * FROM events WHERE ts >= ? AND ts < ?"
        args = [start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds")]
        if obj:
            q += " AND object = ?"
            args.append(obj)
        return [self._to_event(r) for r in self.conn.execute(q + " ORDER BY ts, id", args)]

    def all(self) -> list[Event]:
        return [self._to_event(r) for r in self.conn.execute("SELECT * FROM events ORDER BY ts, id")]

    def clear(self):
        self.conn.execute("DELETE FROM events")
        self.conn.commit()

    def close(self):
        self.conn.close()
