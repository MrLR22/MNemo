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

-- stage 1 (vision-language logger): any action, any object
CREATE TABLE IF NOT EXISTS activities (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_start    TEXT NOT NULL,
    ts_end      TEXT NOT NULL,
    verb        TEXT,                   -- NULL = nothing done with the hands (place only)
    object      TEXT,
    place       TEXT,
    where_      TEXT,                   -- where the object is / ended up
    raw         TEXT,                   -- the model's full answer, for checking
    source      TEXT
);
CREATE INDEX IF NOT EXISTS idx_activities_ts ON activities(ts_start);
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

    # ── activities (stage 1 log) ──
    def add_activity(self, row: dict, source: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO activities (ts_start, ts_end, verb, object, place, where_, raw, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (row["ts_start"].isoformat(timespec="seconds"), row["ts_end"].isoformat(timespec="seconds"),
             row.get("verb"), row.get("object"), row.get("place"), row.get("where"), row.get("raw"), source))
        self.conn.commit()
        return cur.lastrowid

    def extend_activity(self, activity_id: int, ts_end: datetime):
        self.conn.execute("UPDATE activities SET ts_end = ? WHERE id = ?",
                          (ts_end.isoformat(timespec="seconds"), activity_id))
        self.conn.commit()

    def activities(self, start: datetime | None = None, end: datetime | None = None) -> list[dict]:
        q, args = "SELECT * FROM activities WHERE 1=1", []
        if start:
            q += " AND ts_end >= ?"
            args.append(start.isoformat(timespec="seconds"))
        if end:
            q += " AND ts_start < ?"
            args.append(end.isoformat(timespec="seconds"))
        rows = []
        for r in self.conn.execute(q + " ORDER BY ts_start, id", args):
            d = dict(r)
            d["ts_start"], d["ts_end"] = datetime.fromisoformat(d["ts_start"]), datetime.fromisoformat(d["ts_end"])
            d["where"] = d.pop("where_")
            rows.append(d)
        return rows

    def clear_activities(self):
        self.conn.execute("DELETE FROM activities")
        self.conn.commit()

    def objects(self) -> list[str]:
        return [r[0] for r in self.conn.execute("SELECT DISTINCT object FROM events ORDER BY object")]

    def clear(self):
        self.conn.execute("DELETE FROM events")
        self.conn.commit()

    def close(self):
        self.conn.close()
