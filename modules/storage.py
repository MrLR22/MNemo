# modules/storage.py
# M6: SQLite event store = the reference diary. No video and no images are stored. Besides the text diary it can
# keep two kinds of vectors for search (search/): a CLIP image fingerprint per 4-s window (512 numbers, about 1 kB)
# and a meaning vector per distinct diary line. Fingerprints are derived from the camera images, so they are
# private data like the diary itself: never sent anywhere, deleted with the diary.

import os
import sqlite3
from datetime import datetime

import numpy as np

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
    seen        TEXT,                   -- objects in view, comma-separated (prompt lines-v4)
    raw         TEXT,                   -- the model's full answer, for checking
    source      TEXT
);
CREATE INDEX IF NOT EXISTS idx_activities_ts ON activities(ts_start);

-- search vectors (search/engine.py): float16, unit length
CREATE TABLE IF NOT EXISTS fingerprints (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_start    TEXT NOT NULL,
    ts_end      TEXT NOT NULL,
    model       TEXT NOT NULL,          -- e.g. "ViT-B-32/openai": vectors of different models never mix
    vec         BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fingerprints_ts ON fingerprints(ts_start);
CREATE TABLE IF NOT EXISTS text_vectors (
    text        TEXT NOT NULL,          -- the diary line without its time (search.rank.diary_text)
    model       TEXT NOT NULL,
    vec         BLOB NOT NULL,
    PRIMARY KEY (text, model)
);
"""


def _blob(v) -> bytes:
    return np.asarray(v, np.float16).tobytes()


def _vec(b: bytes) -> np.ndarray:
    return np.frombuffer(b, np.float16).astype(np.float32)


class EventStore:
    def __init__(self, path: str = C.DB_PATH):
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(activities)")}
        if "seen" not in cols:                       # diaries written before the SEEN line
            self.conn.execute("ALTER TABLE activities ADD COLUMN seen TEXT")

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
            "INSERT INTO activities (ts_start, ts_end, verb, object, place, where_, seen, raw, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (row["ts_start"].isoformat(timespec="seconds"), row["ts_end"].isoformat(timespec="seconds"),
             row.get("verb"), row.get("object"), row.get("place"), row.get("where"), row.get("seen"),
             row.get("raw"), source))
        self.conn.commit()
        return cur.lastrowid

    def extend_activity(self, activity_id: int, ts_end: datetime, seen: str | None = None):
        self.conn.execute("UPDATE activities SET ts_end = ?, seen = COALESCE(?, seen) WHERE id = ?",
                          (ts_end.isoformat(timespec="seconds"), seen, activity_id))
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
        """Delete the diary and its search vectors."""
        for table in ("activities", "fingerprints", "text_vectors"):
            self.conn.execute(f"DELETE FROM {table}")
        self.conn.commit()

    # ── search vectors ──

    def add_fingerprint(self, ts_start: datetime, ts_end: datetime, vec, model: str):
        self.conn.execute("INSERT INTO fingerprints (ts_start, ts_end, model, vec) VALUES (?, ?, ?, ?)",
                          (ts_start.isoformat(timespec="milliseconds"), ts_end.isoformat(timespec="milliseconds"),
                           model, _blob(vec)))
        self.conn.commit()

    def fingerprints(self, model: str, start: datetime | None = None, end: datetime | None = None
                     ) -> tuple[list[datetime], list[datetime], np.ndarray]:
        """(window starts, window ends, [n, d] vectors) of one model, in time order."""
        q, args = "SELECT ts_start, ts_end, vec FROM fingerprints WHERE model = ?", [model]
        if start:
            q += " AND ts_end > ?"                     # a window ending exactly at start is outside the range
            args.append(start.isoformat(timespec="milliseconds"))
        if end:
            q += " AND ts_start < ?"
            args.append(end.isoformat(timespec="milliseconds"))
        rows = self.conn.execute(q + " ORDER BY ts_start", args).fetchall()
        if not rows:
            return [], [], np.zeros((0, 0), np.float32)
        return ([datetime.fromisoformat(r[0]) for r in rows], [datetime.fromisoformat(r[1]) for r in rows],
                np.stack([_vec(r[2]) for r in rows]))

    def text_vectors(self, texts: list[str], model: str) -> dict[str, np.ndarray]:
        out = {}
        for i in range(0, len(texts), 500):                     # SQLite limits the number of ? per query
            part = texts[i:i + 500]
            q = f"SELECT text, vec FROM text_vectors WHERE model = ? AND text IN ({','.join('?' * len(part))})"
            out.update({t: _vec(v) for t, v in self.conn.execute(q, [model, *part])})
        return out

    def add_text_vectors(self, vecs: dict[str, np.ndarray], model: str):
        self.conn.executemany("INSERT OR REPLACE INTO text_vectors (text, model, vec) VALUES (?, ?, ?)",
                              [(t, model, _blob(v)) for t, v in vecs.items()])
        self.conn.commit()

    def objects(self) -> list[str]:
        return [r[0] for r in self.conn.execute("SELECT DISTINCT object FROM events ORDER BY object")]

    def clear(self):
        self.conn.execute("DELETE FROM events")
        self.conn.commit()

    def close(self):
        self.conn.close()
