# query/query_engine.py
# M7: rule-based questions over the event store. Answers only from stored events.
#
# Supported:
#   "where is my wallet?" / "where are my keys" / "where did I leave my phone"
#   "did I take my pills today?" / "did I take my medicine"
#   "when did I last pick up my keys?"
#   "what happened today?" / "history of my wallet"

import re
from datetime import datetime, timedelta

import config as C
from modules.storage import EventStore
from modules.types import Event

PRETTY = {"keys": "keys", "wallet": "wallet", "phone": "phone", "pill_bottle": "pill bottle"}


def find_object(text: str) -> str | None:
    t = text.lower()
    # longest synonym first, so "pill bottle" beats "pill"
    pairs = sorted(((syn, obj) for obj, syns in C.OBJECT_SYNONYMS.items() for syn in syns),
                   key=lambda p: len(p[0]), reverse=True)
    for syn, obj in pairs:
        if re.search(rf"\b{re.escape(syn)}\b", t):
            return obj
    return None


def when(ts: datetime, now: datetime) -> str:
    clock = ts.strftime("%I:%M %p").lstrip("0")
    days = (now.date() - ts.date()).days
    if days == 0:
        return f"today at {clock}"
    if days == 1:
        return f"yesterday at {clock}"
    return f"on {ts.strftime('%a %d %b')} at {clock}"


def place(landmark: str | None) -> str:
    return f"the {landmark.replace('_', ' ')}" if landmark else "an unknown place"


def describe(ev: Event, now: datetime) -> str:
    name = PRETTY.get(ev.object, ev.object)
    if ev.action == "putdown":
        s = f"you put your {name} down at {place(ev.landmark)} {when(ev.ts, now)}"
    else:
        s = f"you picked up your {name} from {place(ev.landmark)} {when(ev.ts, now)}"
    if ev.note:
        s += f" ({ev.note})"
    return s


class QueryEngine:
    def __init__(self, store: EventStore):
        self.store = store

    def ask(self, question: str, now: datetime | None = None) -> str:
        now = now or datetime.now()
        q = question.lower().strip()
        obj = find_object(q)

        if re.search(r"\b(what happened|what did i do|today'?s events|summary)\b", q):
            return self._day_summary(now)
        if obj is None:
            return ("I can answer about: " + ", ".join(PRETTY.values()) +
                    ". Try: 'where is my wallet?' or 'did I take my pills today?'")
        if re.search(r"\bhistory\b|\ball\b.*\btimes\b", q):
            return self._history(obj, now)
        if re.search(r"\bdid i (take|have)\b", q):
            return self._did_take(obj, now)
        if re.search(r"\bwhen did i\b.*\bpick", q):
            ev = self.store.latest(obj, "pickup", before=now)
            return (describe(ev, now).capitalize() + ".") if ev else \
                f"I have no record of you picking up your {PRETTY[obj]}."
        return self._where(obj, now)

    def _where(self, obj: str, now: datetime) -> str:
        name = PRETTY[obj]
        ev = self.store.latest(obj, before=now)
        if ev is None:
            return f"I haven't seen your {name} yet."
        it = "them" if obj == "keys" else "it"
        if ev.action == "putdown":
            if ev.landmark:
                ans = f"Your {name} should be at {place(ev.landmark)}. You put {it} there {when(ev.ts, now)}."
            else:
                ans = (f"You put your {name} down {when(ev.ts, now)}, but I couldn't recognise the "
                       f"place.")
            if ev.confidence < 0.5:
                ans += " (low confidence)"
            return ans
        # latest is a pick-up: we never saw where it went afterwards
        return (f"You picked up your {name} {when(ev.ts, now)}"
                + (f" from {place(ev.landmark)}" if ev.landmark else "")
                + f" and I haven't seen you put {it} down since. "
                + ("They" if it == "them" else "It") + " may still be with you (pocket or bag).")

    def _did_take(self, obj: str, now: datetime) -> str:
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        evs = [e for e in self.store.between(start, now + timedelta(seconds=1), obj) if e.action == "pickup"]
        name = PRETTY[obj]
        if not evs:
            return f"No — I have no record of you picking up your {name} today."
        times = ", ".join(e.ts.strftime("%I:%M %p").lstrip("0") for e in evs)
        return f"Yes — you picked up your {name} today at {times}."

    def _history(self, obj: str, now: datetime) -> str:
        evs = self.store.between(now - timedelta(days=2), now + timedelta(seconds=1), obj)
        if not evs:
            return f"No events for your {PRETTY[obj]} in the last 2 days."
        return "\n".join("- " + describe(e, now) for e in evs[-15:])

    def _day_summary(self, now: datetime) -> str:
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        evs = self.store.between(start, now + timedelta(seconds=1))
        if not evs:
            return "Nothing recorded today."
        return "Today:\n" + "\n".join(
            f"- {e.ts.strftime('%I:%M %p').lstrip('0')}: " + describe(e, now).split(f' {when(e.ts, now)}')[0]
            for e in evs)
