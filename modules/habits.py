# modules/habits.py
# M8: learn daily routines from event history and flag missed ones.
#
# For each (object, action), take the time-of-day of every event over the last
# HABIT_LOOKBACK_DAYS. Times closer than HABIT_CLUSTER_GAP_MIN are grouped, so
# "pills at 8 am" and "pills at 9 pm" become two separate habits. A group is a habit
# if it happened on at least HABIT_MIN_DAY_FRACTION of the observed days.
# Window = median time +- max(3 x robust SD, HABIT_MIN_WINDOW_MIN), robust SD = 1.4826 x MAD.
# Groups with MAD > HABIT_MAX_SPREAD_MIN are scattered, not routines.
# Alert: the window has passed today and no matching event happened from (window start -
# window width) until now, so doing it late or a little early does not raise an alert.
#
# Limits (state them in the report): a pill-bottle pick-up is a proxy for "took medicine";
# needs >= HABIT_MIN_DAYS of history; the demo uses clearly-labelled simulated days.

import random
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

import config as C
from modules.storage import EventStore
from modules.types import Event


@dataclass
class Habit:
    object: str
    action: str
    typical_min: float          # minutes after midnight
    window_min: float           # half-width of the window
    days_seen: int
    days_observed: int

    def window(self, day: datetime) -> tuple[datetime, datetime]:
        mid = day.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(minutes=self.typical_min)
        return mid - timedelta(minutes=self.window_min), mid + timedelta(minutes=self.window_min)

    def describe(self) -> str:
        h, m = divmod(int(round(self.typical_min)), 60)
        t = datetime(2000, 1, 1, h % 24, m).strftime("%I:%M %p").lstrip("0")
        verb = "pick up" if self.action == "pickup" else "put down"
        return (f"{verb} {self.object.replace('_', ' ')} around {t} "
                f"(+-{int(self.window_min)} min, {self.days_seen}/{self.days_observed} days)")


def _clock(ts: datetime) -> str:
    return ts.strftime("%I:%M %p").lstrip("0")


def _minutes(ts: datetime) -> float:
    return ts.hour * 60 + ts.minute + ts.second / 60


def learn_habits(events: list[Event], today: datetime) -> list[Habit]:
    """Learn from events strictly before `today` (so today's behaviour isn't used to judge itself)."""
    day0 = today.replace(hour=0, minute=0, second=0, microsecond=0)
    start = day0 - timedelta(days=C.HABIT_LOOKBACK_DAYS)
    past = [e for e in events if start <= e.ts < day0]
    observed_days = {e.ts.date() for e in past}
    n_days = len(observed_days)
    if n_days < C.HABIT_MIN_DAYS:
        return []

    habits = []
    for key in sorted({(e.object, e.action) for e in past}):
        evs = sorted((e for e in past if (e.object, e.action) == key), key=lambda e: _minutes(e.ts))
        groups, cur = [], [evs[0]]
        for e in evs[1:]:
            if _minutes(e.ts) - _minutes(cur[-1].ts) > C.HABIT_CLUSTER_GAP_MIN:
                groups.append(cur)
                cur = [e]
            else:
                cur.append(e)
        groups.append(cur)
        for g in groups:
            # one time per day (the first) so a day with 3 pick-ups doesn't dominate
            first_per_day = {}
            for e in g:
                d = e.ts.date()
                if d not in first_per_day or e.ts < first_per_day[d]:
                    first_per_day[d] = e.ts
            days_seen = len(first_per_day)
            if days_seen / n_days < C.HABIT_MIN_DAY_FRACTION:
                continue
            mins = np.array([_minutes(t) for t in first_per_day.values()])
            med = float(np.median(mins))
            mad = float(np.median(np.abs(mins - med)))
            if mad > C.HABIT_MAX_SPREAD_MIN:
                continue        # happens most days but at scattered times: not a routine
            robust_sd = 1.4826 * mad            # MAD → standard deviation for normal data
            habits.append(Habit(key[0], key[1], med, max(3 * robust_sd, C.HABIT_MIN_WINDOW_MIN),
                                days_seen, n_days))
    return habits


def check_missed(store: EventStore, now: datetime) -> tuple[list[Habit], list[str]]:
    """Return (habits, alerts) for `now`."""
    habits = learn_habits(store.all(), now)
    alerts, alerted = [], []                    # alerted: (object, start, end) already reported
    # pick-ups first: "pick up pills" and "put down pills" at ~9 PM are one routine, one alert
    for h in sorted(habits, key=lambda h: h.action != "pickup"):
        start, end = h.window(now)
        if now < end:
            continue                            # window still open
        if any(o == h.object and s < end and start < e for o, s, e in alerted):
            continue
        # Done late (after the window, before now) or a little early still counts.
        early = start - timedelta(minutes=h.window_min)
        done = [e for e in store.between(early, now + timedelta(seconds=1), h.object)
                if e.action == h.action]
        if not done:
            alerted.append((h.object, start, end))
            alerts.append(f"MISSED: you usually {h.describe().split(' (')[0]}, "
                          f"but nothing was recorded between {_clock(start)} and {_clock(end)} today.")
    return habits, alerts


def simulate_days(store: EventStore, days: int, end_day: datetime, miss_last_pill: bool = True,
                  seed: int = 0) -> None:
    """
    SIMULATED history for demonstrating/testing the habit engine (label it as simulated in the
    report). Routine: keys picked up ~8:30 and put down ~18:30 at entryway_table; pills ~9:00 PM
    at kitchen_counter; phone put down at random times. On the final day the 9 PM pills are
    skipped (if miss_last_pill) so an alert should fire.
    """
    rng = random.Random(seed)
    day0 = end_day.replace(hour=0, minute=0, second=0, microsecond=0)

    def at(day, minutes, jitter):
        return day + timedelta(minutes=minutes + rng.gauss(0, jitter))

    for k in range(days, -1, -1):
        day = day0 - timedelta(days=k)
        evs = [
            Event(at(day, 8 * 60 + 30, 10), "keys", "pickup", "entryway_table", 0.9),
            Event(at(day, 18 * 60 + 30, 20), "keys", "putdown", "entryway_table", 0.9),
            Event(at(day, rng.uniform(9 * 60, 22 * 60), 0), "phone", "putdown",
                  rng.choice(["desk", "bedside_table", "sofa"]), 0.8),
        ]
        if not (k == 0 and miss_last_pill):
            pick = at(day, 21 * 60, 12)
            evs.append(Event(pick, "pill_bottle", "pickup", "kitchen_counter", 0.9))
            evs.append(Event(pick + timedelta(minutes=rng.uniform(1, 3)), "pill_bottle", "putdown",
                             "kitchen_counter", 0.9))
        for e in evs:
            e.note = "SIMULATED"
            store.add(e, source="simulated")
