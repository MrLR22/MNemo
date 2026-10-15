from datetime import datetime, timedelta

from modules.habits import check_missed, learn_habits, simulate_days
from modules.storage import EventStore
from modules.types import Event
from query.query_engine import QueryEngine, find_object

NOW = datetime(2026, 10, 15, 22, 30)


def store_with(*events):
    s = EventStore(":memory:")
    for e in events:
        s.add(e)
    return s


def test_object_synonyms():
    assert find_object("where are my car keys?") == "keys"
    assert find_object("did I take my medicine") == "pill_bottle"
    assert find_object("where's my mobile") == "phone"
    assert find_object("where is my umbrella") is None


def test_where_is_latest_putdown():
    s = store_with(
        Event(NOW - timedelta(hours=5), "wallet", "putdown", "desk", 0.9),
        Event(NOW - timedelta(hours=2), "wallet", "pickup", "desk", 0.9),
        Event(NOW - timedelta(hours=1), "wallet", "putdown", "entryway_table", 0.9),
    )
    ans = QueryEngine(s).ask("where is my wallet?", now=NOW)
    assert "entryway table" in ans and "9:30 PM" in ans


def test_where_after_pickup_says_still_with_you():
    s = store_with(Event(NOW - timedelta(minutes=10), "keys", "pickup", "entryway_table", 0.9))
    ans = QueryEngine(s).ask("where are my keys", now=NOW)
    assert "picked up" in ans and "haven't seen you put them down" in ans and "They may" in ans


def test_where_unknown_landmark_and_never_seen():
    s = store_with(Event(NOW - timedelta(minutes=10), "phone", "putdown", None, 0.9))
    q = QueryEngine(s)
    assert "couldn't recognise the place" in q.ask("where is my phone", now=NOW)
    assert "haven't seen" in q.ask("where is my wallet", now=NOW)


def test_future_events_are_ignored():
    s = store_with(Event(NOW + timedelta(hours=1), "wallet", "putdown", "desk", 0.9))
    assert "haven't seen" in QueryEngine(s).ask("where is my wallet", now=NOW)


def test_did_i_take_pills():
    s = store_with(Event(NOW.replace(hour=21, minute=5), "pill_bottle", "pickup", "kitchen_counter", 0.9))
    q = QueryEngine(s)
    assert q.ask("did I take my pills today?", now=NOW).startswith("Yes")
    assert q.ask("did I take my pills today?", now=NOW + timedelta(days=1)).startswith("No")


def test_habit_learned_and_missed_alert():
    s = EventStore(":memory:")
    simulate_days(s, days=10, end_day=NOW, miss_last_pill=True)
    habits, alerts = check_missed(s, NOW)
    names = {(h.object, h.action) for h in habits}
    assert ("pill_bottle", "pickup") in names and ("keys", "pickup") in names
    pill = next(h for h in habits if (h.object, h.action) == ("pill_bottle", "pickup"))
    assert abs(pill.typical_min - 21 * 60) < 20
    assert sum("pill bottle" in a for a in alerts) == 1      # one routine → one alert, not two
    # random phone put-downs are not a habit
    assert not any(h.object == "phone" for h in habits)


def test_no_alert_when_routine_done_or_window_open():
    s = EventStore(":memory:")
    simulate_days(s, days=10, end_day=NOW, miss_last_pill=False)
    _, alerts = check_missed(s, NOW)
    assert not any("pill bottle" in a for a in alerts)
    s2 = EventStore(":memory:")
    simulate_days(s2, days=10, end_day=NOW, miss_last_pill=True)
    _, alerts = check_missed(s2, NOW.replace(hour=20, minute=30))      # 9 PM window not over yet
    assert not any("pill bottle" in a for a in alerts)


def test_too_little_history_learns_nothing():
    s = EventStore(":memory:")
    simulate_days(s, days=1, end_day=NOW)
    assert learn_habits(s.all(), NOW) == []
