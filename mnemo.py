# mnemo.py
# Mnemo: in-home personal memory assistant (command-line entry point).
#
#   python mnemo.py run VIDEO [--start "2026-10-15 08:30"] [--mode rules|model] [--show]
#   python mnemo.py run 0 --show                     (live camera index 0)
#   python mnemo.py enroll [--check]                 build landmark index from data/enroll/
#   python mnemo.py ask "where is my wallet?"
#   python mnemo.py events                           list stored events
#   python mnemo.py habits [--now "2026-10-15 22:00"] [--simulated]
#   python mnemo.py simulate --days 14               write SIMULATED history to a separate DB
#
# Stage 1 / stage 2 (vision-language model; needs a GPU such as the Colab T4):
#   python mnemo.py log VIDEO [--start ...]          video → action log (any action, any object)
#   python mnemo.py qa "did I close the fridge?" [--llm]   answer from the action log
#   python mnemo.py activities                       list the action log

import argparse
import sys
from datetime import datetime

import config as C


def _dt(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


def _objects(a):
    if getattr(a, "objects", None):
        C.set_objects(a.objects)


def cmd_run(a):
    _objects(a)
    from modules.pipeline import MnemoPipeline
    from modules.storage import EventStore
    source = int(a.source) if a.source.isdigit() else a.source
    store = EventStore(a.db)
    pipe = MnemoPipeline(mode=a.mode, store=store, use_landmarks=not a.no_landmarks)
    print(f"Processing {source} in {a.mode} mode at {C.PROCESS_FPS} fps ... (q in the window to stop)")
    events, stats = pipe.run(source, start_time=_dt(a.start), show=a.show)
    print(f"\n{stats['events']} events stored in {a.db}")
    print(f"{stats['frames']} frames in {stats['seconds']}s → {stats['fps']} fps, peak RAM {stats['peak_ram_mb']} MB")


def cmd_enroll(a):
    from modules.landmarks import ClipEmbedder, LandmarkIndex, build_index, check_index
    if a.check_only:
        idx = LandmarkIndex.load()
    else:
        idx = build_index(ClipEmbedder(), a.dir)
        idx.save()
        print(f"Saved landmark index: {len(idx.labels)} frames, landmarks {idx.names} → {C.LANDMARK_INDEX}")
    if a.check or a.check_only:
        r = check_index(idx)
        print("\nLeave-one-source-out check (each enrollment clip vs. all others):")
        for k, v in r.items():
            print(f"  {k}: {v}")
        if any(n < 2 for n in r["sources_per_landmark"].values()):
            print("  NOTE: landmarks with only 1 clip/photo can't be checked. Record 2-3 clips per landmark "
                  "from different angles/lighting.")


def cmd_ask(a):
    from modules.storage import EventStore
    from query.query_engine import QueryEngine
    db = C.SIM_DB_PATH if a.simulated else a.db
    print(QueryEngine(EventStore(db)).ask(" ".join(a.question), now=_dt(a.now)))


def cmd_events(a):
    from modules.storage import EventStore
    for e in EventStore(C.SIM_DB_PATH if a.simulated else a.db).all()[-a.limit:]:
        print(f"{e.ts:%Y-%m-%d %H:%M:%S}  {e.action:7s}  {e.object:12s}  {e.landmark or 'unknown':18s} "
              f"conf={e.confidence:.2f}" + (f"  {e.note}" if e.note else ""))


def cmd_habits(a):
    from modules.habits import check_missed
    from modules.storage import EventStore
    now = _dt(a.now) or datetime.now()
    db = C.SIM_DB_PATH if a.simulated else a.db
    habits, alerts = check_missed(EventStore(db), now)
    if a.simulated:
        print("*** SIMULATED DATA ***")
    print(f"Habits learned (as of {now:%Y-%m-%d %H:%M}):")
    for h in habits:
        print("  - " + h.describe())
    if not habits:
        print(f"  none yet (need >= {C.HABIT_MIN_DAYS} days of history)")
    print("Alerts:" if alerts else "No missed routines.")
    for al in alerts:
        print("  ! " + al)


def cmd_simulate(a):
    from modules.habits import simulate_days
    from modules.storage import EventStore
    store = EventStore(C.SIM_DB_PATH)
    store.clear()
    end = _dt(a.end) or datetime.now()
    simulate_days(store, a.days, end, miss_last_pill=not a.no_miss)
    print(f"Wrote {len(store.all())} SIMULATED events to {C.SIM_DB_PATH}")
    print(f'Try: python mnemo.py habits --simulated --now "{end:%Y-%m-%d} 22:30"')


def cmd_log(a):
    from modules.storage import EventStore
    from modules.vlm_logger import ActionLogger, VLMDescriber
    source = int(a.source) if a.source.isdigit() else a.source
    call = scores = None
    if a.trigger:
        if isinstance(source, int):
            raise SystemExit("--trigger needs a video file (live camera: not yet)")
        from modules.hoi_gate import HOIGate, gate_decisions
        scores, g = HOIGate(a.checkpoint).scores(source, a.max_seconds)
        call = gate_decisions(scores, a.threshold)
        print(f"trigger: VLM on {call.sum()}/{len(call)} windows ({g['gate_seconds']}s for hands/objects/HOI-TCN)")
    print(f"Loading {a.model} ...")
    logger = ActionLogger(VLMDescriber(a.model), EventStore(a.db))
    rows, stats = logger.run(source, start_time=_dt(a.start), max_seconds=a.max_seconds, call=call, scores=scores)
    print(f"\n{stats['rows']} log rows stored in {a.db}")
    print(f"{stats['video_seconds']}s of video, VLM on {stats['vlm_calls']}/{stats['windows']} windows, "
          f"model time {stats['model_seconds']}s → real-time factor {stats['real_time_factor']} "
          f"(below 1.0 = faster than real time)")


def cmd_qa(a):
    from modules.storage import EventStore
    from query.llm_qa import answer
    llm = None
    if a.llm:
        from modules.vlm_logger import VLMDescriber
        llm = VLMDescriber(a.model)
    r = answer(" ".join(a.question), EventStore(a.db), llm=llm, now=_dt(a.now))
    print(r["answer"])


def cmd_activities(a):
    from modules.storage import EventStore
    from query.llm_qa import fmt_row
    for r in EventStore(a.db).activities()[-a.limit:]:
        print(f"{r['ts_start']:%Y-%m-%d} " + fmt_row(r))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Mnemo personal memory assistant")
    ap.add_argument("--db", default=C.DB_PATH)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="process a video file or camera")
    r.add_argument("source")
    r.add_argument("--start", help='real start time of the recording, e.g. "2026-10-15 08:30"')
    r.add_argument("--mode", choices=["rules", "model"], default="rules")
    r.add_argument("--show", action="store_true", help="display the video with overlays (nothing is saved)")
    r.add_argument("--no-landmarks", action="store_true")
    r.add_argument("--objects", nargs="+", help="objects to track instead of keys/wallet/phone/pill_bottle")
    r.set_defaults(fn=cmd_run)

    e = sub.add_parser("enroll", help="build the landmark index from data/enroll/")
    e.add_argument("--dir", default=C.ENROLL_DIR)
    e.add_argument("--check", action="store_true", help="also run the leave-one-out accuracy check")
    e.add_argument("--check-only", action="store_true", help="check the existing index without rebuilding")
    e.set_defaults(fn=cmd_enroll)

    q = sub.add_parser("ask", help='ask e.g. "where is my wallet?"')
    q.add_argument("question", nargs="+")
    q.add_argument("--now")
    q.add_argument("--simulated", action="store_true")
    q.set_defaults(fn=cmd_ask)

    ev = sub.add_parser("events", help="list stored events")
    ev.add_argument("--limit", type=int, default=50)
    ev.add_argument("--simulated", action="store_true")
    ev.set_defaults(fn=cmd_events)

    h = sub.add_parser("habits", help="show learned routines and missed-routine alerts")
    h.add_argument("--now")
    h.add_argument("--simulated", action="store_true")
    h.set_defaults(fn=cmd_habits)

    s = sub.add_parser("simulate", help="generate SIMULATED multi-day history for the habit demo")
    s.add_argument("--days", type=int, default=14)
    s.add_argument("--end", help="last simulated day (default today)")
    s.add_argument("--no-miss", action="store_true", help="don't skip the last evening's pills")
    s.set_defaults(fn=cmd_simulate)

    lg = sub.add_parser("log", help="STAGE 1: video → action log with a vision-language model (GPU)")
    lg.add_argument("source")
    lg.add_argument("--start", help='real start time of the recording, e.g. "2026-10-17 08:30"')
    lg.add_argument("--model", default=C.VLM_MODEL)
    lg.add_argument("--max-seconds", type=float, default=None, help="only the first N seconds")
    lg.add_argument("--trigger", action="store_true", help="only describe windows where HOI-TCN sees hand interaction")
    lg.add_argument("--threshold", type=float, default=C.GATE_THRESHOLD)
    lg.add_argument("--checkpoint", default=C.MODEL_CHECKPOINT)
    lg.set_defaults(fn=cmd_log)

    qa = sub.add_parser("qa", help='STAGE 2: answer any question from the action log')
    qa.add_argument("question", nargs="+")
    qa.add_argument("--llm", action="store_true", help="answer with the language model (GPU); "
                                                       "without it, matching log rows are listed")
    qa.add_argument("--model", default=C.VLM_MODEL)
    qa.add_argument("--now")
    qa.set_defaults(fn=cmd_qa)

    ac = sub.add_parser("activities", help="list the action log")
    ac.add_argument("--limit", type=int, default=100)
    ac.set_defaults(fn=cmd_activities)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
