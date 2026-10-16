# query/llm_qa.py
# STAGE 2: action log → answer.
#
# 1. Pick the time range from the question ("today", "yesterday", "this morning", ...).
# 2. Find the log rows that match the question's words (with simple synonyms: shut = close,
#    fridge = refrigerator, kept = put, tablets = medicine, ...).
# 3. A language model answers using ONLY those rows, with times, and says "I didn't see that"
#    when the log has no answer. Without a language model (e.g. on a laptop CPU) the matching
#    rows are listed instead, which never invents anything.

import re
from datetime import datetime, timedelta

import config as C

STOP = set("""a an the i me my mine we our you your he she it its they them is are was were be been am do does did
done have has had to of in on at by for from with and or but not no yes what when where which who whom how
why did any some this that these those there here please tell show can could would should will shall
today yesterday morning evening night tonight last hour hours ago now time""".split())

CANON = {
    "shut": "close", "closed": "close", "closing": "close",
    "opened": "open", "opening": "open",
    "fridge": "refrigerator", "refrigerators": "refrigerator",
    "kept": "put", "keep": "put", "place": "put", "placed": "put", "leave": "put", "left": "put",
    "putting": "put", "set": "put", "drop": "put", "dropped": "put",
    "took": "take", "taken": "take", "taking": "take", "grab": "take", "grabbed": "take",
    "pick": "take", "picked": "take", "picks": "take",
    "switch": "turn", "switched": "turn", "turned": "turn",
    "tablet": "medicine", "tablets": "medicine", "pill": "medicine", "pills": "medicine",
    "meds": "medicine", "medication": "medicine",
    "mobile": "phone", "cellphone": "phone", "smartphone": "phone",
    "television": "tv", "telly": "tv",
    "ate": "eat", "eaten": "eat", "eating": "eat", "drank": "drink", "drunk": "drink", "drinking": "drink",
    "washed": "wash", "washing": "wash", "cleaned": "clean", "cleaning": "clean",
    "key": "keys", "door": "door", "doors": "door", "light": "light", "lights": "light",
}


def norm_tokens(text: str | None) -> set[str]:
    out = set()
    for w in re.findall(r"[a-z]+", (text or "").lower()):
        if w in STOP or len(w) < 2:
            continue
        w = CANON.get(w, w)
        if w not in CANON.values():                       # light stemming for words we don't map
            for suf in ("ing", "ed", "es", "s"):
                if w.endswith(suf) and len(w) - len(suf) >= 3:
                    w = w[: -len(suf)]
                    break
        out.add(CANON.get(w, w))
    return out


def time_range(question: str, now: datetime) -> tuple[datetime, datetime, str]:
    q = question.lower()
    day0 = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if "yesterday" in q:
        return day0 - timedelta(days=1), day0, "yesterday"
    if "this morning" in q:
        return day0, min(now, day0 + timedelta(hours=12)), "this morning"
    if re.search(r"\b(tonight|this evening)\b", q):
        return day0 + timedelta(hours=17), now, "this evening"
    m = re.search(r"\blast (\d+) (minute|hour)s?\b", q)
    if m:
        n = int(m.group(1))
        delta = timedelta(minutes=n) if m.group(2) == "minute" else timedelta(hours=n)
        return now - delta, now, f"in the last {n} {m.group(2)}s"
    if "last hour" in q:
        return now - timedelta(hours=1), now, "in the last hour"
    return day0, now, "today"


def fmt_time(ts: datetime) -> str:
    return ts.strftime("%I:%M %p").lstrip("0")


def fmt_row(r: dict) -> str:
    what = " ".join(x for x in (r["verb"], r["object"]) if x) or "(no hand action)"
    extra = []
    if r.get("where"):
        extra.append(f"object at: {r['where']}")
    if r.get("place"):
        extra.append(f"place: {r['place']}")
    return f"{r['ts_start']:%H:%M:%S}-{r['ts_end']:%H:%M:%S} | {what}" + (f" | {'; '.join(extra)}" if extra else "")


def retrieve(rows: list[dict], question: str, k: int = C.QA_MAX_LOG_LINES) -> list[dict]:
    """Rows sharing words with the question, best first (ties: most recent first)."""
    q = norm_tokens(question)
    scored = []
    for r in rows:
        toks = norm_tokens(" ".join(str(r.get(f) or "") for f in ("verb", "object", "where", "place")))
        s = len(q & toks)
        if s:
            scored.append((s, r["ts_start"], r))
    scored.sort(key=lambda x: (-x[0], -x[1].timestamp()))
    return [r for _, _, r in scored[:k]]


PROMPT = """You are a memory assistant for a person with memory problems. Below is a log of what the \
person did, written automatically from a camera they wear. The log may contain mistakes and may miss things.

LOG ({period}):
{log}

QUESTION: {question}

Answer in 1-3 short sentences using ONLY the log. Say the time (and the place, if the log has it). \
If the log does not answer the question, say exactly: "I didn't see that in my records." Do not guess."""


def answer(question: str, store, llm=None, now: datetime | None = None) -> dict:
    """llm: object with .chat(prompt) (VLMDescriber) or None for the no-LLM fallback."""
    now = now or datetime.now()
    start, end, period = time_range(question, now)
    rows = store.activities(start, end)
    if not rows and period == "today":                    # nothing today: look back a week
        start, period = now - timedelta(days=7), "the last 7 days"
        rows = store.activities(start, end)
    hits = retrieve(rows, question)
    if not rows:
        return {"answer": f"I have no records for {period}.", "lines": [], "mode": "none"}

    if llm is None:
        if not hits:
            return {"answer": f"I didn't see that in my records ({len(rows)} entries {period}).",
                    "lines": [], "mode": "search"}
        top = sorted(hits[:5], key=lambda r: r["ts_start"])
        text = "Matching records:\n" + "\n".join(
            f"- {fmt_time(r['ts_start'])}: {' '.join(x for x in (r['verb'], r['object']) if x)}"
            + (f" ({r['where'] or r['place']})" if (r.get('where') or r.get('place')) else "") for r in top)
        return {"answer": text, "lines": top, "mode": "search"}

    # with a language model: matching rows if any, otherwise the most recent part of the period
    context = hits if hits else rows[-C.QA_MAX_LOG_LINES:]
    context = sorted(context, key=lambda r: r["ts_start"])
    prompt = PROMPT.format(period=period, log="\n".join(fmt_row(r) for r in context), question=question)
    return {"answer": llm.chat(prompt).strip(), "lines": context, "mode": "llm"}
