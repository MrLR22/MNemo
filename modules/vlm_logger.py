# modules/vlm_logger.py
# STAGE 1: video → action log, using a vision-language model (VLM).
#
# Every VLM_WINDOW_S seconds, VLM_FRAMES frames go to the VLM with one question: "what does the
# camera wearer do with their hands?". The answer is parsed into rows
#   time | verb | object | place | where-it-ended-up
# and stored as TEXT in the activities table. Frames stay in RAM only (privacy, as before).
# Unlike the pick/put pipeline this is not limited to a fixed list of actions or objects.
#
# Two additions on top of plain captioning:
#   - OBJECT STATES: the VLM also reports whether the fridge/door/drawer/... is open or closed and
#     whether the tap/stove/light is on or off. A change between two described windows becomes a log
#     row ("closes fridge"), even when the VLM never wrote the action itself. Open/close and on/off
#     are exactly what a person with memory problems worries about, and what plain captions miss.
#   - INTERACTION TRIGGER: an optional per-window call/skip mask (modules/hoi_gate.py) so the VLM only
#     runs where the hands interact with something.
#   - STATE CHECK: the VLM reports "open"/"on" readily but almost never says "closed" (1 of 120 windows on
#     EPIC). So while the memory believes something is open/on, the next prompts explicitly ask for its
#     state; "closes fridge" is then seen instead of guessed, and alerts are verified, not stale.
#   - LOOP STOP: the VLM sometimes repeats one line over and over (27% of rows, 32% of VLM time on EPIC);
#     generation stops at the first repeated line or after VLM_MAX_ACTIONS actions.
# The raw VLM answer of every window is kept in memory (ActionLogger.windows), so the log can be
# rebuilt with other settings (no states, other gate threshold) without calling the VLM again.

import json
import re
import time
from datetime import datetime, timedelta

import cv2
import numpy as np

import config as C

_INTRO = """These {n} frames cover {sec:.0f} seconds of video from a camera worn on the chest of the user \
(first-person view). Hands in the lower part of the image are the user's own hands.

List what the USER does with their hands or body in these {sec:.0f} seconds, for example: picks up, \
puts down, opens, closes, turns on, turns off, pours, eats, drinks, washes, cuts, wipes, presses, \
types, takes medicine. Ignore what other people do.

"""

PROMPT = _INTRO + """Answer with JSON only, in exactly this form:
{{"place": "<room or surface in view, e.g. kitchen counter>", "actions": [{{"verb": "<one verb>", "object": "<the object>", "where": "<where the object is or ends up, if visible>"}}]}}

If the user does nothing with their hands, use "actions": []. Do not invent objects you cannot see."""

PROMPT_STATES = _INTRO + """Answer with JSON only, in exactly this form:
{{"place": "<room or surface in view, e.g. kitchen counter>", "actions": [{{"verb": "<one verb>", "object": "<the object>", "where": "<where the object is or ends up, if visible>"}}], "states": [{{"object": "<name>", "state": "<open|closed|on|off>"}}]}}

"states": for each of these that is visible in the LAST frame, its state at that moment: fridge, freezer, \
oven, microwave, dishwasher, washing machine, door, drawer, cupboard (open or closed); tap, stove, light, \
tv (on or off). Leave out things you cannot see.
If the user does nothing with their hands, use "actions": []. Do not invent objects you cannot see."""

# v2 (VLM_FORMAT = "lines"): short lines instead of JSON (the VLM's time goes into WRITING, so fewer
# tokens = faster) and a fixed verb list, because free verbs ("moves plate", "transfers rice") neither
# answer "where did I put ...?" nor match annotations.
VERBS = ("picks up, puts down, opens, closes, turns on, turns off, pours, cuts, washes, wipes, stirs, eats, "
         "drinks, takes medicine, wears, takes off, presses, types, throws away")
_LINES_HEAD = """These {n} frames cover {sec:.0f} seconds of video from a camera worn on the chest of the user \
(first-person view). Hands in the lower part of the image are the user's own hands.

Report what the USER does with their hands or body in these {sec:.0f} seconds. Ignore what other people do.

Answer in exactly this format and nothing else:
PLACE: <room or surface in view, e.g. kitchen counter>
"""
_DID_LINE = "DID: <verb> | <object> | <where the object is or ends up, or ->\n"
_LINES_RULES = """
DID lines: one per action, in time order, at most """ + str(C.VLM_MAX_ACTIONS) + """, never the same line twice. \
The verb MUST be one of: """ + VERBS + """. Use the closest one (moving something to a new place = picks up, then \
puts down). No DID line if the user does nothing with their hands."""
_LINES_STATE_RULES = """
STATE lines: one for each of these that is visible in the LAST frame, with its state at that moment: fridge, \
freezer, oven, microwave, dishwasher, washing machine, door, drawer, cupboard (open or closed); tap, stove, light, \
tv (on or off)."""
_CHECK = """
CHECK: the user may have left these open or on: {objects}. For EACH of them write a STATE line \
(open/closed or on/off) as seen in the LAST frame, or "= not visible" if it is not in view."""
PROMPT_LINES = _LINES_HEAD + _DID_LINE + _LINES_RULES + "\nDo not invent objects you cannot see."
PROMPT_LINES_STATES = (_LINES_HEAD + "STATE: <object> = <open|closed|on|off>\n" + _DID_LINE + _LINES_STATE_RULES
                       + _LINES_RULES + "\nDo not invent objects you cannot see.")


def build_prompt(n: int, sec: float, track_states: bool = True, check: list[str] | None = None) -> str:
    if C.VLM_FORMAT != "lines":
        return (PROMPT_STATES if track_states else PROMPT).format(n=n, sec=sec)
    if not track_states:
        return PROMPT_LINES.format(n=n, sec=sec)
    p = PROMPT_LINES_STATES.format(n=n, sec=sec)
    if check:
        p += _CHECK.format(objects=", ".join(check))
    return p


def loop_or_full(text: str, max_actions: int = C.VLM_MAX_ACTIONS) -> bool:
    """Stop generating? True once a complete DID line repeats an earlier one, or max_actions is reached."""
    lines = [l.strip().lower() for l in text.split("\n")[:-1]]          # complete lines only
    did = [l for l in lines if l.startswith("did")]
    return len(did) >= max_actions or len(did) != len(set(did))

ASK_VIDEO = """These {n} frames cover {sec:.0f} seconds of video from a camera worn on the chest of the user \
(first-person view). Hands in the lower part of the image are the user's own hands.

QUESTION: {question}
Answer in one short sentence, starting with "Yes" or "No"."""

# ── object states ──
# Canonical names, most specific first ("fridge door" is the fridge, not a door).
_STATE_OBJECTS = [
    ("freezer", ("freezer",), "open"), ("fridge", ("fridge", "refrigerator"), "open"),
    ("dishwasher", ("dishwasher",), "both"), ("washing machine", ("washing machine", "washer"), "both"),
    ("microwave", ("microwave",), "both"), ("oven", ("oven",), "both"),
    ("drawer", ("drawer",), "open"), ("cupboard", ("cupboard", "cabinet", "closet", "wardrobe"), "open"),
    ("tap", ("tap", "faucet", "sink tap"), "on"), ("stove", ("stove", "hob", "burner", "cooktop", "gas"), "on"),
    ("tv", ("tv", "television"), "on"), ("light", ("light", "lamp"), "on"),
    ("door", ("door",), "open"),
]
_STATE_VALUE = {"open": "open", "opened": "open", "ajar": "open", "closed": "closed", "close": "closed",
                "shut": "closed", "on": "on", "running": "on", "lit": "on", "off": "off"}
_STATE_VERB = {"open": "opens", "closed": "closes", "on": "turns on", "off": "turns off"}


def canon_state_object(name: str | None) -> tuple[str, str] | None:
    """'Fridge door' → ('fridge', 'open'): canonical object and its kind ('open' = open/closed, 'on' = on/off,
    'both' = an appliance with a door and a power state)."""
    n = f" {(name or '').lower()} "
    for key, words, kind in _STATE_OBJECTS:
        if any(re.search(rf"\b{re.escape(w)}s?\b", n) for w in words):
            return key, kind
    return None


def canon_state(obj: str | None, state: str | None) -> tuple[str, str] | None:
    """('Refrigerator', 'Opened') → ('fridge', 'open'); ('oven', 'on') → ('oven power', 'on');
    None if unknown or mismatched (a door that is 'on')."""
    co = canon_state_object(obj)
    st = _STATE_VALUE.get((state or "").strip().lower())
    if not co or not st:
        return None
    key, kind = co
    door = st in ("open", "closed")
    if kind == "both":
        return (key, st) if door else (f"{key} power", st)
    if (kind == "open") != door:
        return None
    return key, st


def state_object_name(key: str) -> str:
    return key.removesuffix(" power")


class StateTracker:
    """Remembers the last seen state of each object; a change becomes an action row."""

    def __init__(self):
        self.last: dict[str, tuple[str, datetime]] = {}

    def update(self, states: list[dict], ts: datetime) -> list[dict]:
        """states: [{"object", "state"}] seen at the end of the window ending at ts → derived actions."""
        out = []
        for s in states:
            c = canon_state(s.get("object"), s.get("state"))
            if not c:
                continue
            key, st = c
            prev = self.last.get(key)
            if prev and prev[0] != st:
                out.append({"verb": _STATE_VERB[st], "object": state_object_name(key), "where": None,
                            "state_change": f"{key}: {prev[0]} → {st}"})
            self.last[key] = (st, ts)
        return out

    def to_check(self, ts: datetime, max_age_s: float = C.STATE_CHECK_MAX_AGE_S, k: int = 3) -> list[str]:
        """Objects believed open/on (seen in the last max_age_s), most recent first: ask the VLM about them."""
        act = [(t, key) for key, (st, t) in self.last.items()
               if st in ("open", "on") and (ts - t).total_seconds() <= max_age_s]
        out = []
        for _, key in sorted(act, key=lambda x: x[0], reverse=True):     # stable: ties keep report order
            name = state_object_name(key) + (" (on or off)" if key.endswith(" power") else "")
            if name not in out:
                out.append(name)
        return out[:k]

    def current(self) -> dict[str, tuple[str, datetime]]:
        return dict(self.last)


def _same_action(a: dict, b: dict) -> bool:
    """Does derived action a repeat something the VLM already wrote in b? ('opens', 'fridge door') vs ('opens', 'fridge')."""
    va, vb = (a.get("verb") or "").split()[:1], (b.get("verb") or "").split()[:1]
    if not va or va != vb:
        return False
    if (a.get("verb") or "").startswith("turn") and (a.get("verb") != b.get("verb")):
        return False
    ca = canon_state_object(a.get("object"))
    cb = canon_state_object(b.get("object"))
    return bool(ca and cb and ca[0] == cb[0])


# ── parsing ──

def _clean(x) -> str | None:
    return str(x or "").strip().lower() or None


_NONE = {"", "-", "->", "none", "n/a", "unknown", "null"}


def _parse_lines(text: str, out: dict) -> dict:
    for line in text.splitlines():
        m = re.match(r"\s*[-*]?\s*(PLACE|DID|STATE)\s*:\s*(.*)", line, re.I)
        if not m:
            continue
        kind, body = m.group(1).upper(), m.group(2).strip()
        if kind == "PLACE":
            out["place"] = None if body.lower() in _NONE else body
        elif kind == "DID":
            parts = [x.strip().lower() for x in body.split("|")] + ["", ""]
            parts[2] = parts[2].removeprefix("->").removesuffix("->").strip()   # "-> counter": the template's arrow
            if parts[0] not in _NONE:
                out["actions"].append({"verb": parts[0], "object": None if parts[1] in _NONE else parts[1],
                                       "where": None if parts[2] in _NONE else parts[2]})
        else:
            sm = re.match(r"(.+?)\s*[=:]\s*([A-Za-z]+)\s*$", body)
            if sm:
                out["states"].append({"object": sm.group(1).strip().lower(), "state": sm.group(2).lower()})
    return out


def _dedupe(out: dict) -> dict:
    """Same (verb, object) twice in one window is a generation loop, not two actions: keep the first."""
    seen, acts = set(), []
    for a in out["actions"]:
        key = (a["verb"], a["object"])
        if key not in seen:
            seen.add(key)
            acts.append(a)
    out["actions"] = acts[:C.VLM_MAX_ACTIONS]
    return out


def parse_output(text: str) -> dict:
    """Extract {"place", "actions": [{verb, object, where}], "states": [{object, state}], "raw"} from
    model text: the line format or v1 JSON. Tolerates code fences, extra words and answers cut off
    mid-way; never raises."""
    return _dedupe(_parse_any(text))


def _parse_any(text: str) -> dict:
    out = {"place": None, "actions": [], "states": [], "raw": text.strip()}
    if re.search(r"^\s*[-*]?\s*(PLACE|DID|STATE)\s*:", text, re.I | re.M):
        return _parse_lines(text, out)
    m = re.search(r"\{.*\}", text, re.S)
    d = None
    if m:
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            d = None
    if isinstance(d, dict):
        place = d.get("place")
        out["place"] = (str(place).strip() or None) if place else None
        actions, states = d.get("actions") or [], d.get("states") or []
    else:
        # broken/cut-off JSON: keep every complete {...} item that has a verb or a state
        pm = re.search(r'"place"\s*:\s*"([^"]*)"', text)
        out["place"] = (pm.group(1).strip() or None) if pm else None
        actions, states = [], []
        for item in re.findall(r"\{[^{}]*\}", text):
            try:
                x = json.loads(item)
            except json.JSONDecodeError:
                continue
            if isinstance(x, dict) and "verb" in x:
                actions.append(x)
            elif isinstance(x, dict) and "state" in x:
                states.append(x)
    for a in actions if isinstance(actions, list) else []:
        if not isinstance(a, dict) or not a.get("verb"):
            continue
        out["actions"].append({"verb": _clean(a.get("verb")), "object": _clean(a.get("object")),
                               "where": _clean(a.get("where"))})
    for s in states if isinstance(states, list) else []:
        if isinstance(s, dict) and s.get("object") and s.get("state"):
            out["states"].append({"object": _clean(s["object"]), "state": _clean(s["state"])})
    return out


# ── the model ──

class VLMDescriber:
    """Wraps a Hugging Face vision-language model (Qwen3-VL by default)."""

    def __init__(self, model_id: str = C.VLM_MODEL, model=None, processor=None, device: str | None = None,
                 track_states: bool = C.VLM_TRACK_STATES):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.track_states = track_states
        self.processor = processor or AutoProcessor.from_pretrained(model_id)
        if model is None:
            cuda = torch.cuda.is_available()
            # float16 on GPU (the T4 has no bfloat16 support); float32 on CPU
            model = AutoModelForImageTextToText.from_pretrained(
                model_id, dtype=torch.float16 if cuda else torch.float32,
                device_map="auto" if cuda else None)
        self.model = model.eval()
        self.device = next(self.model.parameters()).device

    def _generate(self, messages, videos=None, metadata=None, max_new_tokens=C.VLM_MAX_NEW_TOKENS,
                  stop_on_loop: bool = False) -> str:
        from transformers import StoppingCriteria, StoppingCriteriaList
        from transformers.video_utils import VideoMetadata
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        kwargs = dict(text=[text], return_tensors="pt")
        if videos is not None:
            kwargs.update(videos=videos, do_sample_frames=False,
                          video_metadata=[VideoMetadata(**m) for m in metadata])
        inputs = self.processor(**kwargs).to(self.device)
        n0, proc, torch = inputs["input_ids"].shape[1], self.processor, self.torch

        class _Loop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kw):
                done = loop_or_full(proc.decode(input_ids[0, n0:], skip_special_tokens=True))
                return torch.full((input_ids.shape[0],), done, dtype=torch.bool, device=input_ids.device)

        extra = {"stopping_criteria": StoppingCriteriaList([_Loop()])} if stop_on_loop else {}
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, **extra)
        return self.processor.decode(out[0, n0:], skip_special_tokens=True)

    def _video_call(self, frames_bgr: list[np.ndarray], seconds: float, prompt_fn, max_new_tokens,
                    stop_on_loop: bool = False) -> str:
        if len(frames_bgr) % 2:
            frames_bgr = frames_bgr + [frames_bgr[-1]]          # the model pairs frames in time
        h, w = frames_bgr[0].shape[:2]
        size = (C.VLM_FRAME_W, round(h * C.VLM_FRAME_W / w))
        video = np.stack([cv2.cvtColor(cv2.resize(f, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                          for f in frames_bgr])
        fps = len(frames_bgr) / max(seconds, 1e-3)
        msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt_fn(len(frames_bgr))}]}]
        meta = {"total_num_frames": len(frames_bgr), "fps": fps, "frames_indices": list(range(len(frames_bgr)))}
        return self._generate(msgs, videos=[video], metadata=[meta], max_new_tokens=max_new_tokens,
                              stop_on_loop=stop_on_loop)

    def describe(self, frames_bgr: list[np.ndarray], seconds: float, check: list[str] | None = None) -> str:
        """frames_bgr: evenly spaced frames covering `seconds` of video. check: objects whose state must be
        reported (StateTracker.to_check)."""
        return self._video_call(frames_bgr, seconds,
                                lambda n: build_prompt(n, seconds, self.track_states, check),
                                C.VLM_MAX_NEW_TOKENS, stop_on_loop=C.VLM_FORMAT == "lines")

    def ask_video(self, frames_bgr: list[np.ndarray], seconds: float, question: str) -> str:
        """Ask a question directly about the frames (the no-memory baseline in the QA benchmark)."""
        return self._video_call(frames_bgr, seconds,
                                lambda n: ASK_VIDEO.format(n=n, sec=seconds, question=question), 60)

    def chat(self, prompt: str, max_new_tokens: int = 300) -> str:
        """Text-only question (used by stage 2 to answer from the log)."""
        msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        return self._generate(msgs, max_new_tokens=max_new_tokens)


# ── the logger ──

class ActionLogger:
    """Runs the describer over a video and writes activity rows to the store."""

    def __init__(self, describer, store=None, window_s: float = C.VLM_WINDOW_S, frames: int = C.VLM_FRAMES,
                 track_states: bool = C.VLM_TRACK_STATES):
        self.describer = describer
        self.store = store
        self.window_s = window_s
        self.frames = frames
        self.track_states = track_states
        self.windows: list[dict] = []          # raw VLM answer per window (for replay())
        self.tracker = StateTracker()

    def run(self, source, start_time: datetime | None = None, verbose: bool = True,
            max_seconds: float | None = None, call: np.ndarray | None = None,
            scores: np.ndarray | None = None) -> tuple[list[dict], dict]:
        """call: optional bool per window (from hoi_gate.gate_decisions); False = skip the VLM there.
        scores: optional trigger score per window, only printed."""
        from modules.video_source import get_frames
        start_time = start_time or datetime.now()
        fps = self.frames / self.window_s
        rows, window, t_run, busy = [], [], time.time(), 0.0
        self.windows, self.tracker = [], StateTracker()
        last_t = 0.0
        for frame in get_frames(source, start_time, fps=fps):
            if max_seconds is not None and frame.t >= max_seconds:
                break
            window.append(frame)
            last_t = frame.t
            if len(window) == self.frames:
                busy += self._window(window, rows, verbose, call, scores)
                window = []
        if len(window) >= 2:                                  # leftover tail of the video
            busy += self._window(window, rows, verbose, call, scores)
        video_s = last_t + 1.0 / fps
        n_called = sum(w["called"] for w in self.windows)
        stats = {"windows": len(self.windows), "vlm_calls": n_called, "video_seconds": round(video_s, 1),
                 "model_seconds": round(busy, 1), "wall_seconds": round(time.time() - t_run, 1),
                 # < 1.0 means faster than real time
                 "real_time_factor": round(busy / video_s, 2) if video_s else None,
                 "rows": len(rows)}
        return rows, stats

    def _window(self, window, rows, verbose, call, scores) -> float:
        k = len(self.windows)
        seconds = window[-1].t - window[0].t + self.window_s / self.frames
        rec = {"k": k, "t": window[0].t, "ts": window[0].ts, "seconds": seconds, "text": None, "dt": 0.0,
               "called": True}
        if call is not None and k < len(call) and not call[k]:
            rec["called"] = False
            self.windows.append(rec)
            if verbose:
                sc = f" (trigger {scores[k]:.2f})" if scores is not None and k < len(scores) else ""
                print(f"  [{window[0].t:6.1f}s]  --  | skipped: no hand interaction{sc}", flush=True)
            return 0.0
        check = self.tracker.to_check(window[0].ts) if self.track_states else []
        rec["check"] = check
        t0 = time.time()
        frames = [f.image for f in window]
        # describers without the check option (tests, older code) are called the old way
        rec["text"] = (self.describer.describe(frames, seconds, check=check) if check
                       else self.describer.describe(frames, seconds))
        rec["dt"] = time.time() - t0
        self.windows.append(rec)
        new, parsed = ingest(rec, rows, self.tracker if self.track_states else None, self.store)
        if verbose:
            acts = "; ".join(f"{r['verb']} {r['object'] or ''}".strip() + (" [state]" if r.get("source") == "state" else "")
                             for r in new if r["verb"]) or "-"
            st = " ".join(f"{c[0]}={c[1]}" for c in (canon_state(x["object"], x["state"]) for x in parsed["states"]) if c)
            print(f"  [{window[0].t:6.1f}s] {rec['dt']:4.1f}s | place: {parsed['place'] or '?':18s} | {acts}"
                  + (f"  | states: {st}" if st else ""), flush=True)
        return rec["dt"]


def ingest(rec: dict, rows: list[dict], tracker: StateTracker | None, store=None) -> tuple[list[dict], dict]:
    """One described window → new rows appended to `rows` (and the store). Shared by run() and replay()."""
    parsed = parse_output(rec["text"] or "")
    ts0, ts1 = rec["ts"], rec["ts"] + timedelta(seconds=rec["seconds"])
    actions = [dict(a, source="vlm") for a in parsed["actions"]]
    if tracker is not None:
        for d in tracker.update(parsed["states"], ts1):
            if not any(_same_action(d, a) for a in actions):
                actions.append(dict(d, source="state"))
    new = []
    for a in actions or [{"verb": None, "object": None, "where": None, "source": "vlm"}]:
        row = {"ts_start": ts0, "ts_end": ts1, "verb": a["verb"], "object": a["object"],
               "where": a["where"], "place": parsed["place"],
               "raw": a.get("state_change") or parsed["raw"], "t": rec["t"], "t_end": rec["t"] + rec["seconds"],
               "source": a["source"]}
        # same action continuing from the previous window → extend it instead of a new row
        prev = rows[-1] if rows else None
        if (prev and row["verb"] and prev["verb"] == row["verb"] and prev["object"] == row["object"]
                and (row["ts_start"] - prev["ts_end"]).total_seconds() <= 1.0):
            prev["ts_end"], prev["t_end"] = ts1, row["t_end"]
            if store is not None and "id" in prev:
                store.extend_activity(prev["id"], ts1)
            continue
        if store is not None:
            row["id"] = store.add_activity(row, source=row["source"])
        rows.append(row)
        new.append(row)
    return new, parsed


def replay(windows: list[dict], call: np.ndarray | None = None, track_states: bool = True,
           store=None) -> list[dict]:
    """Rebuild the log from saved raw answers, as if only windows with call[k] had been described."""
    rows, tracker = [], (StateTracker() if track_states else None)
    for rec in windows:
        if rec["text"] is None or (call is not None and not call[rec["k"]]):
            continue
        ingest(rec, rows, tracker, store)
    return rows
