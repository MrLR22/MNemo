# modules/vlm_logger.py
# STAGE 1: video → action log, using a vision-language model (VLM).
#
# Every VLM_WINDOW_S seconds, VLM_FRAMES frames go to the VLM with one question: "what does the
# camera wearer do with their hands?". The answer is parsed into rows
#   time | verb | object | place | where-it-ended-up
# and stored as TEXT in the activities table. Frames stay in RAM only (privacy, as before).
# Unlike the pick/put pipeline this is not limited to a fixed list of actions or objects.

import json
import re
import time
from datetime import datetime, timedelta

import cv2
import numpy as np

import config as C

PROMPT = """These {n} frames cover {sec:.0f} seconds of video from a camera worn on the chest of the user \
(first-person view). Hands in the lower part of the image are the user's own hands.

List what the USER does with their hands or body in these {sec:.0f} seconds, for example: picks up, \
puts down, opens, closes, turns on, turns off, pours, eats, drinks, washes, cuts, wipes, presses, \
types, takes medicine. Ignore what other people do.

Answer with JSON only, in exactly this form:
{{"place": "<room or surface in view, e.g. kitchen counter>", "actions": [{{"verb": "<one verb>", "object": "<the object>", "where": "<where the object is or ends up, if visible>"}}]}}

If the user does nothing with their hands, use "actions": []. Do not invent objects you cannot see."""


def parse_output(text: str) -> dict:
    """Extract {"place": str|None, "actions": [ {verb, object, where} ]} from model text.
    Tolerates code fences and extra words around the JSON; never raises."""
    out = {"place": None, "actions": [], "raw": text.strip()}
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return out
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return out
    if not isinstance(d, dict):
        return out
    place = d.get("place")
    out["place"] = (str(place).strip() or None) if place else None
    for a in d.get("actions") or []:
        if not isinstance(a, dict) or not a.get("verb"):
            continue
        out["actions"].append({
            "verb": str(a.get("verb", "")).strip().lower(),
            "object": str(a.get("object", "") or "").strip().lower() or None,
            "where": str(a.get("where", "") or "").strip().lower() or None,
        })
    return out


class VLMDescriber:
    """Wraps a Hugging Face vision-language model (Qwen3-VL by default)."""

    def __init__(self, model_id: str = C.VLM_MODEL, model=None, processor=None, device: str | None = None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.processor = processor or AutoProcessor.from_pretrained(model_id)
        if model is None:
            cuda = torch.cuda.is_available()
            # float16 on GPU (the T4 has no bfloat16 support); float32 on CPU
            model = AutoModelForImageTextToText.from_pretrained(
                model_id, dtype=torch.float16 if cuda else torch.float32,
                device_map="auto" if cuda else None)
        self.model = model.eval()
        self.device = next(self.model.parameters()).device

    def _generate(self, messages, videos=None, metadata=None, max_new_tokens=C.VLM_MAX_NEW_TOKENS) -> str:
        from transformers.video_utils import VideoMetadata
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        kwargs = dict(text=[text], return_tensors="pt")
        if videos is not None:
            kwargs.update(videos=videos, do_sample_frames=False,
                          video_metadata=[VideoMetadata(**m) for m in metadata])
        inputs = self.processor(**kwargs).to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def describe(self, frames_bgr: list[np.ndarray], seconds: float) -> str:
        """frames_bgr: evenly spaced frames covering `seconds` of video."""
        if len(frames_bgr) % 2:
            frames_bgr = frames_bgr + [frames_bgr[-1]]          # the model pairs frames in time
        h, w = frames_bgr[0].shape[:2]
        size = (C.VLM_FRAME_W, round(h * C.VLM_FRAME_W / w))
        video = np.stack([cv2.cvtColor(cv2.resize(f, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                          for f in frames_bgr])
        fps = len(frames_bgr) / max(seconds, 1e-3)
        prompt = PROMPT.format(n=len(frames_bgr), sec=seconds)
        msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
        meta = {"total_num_frames": len(frames_bgr), "fps": fps, "frames_indices": list(range(len(frames_bgr)))}
        return self._generate(msgs, videos=[video], metadata=[meta])

    def chat(self, prompt: str, max_new_tokens: int = 300) -> str:
        """Text-only question (used by stage 2 to answer from the log)."""
        msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        return self._generate(msgs, max_new_tokens=max_new_tokens)


class ActionLogger:
    """Runs the describer over a video and writes activity rows to the store."""

    def __init__(self, describer, store=None, window_s: float = C.VLM_WINDOW_S, frames: int = C.VLM_FRAMES):
        self.describer = describer
        self.store = store
        self.window_s = window_s
        self.frames = frames

    def run(self, source, start_time: datetime | None = None, verbose: bool = True,
            max_seconds: float | None = None) -> tuple[list[dict], dict]:
        from modules.video_source import get_frames
        start_time = start_time or datetime.now()
        fps = self.frames / self.window_s
        rows, window, t_run, n_windows, busy = [], [], time.time(), 0, 0.0
        last_t = 0.0
        for frame in get_frames(source, start_time, fps=fps):
            if max_seconds is not None and frame.t >= max_seconds:
                break
            window.append(frame)
            last_t = frame.t
            if len(window) == self.frames:
                busy += self._window(window, rows, verbose)
                n_windows += 1
                window = []
        if len(window) >= 2:                                  # leftover tail of the video
            busy += self._window(window, rows, verbose)
            n_windows += 1
        video_s = last_t + 1.0 / fps
        stats = {"windows": n_windows, "video_seconds": round(video_s, 1),
                 "model_seconds": round(busy, 1), "wall_seconds": round(time.time() - t_run, 1),
                 # < 1.0 means faster than real time
                 "real_time_factor": round(busy / video_s, 2) if video_s else None,
                 "rows": len(rows)}
        return rows, stats

    def _window(self, window, rows, verbose) -> float:
        t0 = time.time()
        seconds = window[-1].t - window[0].t + self.window_s / self.frames
        text = self.describer.describe([f.image for f in window], seconds)
        dt = time.time() - t0
        parsed = parse_output(text)
        ts0, ts1 = window[0].ts, window[0].ts + timedelta(seconds=seconds)
        new = []
        for a in parsed["actions"] or [{"verb": None, "object": None, "where": None}]:
            row = {"ts_start": ts0, "ts_end": ts1, "verb": a["verb"], "object": a["object"],
                   "where": a["where"], "place": parsed["place"], "raw": parsed["raw"],
                   "t": window[0].t}
            # same action continuing from the previous window → extend it instead of a new row
            prev = rows[-1] if rows else None
            if (prev and row["verb"] and prev["verb"] == row["verb"] and prev["object"] == row["object"]
                    and (row["ts_start"] - prev["ts_end"]).total_seconds() <= 1.0):
                prev["ts_end"] = ts1
                if self.store:
                    self.store.extend_activity(prev["id"], ts1)
                continue
            if self.store:
                row["id"] = self.store.add_activity(row, source="vlm")
            rows.append(row)
            new.append(row)
        if verbose:
            acts = "; ".join(f"{r['verb']} {r['object'] or ''}".strip() for r in new if r["verb"]) or "-"
            print(f"  [{window[0].t:6.1f}s] {dt:4.1f}s | place: {parsed['place'] or '?':18s} | {acts}", flush=True)
        return dt
