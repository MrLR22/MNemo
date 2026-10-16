# evaluation/qa_benchmark.py
# EXPERIMENT B — does Mnemo answer memory questions correctly, and does it avoid inventing things?
#
# Questions are generated automatically from the EPIC ground truth of each clip:
#   "yes" questions   from true actions in the clip           ("Did I open the fridge?")
#   "no"  questions   about home objects that never appear in the clip at all ("Did I open the oven?")
# A "no" object must not occur in ANY true action of the clip (checked via EPIC noun classes), so
# a "yes" answer to it is a hallucination, not an annotation gap.
#
# Systems compared on the same questions:
#   mnemo    stage 2: question → matching log rows → LLM answers only from them, or abstains
#   direct   no memory: the VLM gets 32 frames of the whole clip plus the question
#            (what "just ask a big video model" gives you)
#
# Answers are read as YES (claims it happened) or NO (says no / "I didn't see that").

import random
import re

YES_VERBS = {"open": "open", "close": "close", "turn-on": "turn on", "turn-off": "turn off",
             "take": "pick up", "put": "put down", "wash": "wash", "pour": "pour", "cut": "cut"}
_PRIORITY = ["open", "close", "turn-on", "turn-off", "wash", "pour", "cut", "take", "put"]

# (question, noun words used to check the object is absent from the clip)
NO_POOL = [
    ("Did I open the oven?", ["oven"]), ("Did I close the oven?", ["oven"]),
    ("Did I open the microwave?", ["microwave"]), ("Did I open the dishwasher?", ["dishwasher"]),
    ("Did I open the washing machine?", ["washing machine"]), ("Did I take my medicine?", ["medicine", "pill"]),
    ("Did I turn on the TV?", ["tv", "television"]), ("Did I pick up my phone?", ["phone"]),
    ("Did I open the front door?", ["door"]), ("Did I wash the plate?", ["plate"]),
    ("Did I cut the onion?", ["onion"]), ("Did I pour the milk?", ["milk"]),
    ("Did I open the fridge?", ["fridge"]), ("Did I close the fridge?", ["fridge"]),
    ("Did I turn on the tap?", ["tap"]), ("Did I pick up the knife?", ["knife"]),
    ("Did I turn on the kettle?", ["kettle"]), ("Did I open the bin?", ["bin"]),
    ("Did I put down my keys?", ["keys"]), ("Did I pick up the cup?", ["cup"]),
    # everyday home actions outside the kitchen (never in EPIC, so the answer must be no)
    ("Did I brush my teeth?", ["toothbrush", "tooth", "teeth"]), ("Did I put on my shoes?", ["shoe"]),
    ("Did I water the plants?", ["plant"]), ("Did I pick up my wallet?", ["wallet"]),
    ("Did I open my laptop?", ["laptop"]), ("Did I comb my hair?", ["comb", "hair"]),
    ("Did I make the bed?", ["bed"]), ("Did I lock the car?", ["car"]),
    ("Did I read the newspaper?", ["newspaper"]), ("Did I iron my shirt?", ["shirt", "iron"]),
]

_NO_PATTERNS = re.compile(r"didn['’]t see|did not see|no record|not in (my|the) (records|log)|"
                          r"^\W*no\b|i don['’]t know|cannot (tell|see|determine)|can['’]t (tell|see)|"
                          r"not (visible|shown|clear)|there is no", re.I)


def says_yes(answer: str) -> bool:
    return not _NO_PATTERNS.search((answer or "").strip())


def make_questions(truth: list[dict], vocab, n_yes: int = 6, n_no: int = 6, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    pairs = {}
    for g in truth:
        vk = vocab.verb_key[g["verb_class"]]
        if vk in YES_VERBS:
            pairs.setdefault((vk, g["noun_class"]), g["start"])
    order = sorted(pairs, key=lambda p: (_PRIORITY.index(p[0]), pairs[p]))
    state = [p for p in order if p[0] in ("open", "close", "turn-on", "turn-off")]
    other = [p for p in order if p not in state]
    pick = state[:n_yes // 2] + other[:n_yes - min(len(state), n_yes // 2)]   # half open/close, half other
    pick += [p for p in state if p not in pick][:n_yes - len(pick)]
    yes = [{"q": f"Did I {YES_VERBS[v]} the {vocab.noun_text(n)}?", "expect": True, "t": pairs[(v, n)]}
           for v, n in pick]
    present = set().union(*(g["nouns"] for g in truth)) if truth else set()
    narr = " ".join(g["narration"] for g in truth).lower()
    no = []
    for q, words in rng.sample(NO_POOL, len(NO_POOL)):
        classes = set().union(*(vocab.noun_classes(w) for w in words))
        if classes & present or any(re.search(rf"\b{w}", narr) for w in words):
            continue
        no.append({"q": q, "expect": False})
        if len(no) == n_no:
            break
    return yes + no


def score_answers(items: list[dict]) -> dict:
    """items: [{"expect": bool, "yes": bool}] → yes-recall, hallucination rate, accuracy."""
    pos = [x for x in items if x["expect"]]
    neg = [x for x in items if not x["expect"]]
    tp = sum(x["yes"] for x in pos)
    fp = sum(x["yes"] for x in neg)
    return {"yes_questions": len(pos), "no_questions": len(neg),
            "yes_answered_yes": tp, "no_answered_yes": fp,
            "recall": tp / len(pos) if pos else None,
            "hallucination": fp / len(neg) if neg else None,
            "accuracy": (tp + len(neg) - fp) / len(items) if items else None}


def ask_mnemo(questions: list[dict], store, llm, now) -> list[dict]:
    from query.llm_qa import answer
    out = []
    for q in questions:
        a = answer(q["q"], store, llm=llm, now=now)["answer"]
        out.append(dict(q, answer=a, yes=says_yes(a)))
    return out


def clip_frames(video: str, n: int = 32):
    """n evenly spaced frames of the whole clip (+ its duration in seconds)."""
    import cv2
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    for i in [round(k * (total - 1) / (n - 1)) for k in range(n)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    return frames, total / fps


def ask_direct(questions: list[dict], describer, video: str, n_frames: int = 32) -> list[dict]:
    frames, seconds = clip_frames(video, n_frames)
    out = []
    for q in questions:
        a = describer.ask_video(frames, seconds, q["q"])
        out.append(dict(q, answer=a, yes=a.strip().lower().startswith("yes")))
    return out
