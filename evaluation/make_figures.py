# evaluation/make_figures.py
# The report's three result charts, drawn from a finished Kaggle run (the folder of mnemo_results.zip):
#   fig_trigger.png    actions found vs share of model calls: our trigger vs every N-th window
#   fig_states.png     before / after the object-state memory
#   fig_timespan.png   accuracy over 1 minute vs a whole morning: memory vs asking the model directly
#
#   python -m evaluation.make_figures --results data/kaggle_results/v3_final --out docs/figures
#   python -m evaluation.make_figures --results data/kaggle_results/v3_final --out ../paper/figures --paper
# --paper: vector PDFs without the in-image title / footnote (the paper's captions carry them).

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation.log_accuracy import merge_scores
from evaluation.qa_benchmark import score_answers

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
OURS, BASE = "#2a78d6", "#eb6834"          # categorical slots 1 and 2 (validated, light mode)
PAPER = False


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0, labelsize=9.5)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def _title(fig, title, subtitle):
    if PAPER:
        return
    fig.text(0.04, 0.95, title, fontsize=12.5, fontweight="bold", color=INK, va="top")
    fig.text(0.04, 0.885, subtitle, fontsize=9.5, color=INK2, va="top")


def _dot(ax, x, y, color):
    ax.plot([x], [y], "o", ms=8, color=color, mec=SURFACE, mew=2, zorder=4)


def totals(results, system):
    sc = merge_scores([r["systems"][system]["score"] for r in results])
    calls = sum(r["systems"][system]["vlm_calls"] for r in results) / sum(r["systems"][system]["windows"] for r in results)
    return sc, calls


def fig_trigger(results, out):
    pts = []
    for th in ("0.8", "0.7", "0.6", "0.5"):
        a, calls = totals(results, f"mnemo@{th}")
        b, _ = totals(results, f"every-nth = mnemo@{th}")
        pts.append((calls * 100, a["found"], b["found"]))
    full, _ = totals(results, "+ states")
    pts.append((100.0, full["found"], full["found"]))
    xs = [p[0] for p in pts]
    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.09, right=0.8, top=0.74, bottom=0.14)
    _style(ax)
    ax.plot(xs, [p[2] for p in pts], color=BASE, lw=2, solid_capstyle="round", zorder=3)
    ax.plot(xs, [p[1] for p in pts], color=OURS, lw=2, solid_capstyle="round", zorder=3)
    for x, a, b in pts:
        _dot(ax, x, b, BASE)
        _dot(ax, x, a, OURS)
    x74 = next(p for p in pts if 70 <= p[0] <= 78)
    ax.annotate(f"+{x74[1] - x74[2]} actions vs a fixed rate\nat {x74[0]:.0f}% of calls", xy=(x74[0], x74[1]),
                xytext=(46, x74[1] + 2), fontsize=9, color=INK, va="center",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8, shrinkB=6))
    ax.text(xs[-1] + 1.5, pts[-1][1], f" full log\n {pts[-1][1]} actions", fontsize=9, color=INK2, va="center")
    ax.set_xlim(35, 104)
    ax.set_ylim(40, 102)
    ax.set_xlabel("Share of 4-second windows sent to the vision-language model (%)", fontsize=9.5, color=INK2)
    ax.set_ylabel("Annotated actions found (of 307)", fontsize=9.5, color=INK2)
    ax.plot([], [], color=OURS, lw=2, marker="o", ms=7, label="Our trigger (HOI-TCN)")
    ax.plot([], [], color=BASE, lw=2, marker="o", ms=7, label="Every N-th window, same calls")
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK)
    _title(fig, f"The trigger keeps {x74[1]} of {pts[-1][1]} actions at {x74[0]:.0f}% of model calls",
           f"A fixed rate with the same calls keeps {x74[2]}. Below about 60% of calls the fixed rate wins.\n"
           "20 random EPIC-KITCHENS minutes, 5 people never seen in training, 307 annotated actions.")
    fig.savefig(out, facecolor=SURFACE, **({"bbox_inches": "tight", "pad_inches": 0.03} if PAPER else {}))
    plt.close(fig)
    return pts


def fig_states(results, out):
    a, _ = totals(results, "captions")
    b, _ = totals(results, "+ states")
    qa = {s: score_answers([x for r in results for x in r["questions"][s]]) for s in ("captions", "+ states")}
    rows = [("Questions right (1-min clips)", qa["captions"]["accuracy"], qa["+ states"]["accuracy"]),
            ("All actions found", a["recall"], b["recall"]),
            ("Opens, closes, on/off found", a["state_recall"], b["state_recall"])]
    fig, ax = plt.subplots(figsize=(7.4, 3.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.3, right=0.95, top=0.7, bottom=0.17)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=1)
    for i, (name, before, after) in enumerate(rows):
        ax.plot([before * 100, after * 100], [i, i], color=GRID, lw=3, solid_capstyle="round", zorder=2)
        _dot(ax, before * 100, i, BASE)
        _dot(ax, after * 100, i, OURS)
        ax.text(before * 100 - 1.6, i, f"{before:.0%}", ha="right", va="center", fontsize=9, color=INK2)
        ax.text(after * 100 + 1.6, i, f"{after:.0%}", ha="left", va="center", fontsize=9, color=INK, fontweight="bold")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=9.5, color=INK)
    ax.set_xlim(15, 92)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("%", fontsize=9.5, color=INK2)
    ax.plot([], [], "o", color=BASE, ms=7, label="Plain captions")
    ax.plot([], [], "o", color=OURS, ms=7, label="+ our state memory with re-check")
    fig.legend(loc="upper left", bbox_to_anchor=(0.29, 0.83), ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    gain = b["state_found"] / a["state_found"] - 1
    _title(fig, f"State memory finds {gain:.0%} more opens and closes ({a['state_found']} to {b['state_found']} of {a['state_true']})",
           "Same model answers, every window described; 20 random EPIC-KITCHENS minutes, 5 unseen people.")
    fig.savefig(out, facecolor=SURFACE, **({"bbox_inches": "tight", "pad_inches": 0.03} if PAPER else {}))
    plt.close(fig)
    return rows


def fig_timespan(results, long_qa, out):
    short_m = score_answers([x for r in results for x in r["questions"]["mnemo@0.6"]])
    short_d = score_answers([x for r in results for x in r["questions"]["direct VLM (no memory)"]])
    q = long_qa["questions"]
    long_m = score_answers(q["mnemo"])
    long_d = score_answers(q[[k for k in q if k != "mnemo"][0]])
    when = long_qa["when"]
    when_ok = sum(w["correct"] for w in when)
    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.09, right=0.72, top=0.78, bottom=0.2)
    _style(ax)
    xs = [0, 1]
    for vals, color, name in ((
            [short_d["accuracy"], long_d["accuracy"]], BASE, "Ask the model directly"),
            ([short_m["accuracy"], long_m["accuracy"]], OURS, "Mnemo (memory log)")):
        ys = [v * 100 for v in vals]
        ax.plot(xs, ys, color=color, lw=2, solid_capstyle="round", zorder=3)
        for x, y in zip(xs, ys):
            _dot(ax, x, y, color)
        ax.text(-0.06, ys[0], f"{ys[0]:.0f}%", ha="right", va="center", fontsize=9, color=INK2)
        ax.text(1.06, ys[1], f"{ys[1]:.0f}%  {name}", ha="left", va="center", fontsize=9.5, color=INK,
                fontweight="bold" if color == OURS else "normal")
    ax.set_xticks(xs, [f"1 minute\n({short_m['yes_questions'] + short_m['no_questions']} questions)",
                       f"2.4-hour morning\n({long_m['yes_questions'] + long_m['no_questions']} questions)"], fontsize=9.5, color=INK)
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(55, 95)
    ax.set_ylabel("Questions answered correctly (%)", fontsize=9.5, color=INK2)
    if not PAPER:
        fig.text(0.04, 0.045, f"Morning only: Mnemo gave the right clock time for {when_ok} of {len(when)} \"When did I...?\" questions "
                 f"(the direct model cannot tell time). Invented answers: Mnemo {long_m['no_answered_yes']} of {long_m['no_questions']}, "
                 f"direct {long_d['no_answered_yes']} of {long_d['no_questions']}.", fontsize=8.5, color=INK2, wrap=True)
    _title(fig, "Over a whole morning, memory beats asking the model directly",
           "Direct model: 32 frames of the clip or the whole morning. Mnemo: its log (trigger at 0.6 + state memory).")
    fig.savefig(out, facecolor=SURFACE, **({"bbox_inches": "tight", "pad_inches": 0.03} if PAPER else {}))
    plt.close(fig)
    return (short_m, short_d, long_m, long_d, when_ok, len(when))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="unzipped mnemo_results.zip")
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--paper", action="store_true", help="PDFs without titles, for the paper")
    args = ap.parse_args()
    global PAPER
    PAPER = args.paper
    ext = "pdf" if PAPER else "png"
    os.makedirs(args.out, exist_ok=True)
    results = json.load(open(os.path.join(args.results, "benchmark.json")))
    long_qa = json.load(open(os.path.join(args.results, "long_qa.json")))
    print("trigger points (calls %, ours, every-nth):", fig_trigger(results, os.path.join(args.out, f"fig_trigger.{ext}")))
    print("states rows:", fig_states(results, os.path.join(args.out, f"fig_states.{ext}")))
    s = fig_timespan(results, long_qa, os.path.join(args.out, f"fig_timespan.{ext}"))
    print(f"timespan: 1 min mnemo {s[0]['accuracy']:.0%} direct {s[1]['accuracy']:.0%} | morning mnemo {s[2]['accuracy']:.0%} "
          f"direct {s[3]['accuracy']:.0%} | when {s[4]}/{s[5]}")
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
